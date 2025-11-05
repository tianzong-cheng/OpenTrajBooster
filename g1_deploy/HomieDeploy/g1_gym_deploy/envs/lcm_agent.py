import time

import lcm
import numpy as np
import torch
from lcm_types.arm_action_lcmt import arm_action_lcmt
from lcm_types.body_record_lcmt import body_record_lcmt
from lcm_types.hand_action_lcmt import hand_action_lcmt
from lcm_types.pd_tau_targets_lcmt import pd_tau_targets_lcmt
from utils.cheetah_state_estimator import StateEstimator
from utils.command_profile import RCControllerProfile

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

import json
import pickle
import queue
import socket
import threading
import time

import zmq

latest_sol_q = None
command_g = None
data_lock = threading.Lock()


def zmq_listener():
    global latest_sol_q
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://192.168.123.9:5556")
    socket.setsockopt_string(zmq.SUBSCRIBE, "avp_arm_data")

    while True:
        msg = socket.recv_string()
        topic, data_json = msg.split(" ", 1)
        sol_q = np.array(json.loads(data_json))

        # 更新共享变量（带锁）
        with data_lock:
            latest_sol_q = sol_q


# def zmq_publisher():
#     global command_g
#     context = zmq.Context()
#     socket = context.socket(zmq.PUB)
#     socket.bind("tcp://*:5557")  # 绑定到端口

#     while True:
#         # 获取数据（带锁保护）
#         with data_lock:
#             current_command = command_g  # 获取最新命令的副本

#         if current_command is not None:
#             # 构造数据包
#             leg_q_np = np.array(current_command)
#             leg_q_json = json.dumps(leg_q_np.tolist())
#             zmq_msg = f"avp_leg_command {leg_q_json}"

#             # 发布数据
#             socket.send_string(zmq_msg)
#             print(f"[ZMQ Published] {zmq_msg}")
#             time.sleep(0.1)


class LCMAgent:
    def __init__(self, se: StateEstimator, command_profile: RCControllerProfile):
        self.se = se
        self.command_profile = command_profile

        self.dt = 1 / 50
        self.timestep = 0

        self.num_envs = 1
        self.num_dofs = 27
        self.num_obs = 2 * self.num_dofs + 10 + 12  # 91
        self.num_history_length = 6
        self.num_lower_dofs = 12
        self.num_commands = 4
        self.device = "cuda:0"

        self.default_dof_pos = np.array(
            [
                -0.1000,
                0.0000,
                0.0000,
                0.3000,
                -0.2000,
                0.0000,
                -0.1000,
                0.0000,
                0.0000,
                0.3000,
                -0.2000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                -0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ],
            dtype=np.float,
        )
        self.p_gains = np.array(
            [
                150.0,
                150.0,
                150.0,
                300.0,
                40.0,
                40.0,
                150.0,
                150.0,
                150.0,
                300.0,
                40.0,
                40.0,
                300.0,
                200.0,
                200.0,
                200.0,
                100.0,
                20.0,
                20.0,
                20.0,
                200.0,
                200.0,
                200.0,
                100.0,
                20.0,
                20.0,
                20.0,
            ],
            dtype=np.float,
        )
        self.d_gains = np.array(
            [
                2.0000,
                2.0000,
                2.0000,
                4.0000,
                4.0000,
                4.0000,
                2.0000,
                2.0000,
                2.0000,
                4.0000,
                4.0000,
                4.0000,
                5.0000,
                4.0000,
                4.0000,
                4.0000,
                1.0000,
                0.5000,
                0.5000,
                0.5000,
                4.0000,
                4.0000,
                4.0000,
                1.0000,
                0.5000,
                0.5000,
                0.5000,
            ],
            dtype=np.float,
        )

        self.torque_limit = np.array(
            [
                88.0,
                88.0,
                88.0,
                139.0,
                50.0,
                50.0,
                88.0,
                88.0,
                88.0,
                139.0,
                50.0,
                50.0,
                88.0,
                25.0,
                25.0,
                25.0,
                25.0,
                25.0,
                5.0,
                5.0,
                25.0,
                25.0,
                25.0,
                25.0,
                25.0,
                5.0,
                5.0,
            ]
        )
        self.commands = np.zeros((1, self.num_commands))
        self.actions = torch.zeros(self.num_lower_dofs)
        self.last_actions = torch.zeros(self.num_lower_dofs)  # TODO
        self.gravity_vector = np.zeros(3)
        self.dof_pos = np.zeros(self.num_dofs)
        self.dof_vel = np.zeros(self.num_dofs)
        self.body_angular_vel = np.zeros(3)
        self.joint_pos_target = np.zeros(self.num_dofs + 2)
        self.torques = np.zeros(self.num_dofs + 2)

        self.joint_idxs = self.se.joint_idxs

        self.conn = None
        self.server_socket = None

        # 启动订阅线程
        self.listener_thread = threading.Thread(target=zmq_listener, daemon=True)
        self.listener_thread.start()

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind("tcp://*:5557")  # 绑定到端口

        # self.publisher_thread = threading.Thread(target=zmq_publisher, daemon=True)
        # self.publisher_thread.start()

    def close_connections(self):
        """Clean up network resources"""
        if self.conn:
            self.conn.close()
            self.conn = None
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None

    def __del__(self):
        self.close_connections()

    def get_obs(self):
        self.gravity_vector = self.se.get_gravity_vector()
        cmds = self.command_profile.get_command(self.timestep * self.dt)
        self.commands[:, :] = cmds[: self.num_commands]

        current_command = self.commands[:, :]
        leg_q_np = np.array(current_command)
        leg_q_json = json.dumps(leg_q_np.tolist())
        zmq_msg = f"avp_leg_command {leg_q_json}"

        # 发布数据
        self.socket.send_string(zmq_msg)
        print(f"[ZMQ Published] {zmq_msg}")

        # self.commands[:, :] = self.se.get_command()
        self.dof_pos = self.se.get_dof_pos()
        # body_record = body_record_lcmt()
        # body_record.q = self.dof_posf
        # lc.publish("body_data_record", body_record.encode())
        self.dof_vel = self.se.get_dof_vel()
        self.body_angular_vel = self.se.get_body_angular_vel()
        actions = torch.cat((self.actions.reshape(1, -1).to("cuda:0"), torch.zeros(1, 15).to("cuda:0")), dim=-1).to(
            "cuda:0"
        )
        print("commands: ", self.commands[:, :])
        print("ang_vel: ", self.body_angular_vel)
        print("grav: ", self.gravity_vector)
        print("dof_pos: ", self.dof_pos)
        print("dof_vel: ", self.dof_vel)
        print("action: ", actions)
        ob = np.concatenate(
            (
                self.commands[:, :] * np.array([2.0, 2.0, 0.25, 1.0]),  # 4
                self.body_angular_vel.reshape(1, -1) * 0.5,  # 3
                self.gravity_vector.reshape(1, -1),  # 3
                (self.dof_pos - self.default_dof_pos).reshape(1, -1),
                self.dof_vel.reshape(1, -1) * 0.05,
                actions.cpu().detach().numpy().reshape(1, -1)[:, :12],
            ),
            axis=1,
        )
        return torch.tensor(ob, device=self.device).float()

    def publish_action(self, action, hard_reset=False):
        action = action.cpu().numpy()
        command_for_robot = pd_tau_targets_lcmt()
        scaled_pos_target = action * 0.25 + self.default_dof_pos[:12]
        self.joint_pos_target[:12] = scaled_pos_target[:12]

        with data_lock:
            current_sol_q = latest_sol_q  # 拿最新值或空值

        if current_sol_q is not None:
            print(f"[50Hz Loop] Using sol_q: {current_sol_q}")
            self.joint_pos_target[15:] = current_sol_q
        else:
            print("[50Hz Loop] No data yet, waiting...")
            current_sol_q = self.default_dof_pos[13:]  # fallback
            self.joint_pos_target[15:] = current_sol_q

        command_for_robot.q_des = self.joint_pos_target
        command_for_robot.tau_ff = self.torques
        command_for_robot.timestamp_us = int(time.time() * 10**6)
        lc.publish("pd_plustau_targets", command_for_robot.encode())

        arm_action = arm_action_lcmt()
        arm_action.act = current_sol_q
        lc.publish("arm_action", arm_action.encode())

        # hand_action = hand_action_lcmt()
        # hand_action.act = sol_q_hand
        # lc.publish("hand_action", hand_action.encode())

    # def publish_action(self, action, hard_reset=False):
    #     action = action.cpu().numpy()
    #     command_for_robot = pd_tau_targets_lcmt()
    #     scaled_pos_target = action * 0.25 + self.default_dof_pos[:12]
    #     # torques = (scaled_pos_target - self.dof_pos[:12]) * self.p_gains[:12]  - self.dof_vel[:12] * self.d_gains[:12]
    #     # torques = np.clip(torques[:12], -self.torque_limit[:12], self.torque_limit[:12])
    #     self.joint_pos_target[:12] = scaled_pos_target[:12]
    #     # arm_actions = self.se.get_arm_action()
    #     # self.joint_pos_target[15:] = 0.#arm_actions
    #     # self.joint_pos_target[12] = scaled_pos_target[12] # waist
    #     # self.joint_pos_target[15:] = scaled_pos_target[13:]
    #     # self.torques[:12] = torques[:12]
    #     # print("torques: ", torques)
    #     # print("==============================================================================")
    #     # self.torques[15:] = torques[13:]
    #     command_for_robot.q_des = self.joint_pos_target
    #     command_for_robot.tau_ff = self.torques
    #     command_for_robot.timestamp_us = int(time.time() * 10 ** 6)

    #     lc.publish("pd_plustau_targets", command_for_robot.encode())

    #     # arm_action = arm_action_lcmt()
    #     # arm_action.act = arm_actions
    #     # lc.publish("new_arm_action", arm_action.encode())

    def reset(self):
        self.actions = torch.zeros(12)
        self.time = time.time()
        self.timestep = 0
        return self.get_obs()

    def step(self, actions, hard_reset=False):
        clip_actions = 100.0
        self.last_actions = self.actions[:]
        self.actions = torch.clip(actions[0:1, :], -clip_actions, clip_actions)
        self.publish_action(self.actions, hard_reset=hard_reset)
        time.sleep(max(self.dt - (time.time() - self.time), 0))
        if self.timestep % 100 == 0:
            print(f"frq: {1 / (time.time() - self.time)} Hz")
        self.time = time.time()
        obs = self.get_obs()

        self.timestep += 1
        return obs
