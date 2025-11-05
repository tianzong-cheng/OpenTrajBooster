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

latest_cmd = None
data_lock = threading.Lock()


def zmq_listener():
    global latest_cmd
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://192.168.123.9:5556")
    socket.setsockopt_string(zmq.SUBSCRIBE, "action_cmd")

    while True:
        msg = socket.recv_string()
        topic, data_json = msg.split(" ", 1)
        cmd_dict = json.loads(data_json)

        # 更新共享变量（带锁）
        with data_lock:
            latest_cmd = cmd_dict


class LCMAgent:
    def __init__(
        self, se: StateEstimator, command_profile: RCControllerProfile, own_policy: bool = False, debug: bool = True
    ):
        self.se = se
        self.command_profile = command_profile

        self.dt = 1 / 50
        self.cmd_dt = 1 / 20  # 1/50
        self.timestep = 0

        self.num_envs = 1
        self.num_dofs = 27
        self.num_obs = 2 * self.num_dofs + 10 + 12  # 91
        self.num_history_length = 6
        self.num_lower_dofs = 12
        self.num_commands = 4
        self.device = "cuda:0"
        self.own_policy = own_policy
        self.debug = debug

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

        self.current_q_cmd = None
        self.prev_command = None
        self.current_arm_command = np.zeros(14)
        self.current_base_command = np.zeros(4)
        self.act_step = 0
        self.flag = 0

        # 在类的 __init__ 中
        self.cmd_lock = threading.Lock()
        ## control_loop线程：
        self.control_loop_thread = threading.Thread(target=self.command_get_loop, daemon=True)
        self.control_loop_thread.start()

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

    def command_get_loop(self):
        """Continuously fetch commands from the network"""
        while True:
            self.get_current_q_command()
            time.sleep(self.cmd_dt)

    def get_current_q_command(self):
        with data_lock:
            self.current_q_cmd = latest_cmd.copy() if latest_cmd is not None else None
        if self.current_q_cmd is not None:
            ### 解析
            with self.cmd_lock:
                if self.debug:
                    # print(f"[GR00T Start] Current command: {self.current_q_cmd}")
                    # print('arm cmd shape:', np.atleast_1d(self.current_q_cmd['arm_action'][self.act_step]).shape)
                    # self.current_arm_command = self.default_dof_pos[13:]
                    # print('base cmd shape:', np.atleast_1d(self.current_q_cmd['base_action'][self.act_step]).shape)
                    self.current_arm_command = np.atleast_1d(self.current_q_cmd["arm_action"][self.act_step])
                    self.current_base_command = np.atleast_1d(self.current_q_cmd["base_action"][self.act_step])
                else:
                    self.current_arm_command = np.atleast_1d(self.current_q_cmd["arm_action"][self.act_step])
                    self.current_base_command = np.atleast_1d(self.current_q_cmd["base_action"][self.act_step])
            self.act_step += 1
            if self.act_step == 14:
                self.flag = 1
            else:
                self.flag = 0

            # print('self.act_step:', self.act_step)

            if self.act_step > 15:
                self.act_step = 15
                print("Gr00t too slow")

            if self.current_q_cmd != self.prev_command:
                print("Difference Reset")
                self.act_step = 2

            self.prev_command = self.current_q_cmd.copy()
            zmq_msg = f"infer_start {self.flag}"
            self.socket.send_string(zmq_msg)
            # print(f"[GR00T Start Published] {zmq_msg}")

    def get_obs(self):
        self.gravity_vector = self.se.get_gravity_vector()

        if self.own_policy and self.current_q_cmd is not None:
            with self.cmd_lock:
                self.commands[:, :] = self.current_base_command[: self.num_commands].copy()
        else:
            cmds = self.command_profile.get_command(self.timestep * self.dt)
            self.commands[:, :] = cmds[: self.num_commands]

        current_command = self.commands[:, :]
        leg_q_np = np.array(current_command)
        leg_q_json = json.dumps(leg_q_np.tolist())
        zmq_msg = f"avp_leg_command {leg_q_json}"

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
        # print("commands: ", self.commands[:, :])
        # print("ang_vel: ", self.body_angular_vel)
        # print("grav: ", self.gravity_vector)
        # print("dof_pos: ", self.dof_pos)
        # print("dof_vel: ", self.dof_vel)
        # print("action: ", actions)
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

        with self.cmd_lock:
            current_sol_q = self.current_arm_command.copy() if self.current_q_cmd is not None else None

        if current_sol_q is not None:
            # print(f"[50Hz Loop] Using sol_q: {current_sol_q}")
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

    def reset(self):
        self.actions = torch.zeros(12)
        self.time = time.time()
        self.timestep = 0
        with self.cmd_lock:
            self.current_arm_command = np.zeros(14)
            self.current_base_command = np.zeros(4)
        self.act_step = 0
        self.get_current_q_command()
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
