import argparse

import numpy as np
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

parser = argparse.ArgumentParser()
parser.add_argument("--task_dir", type=str, default="./utils/data", help="path to save data")
parser.add_argument("--frequency", type=int, default=30.0, help="save data's frequency")
parser.add_argument("--vis", action="store_true", help="Save data or not")
parser.add_argument("--no-record", dest="record", action="store_false", help="Do not save data")
parser.set_defaults(record=False)
parser.add_argument(
    "--arm", type=str, default="G1_29", choices=["G1_29", "G1_23", "H1_2", "H1"], help="Select arm controller"
)
parser.add_argument(
    "--hand", type=str, default="dex3", choices=["dex3", "gripper", "inspire1"], help="Select hand controller"
)
parser.add_argument("--goal", type=str, default="pick_pink_fox", help="Language Goal.")  # TODO: 带下划线...?
parser.add_argument(
    "--model-path", type=str, default="./save/multiobj_pick_WBC", help="Path to the model checkpoint directory."
)
parser.add_argument("--embodiment-tag", type=str, default="new_embodiment", help="The embodiment tag for the model.")
parser.add_argument("--data-config", type=str, default="openwbc_g1", help="The name of the data config to use.")
parser.add_argument("--server", action="store_true", help="Whether to run the server.")
parser.add_argument("--client", action="store_true", help="Whether to run the client.")
parser.add_argument("--denoising-steps", type=int, default=4, help="The number of denoising steps to use.")
parser.add_argument("--action_horizon", type=int, default=16, help="The action horizon for the policy.")
parser.add_argument("--filt", action="store_true", help="add filter")

args = parser.parse_args()
print(f"args:{args}\n")

data_config = DATA_CONFIG_MAP[args.data_config]
modality_config = data_config.modality_config()
modality_transform = data_config.transform()
policy = Gr00tPolicy(
    model_path=args.model_path,
    modality_config=modality_config,
    modality_transform=modality_transform,
    embodiment_tag=args.embodiment_tag,
    denoising_steps=args.denoising_steps,
)
print("Policy loaded.")

# Create dummy observation following the format from G1_inference.py
# Dummy image dimensions (typical camera resolution)
H, W, C = 480, 640, 3

obs = {
    # Vision data - all three video keys required by OpenWBCDataConfig
    "video.ego_view": np.random.randint(0, 255, (1, H, W, C), dtype=np.uint8),
    "video.wrist_left": np.random.randint(0, 255, (1, H, W, C), dtype=np.uint8),
    "video.wrist_right": np.random.randint(0, 255, (1, H, W, C), dtype=np.uint8),
    # Robot state data
    "state.left_arm": np.random.uniform(-1.0, 1.0, (1, 7)).astype(np.float32),  # 7 DOF arm
    "state.right_arm": np.random.uniform(-1.0, 1.0, (1, 7)).astype(np.float32),  # 7 DOF arm
    "state.left_hand": np.random.uniform(-1.0, 1.0, (1, 7)).astype(np.float32),  # Hand state (7 DOF for dex3)
    "state.right_hand": np.random.uniform(-1.0, 1.0, (1, 7)).astype(np.float32),  # Hand state (7 DOF for dex3)
    "state.left_leg": np.random.uniform(-1.0, 1.0, (1, 6)).astype(np.float32),  # 6 DOF leg
    "state.right_leg": np.random.uniform(-1.0, 1.0, (1, 6)).astype(np.float32),  # 6 DOF leg
    # Task description
    "annotation.human.action.task_description": ["pick up the cup"],
}

action = policy.get_action(obs)
