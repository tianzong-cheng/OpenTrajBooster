import argparse
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
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
parser.add_argument(
    "--dataset-path",
    type=str,
    default="../datasets/pick_tissue",
    help="Path to the LeRobot format dataset directory.",
)
parser.add_argument(
    "--video-backend",
    type=str,
    default="decord",
    choices=["decord", "torchvision_av"],
    help="Backend to use for video loading.",
)
parser.add_argument(
    "--plot",
    action="store_true",
    default=True,
    help="Generate plots comparing predicted vs ground truth actions.",
)

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

# Load dataset using policy's modality config
dataset_path = pathlib.Path(args.dataset_path)
if not dataset_path.exists():
    raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")

print("\n" + "=" * 80)
print(f"Loading dataset from {dataset_path}")
print("=" * 80)

embodiment_tag = EmbodimentTag(args.embodiment_tag)

# Load dataset with policy's modality config
dataset = LeRobotSingleDataset(
    dataset_path=str(dataset_path),
    modality_configs=modality_config,
    embodiment_tag=embodiment_tag,
    video_backend=args.video_backend,
    transforms=None,  # Policy will handle transforms
)
print(f"Dataset loaded: {len(dataset)} total steps across {len(dataset.trajectory_ids)} episodes")

# Replay dataset for inference test
print("\n" + "=" * 80)
print("Starting inference test on dataset...")
print("=" * 80)

# Test all episodes
num_episodes_to_test = len(dataset.trajectory_ids)

# Storage for visualization
all_predicted_actions = {}  # {action_key: [list of actions across all steps]}
all_gt_actions = {}  # {action_key: [list of actions across all steps]}
all_episode_ids = []  # Track which episode each step belongs to
all_step_indices = []  # Track step index for each action

total_inferences = 0
inference_start_time = time.time()

# Timing diagnostics
total_data_load_time = 0.0
total_inference_time = 0.0
total_postprocess_time = 0.0

for episode_idx in range(1):
    trajectory_id = dataset.trajectory_ids[episode_idx]
    trajectory_length = dataset.trajectory_lengths[episode_idx]

    # Test all steps in the episode
    num_steps_to_test = trajectory_length

    print(f"\nEpisode {episode_idx} (trajectory_id={trajectory_id}, length={trajectory_length}):")

    for step_idx in range(num_steps_to_test):
        # Get observation from dataset using get_step_data for raw data
        # Then we'll format it for the policy
        try:
            data_load_start = time.time()
            data_point = dataset.get_step_data(trajectory_id, step_idx)
            total_data_load_time += time.time() - data_load_start
        except Exception as e:
            print(f"  Step {step_idx}: Failed to get data: {e}")
            continue

        # Extract observation (exclude action keys)
        obs = {key: value for key, value in data_point.items() if not key.startswith("action.")}

        # Run inference
        try:
            inference_start = time.time()
            action = policy.get_action(obs)
            total_inference_time += time.time() - inference_start
            total_inferences += 1

            postprocess_start = time.time()

            # Store actions for visualization
            if args.plot:
                # Get ground truth action (first step of action horizon)
                for action_key in action.keys():
                    if action_key in data_point:
                        # Initialize lists if needed
                        if action_key not in all_predicted_actions:
                            all_predicted_actions[action_key] = []
                            all_gt_actions[action_key] = []

                        # Get predicted action (first step of action horizon)
                        pred_action = action[action_key]
                        if isinstance(pred_action, np.ndarray):
                            # Policy returns (action_horizon, dim) or (dim,)
                            # Take first step of action horizon if it exists
                            if len(pred_action.shape) > 1:
                                pred_action = pred_action[0]  # Take first step
                            # Ensure it's a 1D array
                            pred_action = np.atleast_1d(pred_action).flatten()
                            all_predicted_actions[action_key].append(pred_action.copy())

                        # Get ground truth action
                        gt_action = data_point[action_key]
                        if isinstance(gt_action, np.ndarray):
                            # Dataset action might have shape (1, dim) or (dim,)
                            # Take first element if needed
                            if len(gt_action.shape) > 1:
                                gt_action = gt_action[0]
                            # Ensure it's a 1D array
                            gt_action = np.atleast_1d(gt_action).flatten()
                            all_gt_actions[action_key].append(gt_action.copy())

                all_episode_ids.append(episode_idx)
                all_step_indices.append(step_idx)

            total_postprocess_time += time.time() - postprocess_start

            if step_idx % 10 == 0 or step_idx == num_steps_to_test - 1:
                print(f"  Step {step_idx}/{num_steps_to_test - 1}: Inference successful")
        except Exception as e:
            print(f"  Step {step_idx}: Inference failed with error: {e}")
            import traceback

            traceback.print_exc()

inference_end_time = time.time()
total_wall_time = inference_end_time - inference_start_time
avg_inference_frequency = total_inferences / total_wall_time if total_wall_time > 0 else 0

print("\n" + "=" * 80)
print(f"Inference test completed. Total successful inferences: {total_inferences}")
print(f"Total wall time: {total_wall_time:.2f} seconds")
print(
    f"Average inference frequency: {avg_inference_frequency:.2f} Hz ({avg_inference_frequency:.2f} inferences/second)"
)
print("\nTiming breakdown:")
data_load_pct = 100 * total_data_load_time / total_wall_time if total_wall_time > 0 else 0
inference_pct = 100 * total_inference_time / total_wall_time if total_wall_time > 0 else 0
postprocess_pct = 100 * total_postprocess_time / total_wall_time if total_wall_time > 0 else 0
print(f"  Data loading time: {total_data_load_time:.2f} seconds ({data_load_pct:.1f}%)")
print(f"  GPU inference time: {total_inference_time:.2f} seconds ({inference_pct:.1f}%)")
print(f"  Post-processing time: {total_postprocess_time:.2f} seconds ({postprocess_pct:.1f}%)")
other_time = total_wall_time - total_data_load_time - total_inference_time - total_postprocess_time
print(f"  Other overhead: {other_time:.2f} seconds")
if total_inferences > 0:
    print("\nPer-inference averages:")
    print(f"  Data loading: {total_data_load_time/total_inferences*1000:.2f} ms")
    print(f"  GPU inference: {total_inference_time/total_inferences*1000:.2f} ms")
    print(f"  Post-processing: {total_postprocess_time/total_inferences*1000:.2f} ms")
print("=" * 80)

# Generate visualizations if requested
if args.plot and all_predicted_actions:
    print("\n" + "=" * 80)
    print("Generating action comparison plots...")
    print("=" * 80)

    # Convert lists to numpy arrays
    for action_key in all_predicted_actions.keys():
        all_predicted_actions[action_key] = np.array(all_predicted_actions[action_key])
        all_gt_actions[action_key] = np.array(all_gt_actions[action_key])

    # Calculate statistics
    print("\nAction Comparison Statistics:")
    print("-" * 80)
    for action_key in all_predicted_actions.keys():
        pred = all_predicted_actions[action_key]
        gt = all_gt_actions[action_key]

        # Calculate MSE
        mse = np.mean((pred - gt) ** 2)
        mae = np.mean(np.abs(pred - gt))
        max_error = np.max(np.abs(pred - gt))

        print(f"\n{action_key}:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  Max Error: {max_error:.6f}")
        print(f"  Shape: {pred.shape}")

    # Create plots for each action key
    for action_key in all_predicted_actions.keys():
        pred = all_predicted_actions[action_key]
        gt = all_gt_actions[action_key]

        # Determine number of dimensions
        if len(pred.shape) == 1:
            num_dims = 1
            pred = pred.reshape(-1, 1)
            gt = gt.reshape(-1, 1)
        else:
            num_dims = pred.shape[1]

        # Create figure with subplots
        fig, axes = plt.subplots(nrows=num_dims, ncols=1, figsize=(12, 4 * num_dims), sharex=True)
        if num_dims == 1:
            axes = [axes]

        fig.suptitle(f"Action Comparison: {action_key}", fontsize=16, fontweight="bold")

        for dim in range(num_dims):
            ax = axes[dim]
            time_steps = np.arange(len(pred))

            # Plot predicted and ground truth
            ax.plot(time_steps, pred[:, dim], label="Predicted", linewidth=2, alpha=0.7)
            ax.plot(time_steps, gt[:, dim], label="Ground Truth", linewidth=2, alpha=0.7, linestyle="--")

            # Calculate error for statistics
            error = pred[:, dim] - gt[:, dim]

            ax.set_ylabel(f"Dim {dim}")
            ax.set_title(f"Dimension {dim} - MSE: {np.mean(error**2):.6f}, MAE: {np.mean(np.abs(error)):.6f}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Step")
        plt.tight_layout()

        # Display plot
        plt.show()
        plt.close()
