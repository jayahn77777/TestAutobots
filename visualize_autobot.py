import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from train_autobot import AutoBots

# Load AutoBots Model
def load_autobot_model(model_path, input_dim, d_model, output_dim, nhead, num_set_layers, num_seq_layers):
    model = AutoBots(input_dim, d_model, output_dim, nhead, num_set_layers, num_seq_layers)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Visualization Function
def visualize_predictions(scene_data, model, device="cpu"):
    """
    Visualize the model predictions against ground truth, separating object trajectories.
    Args:
        scene_data (list): Loaded scene data with input and ground truth trajectories.
        model (torch.nn.Module): Trained AutoBot model.
        device (str): Device to use for computation ("cpu" or "cuda").
    """
    model = model.to(device)
    model.eval()

    # Normalize ground truth
    translations = np.array([item["translation"][:2] for item in scene_data])
    x_min, x_max = translations[:, 0].min(), translations[:, 0].max()
    y_min, y_max = translations[:, 1].min(), translations[:, 1].max()

    # Prepare inputs (normalize)
    inputs = torch.tensor([
        [(item["translation"][0] - x_min) / (x_max - x_min),
         (item["translation"][1] - y_min) / (y_max - y_min),
         item["velocity"] / 30.0]  # Normalize velocity to 0-30 range
        for item in scene_data
    ], dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(inputs).cpu().numpy()

    # Denormalize predictions
    predictions_denormalized = predictions * np.array([x_max - x_min, y_max - y_min]) + np.array([x_min, y_min])

    # Group ground truth by instance tokens
    instance_groups = {}
    for item in scene_data:
        token = item["instance_token"]
        if token not in instance_groups:
            instance_groups[token] = []
        instance_groups[token].append(item["translation"][:2])

    # Visualization
    plt.figure(figsize=(10, 8))
    plt.title("Predicted vs Ground Truth Trajectories")

    # Plot ground truth trajectories (object-wise)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(instance_groups)))  # Assign unique colors
    for idx, (token, trajectory) in enumerate(instance_groups.items()):
        trajectory = np.array(trajectory)
        plt.plot(
            trajectory[:, 0], trajectory[:, 1],
            label=f"Ground Truth {token[:4]}",
            color=colors[idx], linestyle="-", alpha=0.6
        )

    # Plot predictions (red)
    plt.plot(
        [p[0] for p in predictions_denormalized],
        [p[1] for p in predictions_denormalized],
        "r--", label="Predictions", linewidth=2
    )

    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

    # Calculate minADE_5
    def minADE_5(predicted_trajectories, ground_truth):
        seq_len = ground_truth.shape[0]
        predicted_trajectories = predicted_trajectories[:, :seq_len]  # 길이 일치
        min_distances = np.min([
            np.mean(np.linalg.norm(sample - ground_truth, axis=1))
            for sample in predicted_trajectories
        ])
        return min_distances

    ground_truth = translations
    ade5 = minADE_5(np.array([predictions_denormalized]), ground_truth)
    print(f"minADE_5: {ade5:.4f}")

# Main Script
if __name__ == "__main__":
    # Paths and Hyperparameters
    model_path = "./autobot_model_transformer.pth"
    input_dim = 3
    d_model = 128
    output_dim = 2
    nhead = 4
    num_set_layers = 2
    num_seq_layers = 2

    # Load Model
    model = load_autobot_model(model_path, input_dim, d_model, output_dim, nhead, num_set_layers, num_seq_layers)

    # Load Processed Data
    with open("./processed_data/scene-scene-0916.json", "r") as f:
        scene_data = json.load(f)

    # Visualize Predictions
    visualize_predictions(scene_data, model, device="cuda" if torch.cuda.is_available() else "cpu")