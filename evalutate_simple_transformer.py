import torch
import numpy as np
from simple_transformer_model import SimpleTransformerModel
from train_sample_transformer import SimpleNuScenesDataset
from torch.utils.data import DataLoader


def calculate_minADE_5(trajectories, ground_truth):
    """
    Calculate minADE_5: Average Displacement Error for the top 5 modes.
    """
    num_modes = trajectories.size(1)  # Number of modes in the prediction
    k = min(5, num_modes)  # Ensure k is not greater than num_modes

    ground_truth = ground_truth.unsqueeze(1).expand_as(trajectories)  # [B, num_modes, T_future, 2]
    errors = torch.norm(trajectories - ground_truth, dim=-1)  # [B, num_modes, T_future]
    mean_errors = errors.mean(dim=-1)  # [B, num_modes]

    min_errors = mean_errors.topk(k, dim=-1, largest=False)[0].mean()  # Top-k ADE
    return min_errors

def evaluate_model(model, data_loader):
    model.eval()
    total_minADE_5 = 0
    count = 0

    with torch.no_grad():
        for batch in data_loader:
            ego_in = batch["ego_in"].cuda()
            ground_truth = batch["ground_truth"].cuda()

            _, trajectories = model(ego_in)  # [B, num_modes, future_len, 2]
            minADE_5 = calculate_minADE_5(trajectories, ground_truth)
            total_minADE_5 += minADE_5.item()
            count += 1

    avg_minADE_5 = total_minADE_5 / count
    print(f"Average minADE_5: {avg_minADE_5}")
    return avg_minADE_5


if __name__ == "__main__":
    # Load dataset
    DATA_PATH = "./converted_data/converted_data.npy"
    dataset = SimpleNuScenesDataset(DATA_PATH)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Load trained model
    model = SimpleTransformerModel(
        input_dim=2, 
        hidden_dim=64, 
        num_heads=4,  # num_heads 추가
        num_modes=3, 
        future_len=30
    ).cuda()
    model.load_state_dict(torch.load("./saved_simple_transformer_model.pth"))
    print("Model loaded successfully.")

    # Evaluate model
    evaluate_model(model, data_loader)