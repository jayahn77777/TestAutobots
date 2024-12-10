import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from simple_transformer_model import SimpleTransformerModel  # 모델 정의

class SimpleNuScenesDataset(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path, allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "ego_in": torch.tensor(sample["center_gt_trajs"][:, :2], dtype=torch.float32),  # [T_obs, 2]
            "ground_truth": torch.tensor(sample["center_gt_trajs_future"][:, :2], dtype=torch.float32),  # [T_future, 2]
        }

def train_model(model, train_loader, val_loader, num_epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            ego_in = batch["ego_in"].cuda()  # [B, T_obs, 2]
            ground_truth = batch["ground_truth"].cuda()  # [B, T_future, 2]

            mode_probs, trajectories = model(ego_in)  # [B, num_modes, T_future, 2]
            print(f"Trajectories shape: {trajectories.shape}, Ground truth shape: {ground_truth.shape}")
            
            # 첫 번째 모드 선택
            best_trajectory = trajectories[:, 0, :, :]  # [B, T_future, 2]

            # 손실 계산
            loss = criterion(best_trajectory, ground_truth)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader)}")

        # Validation step
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in val_loader:
                ego_in = batch["ego_in"].cuda()
                ground_truth = batch["ground_truth"].cuda()
                _, trajectories = model(ego_in)
                best_trajectory = trajectories[:, 0, :, :]  # [B, T_future, 2]

                loss = criterion(best_trajectory, ground_truth)
                val_loss += loss.item()
            print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss / len(val_loader)}")

    # Save model
    torch.save(model.state_dict(), "./saved_simple_transformer_model.pth")
    print("Model saved to ./saved_simple_transformer_model.pth")

if __name__ == "__main__":
    # Dataset paths
    TRAIN_DATA_PATH = "./converted_data/converted_data.npy"
    VAL_DATA_PATH = "./converted_data/converted_data.npy"

    # Datasets and DataLoaders
    train_dataset = SimpleNuScenesDataset(TRAIN_DATA_PATH)
    val_dataset = SimpleNuScenesDataset(VAL_DATA_PATH)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Model configuration
    model = SimpleTransformerModel(
        input_dim=2, 
        hidden_dim=64, 
        num_heads=4,  # Add the num_heads parameter
        future_len=30, 
        num_modes=3
    ).cuda()

    # Train model
    train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=1e-3)