import torch
import numpy as np

class SimpleNuScenesDataset(torch.utils.data.Dataset):
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
