import torch
import torch.nn as nn

class SimpleTransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_modes, future_len):
        super(SimpleTransformerModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.future_len = future_len
        self.num_modes = num_modes

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, batch_first=True
            ),
            num_layers=3  # num_layers는 고정 또는 전달
        )

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.trajectory_prediction = nn.Linear(hidden_dim, future_len * 2 * num_modes)
        self.mode_prediction = nn.Linear(hidden_dim, num_modes)

    def forward(self, ego_in):
        ego_in = self.input_projection(ego_in)  # [B, T, hidden_dim]
        encoded = self.encoder(ego_in)  # [B, T, hidden_dim]
        mode_probs = torch.softmax(self.mode_prediction(encoded[:, -1, :]), dim=-1)  # [B, num_modes]
        trajectories = self.trajectory_prediction(encoded[:, -1, :]).view(
            -1, self.num_modes, self.future_len, 2
        )  # [B, num_modes, future_len, 2]
        return mode_probs, trajectories