import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Custom Dataset
class AutoBotDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []
        for file_name in os.listdir(data_dir):
            if file_name.endswith('.json'):
                file_path = os.path.join(data_dir, file_name)
                with open(file_path, 'r') as f:
                    self.data.extend(json.load(f))

        # Compute normalization statistics
        translations = [item['translation'][:2] for item in self.data]
        self.translation_mean = np.mean(translations, axis=0)
        self.translation_std = np.std(translations, axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if 'translation' not in item or 'velocity' not in item or 'acceleration' not in item:
            raise KeyError(f"Missing required keys in data at index {idx}")

        # Normalize translation inputs
        translation = (np.array(item['translation'][:2]) - self.translation_mean) / self.translation_std
        velocity = item['velocity']
        inputs = torch.tensor([*translation, velocity], dtype=torch.float32)

        # Labels
        labels = torch.tensor([item['velocity'], item['acceleration']], dtype=torch.float32)

        return inputs, labels

# AutoBots Model Definition
class AutoBots(nn.Module):
    def __init__(self, input_dim, d_model, output_dim, nhead, num_set_layers, num_seq_layers):
        super(AutoBots, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_set_layers
        )
        self.latent_layer = nn.Linear(input_dim, d_model)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_seq_layers
        )
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # Encode inputs
        x = self.latent_layer(x)
        x = self.encoder(x.unsqueeze(0))  # Add sequence dimension
        # Decode to predictions
        x = self.decoder(x, x)  # Using itself as memory
        return self.output_layer(x.squeeze(0))

# Training Function
def train_autobot(model, train_loader, criterion, optimizer, num_epochs=10, device="cpu"):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")
    print("Training completed!")
    return model

# Save Model
def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")

# Main Script
if __name__ == "__main__":
    # Paths and Hyperparameters
    data_dir = "./processed_data"
    save_model_path = "./autobot_model_transformer.pth"
    input_dim = 3
    d_model = 128
    output_dim = 2
    nhead = 4
    num_set_layers = 2
    num_seq_layers = 2
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001

    # Load Dataset and DataLoader
    dataset = AutoBotDataset(data_dir)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize Model, Loss Function, and Optimizer
    model = AutoBots(input_dim, d_model, output_dim, nhead, num_set_layers, num_seq_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_autobot(model, train_loader, criterion, optimizer, num_epochs, device)

    # Save Model
    save_model(model, save_model_path)