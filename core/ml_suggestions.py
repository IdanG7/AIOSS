import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import math
from torch.optim.lr_scheduler import OneCycleLR
import os
from tqdm import tqdm


class FastPasswordDataset(Dataset):
    def __init__(self, passwords, labels, char_to_idx, max_length=50):
        """Precompute as much as possible during initialization"""
        self.max_length = max_length

        # Convert all passwords to tensors at once
        self.password_tensors = torch.zeros(
            (len(passwords), max_length), dtype=torch.long
        )
        self.attention_masks = torch.zeros((len(passwords), max_length))
        self.features = torch.zeros((len(passwords), 7))

        # Precompute all features
        for i, password in enumerate(passwords):
            # Password tensor
            length = min(len(password), max_length)
            for j, char in enumerate(password[:length]):
                self.password_tensors[i, j] = char_to_idx.get(char, 1)

            # Attention mask
            self.attention_masks[i, :length] = 1

            # Features
            if len(password) > 0:
                self.features[i, 0] = length / max_length  # normalized length
                self.features[i, 1] = sum(c.isupper() for c in password) / len(password)
                self.features[i, 2] = sum(c.islower() for c in password) / len(password)
                self.features[i, 3] = sum(c.isdigit() for c in password) / len(password)
                self.features[i, 4] = sum(not c.isalnum() for c in password) / len(
                    password
                )
                self.features[i, 5] = len(set(password)) / len(password)
                # Entropy calculation
                counts = Counter(password)
                probs = [count / len(password) for count in counts.values()]
                self.features[i, 6] = -sum(p * math.log2(p) for p in probs)

        # Convert labels
        self.labels = torch.tensor(
            np.where(labels == "strong", 1.0, 0.0), dtype=torch.float32
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "password": self.password_tensors[idx],
            "attention_mask": self.attention_masks[idx],
            "features": self.features[idx],
            "label": self.labels[idx],
        }


class FastPasswordClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Combined CNN and LSTM for faster processing
        self.conv = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True
        )

        # Simplified attention
        self.attention = nn.Linear(hidden_dim * 2, 1)

        # Simplified feature processing
        self.feature_net = nn.Linear(7, hidden_dim)

        # Efficient classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, batch):
        x = self.embedding(batch["password"])

        # CNN
        x = x.transpose(1, 2)
        x = F.relu(self.conv(x))
        x = x.transpose(1, 2)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Simple attention
        attention_weights = self.attention(lstm_out).squeeze(-1)
        attention_weights = attention_weights.masked_fill(
            batch["attention_mask"] == 0, float("-inf")
        )
        attention_weights = F.softmax(attention_weights, dim=1)
        attended = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)

        # Process features
        processed_features = self.feature_net(batch["features"])

        # Combine and classify
        combined = torch.cat([attended, processed_features], dim=1)
        return self.classifier(combined)


def train_password_model(csv_path="data/passwords_labeled.csv", batch_size=128):
    # Load data
    print("Loading data...")
    data = pd.read_csv(csv_path)

    # Create vocabulary (use bytes for faster processing)
    all_chars = set("".join(data["password"].values))
    char_to_idx = {char: idx + 2 for idx, char in enumerate(sorted(all_chars))}
    char_to_idx["<pad>"] = 0
    char_to_idx["<unk>"] = 1

    # Split data
    train_data, valid_data = train_test_split(
        data, test_size=0.2, stratify=data["label"], random_state=42
    )

    # Create datasets with precomputed features
    print("Creating datasets...")
    train_dataset = FastPasswordDataset(
        train_data["password"].values, train_data["label"].values, char_to_idx
    )
    valid_dataset = FastPasswordDataset(
        valid_data["password"].values, valid_data["label"].values, char_to_idx
    )

    # Create dataloaders with multiple workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = FastPasswordClassifier(len(char_to_idx)).to(device)

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = OneCycleLR(
        optimizer, max_lr=1e-3, epochs=10, steps_per_epoch=len(train_loader)
    )

    print("Starting training...")
    best_valid_acc = 0

    for epoch in range(10):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/10")
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch)
            loss = F.binary_cross_entropy(outputs.squeeze(), batch["label"])

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Calculate metrics
            predictions = (outputs.squeeze() > 0.5).float()
            train_correct += (predictions == batch["label"]).sum().item()
            train_total += batch["label"].size(0)
            train_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix(
                {"loss": train_loss / train_total, "acc": train_correct / train_total}
            )

        # Validation phase
        model.eval()
        valid_correct = 0
        valid_total = 0

        with torch.no_grad():
            for batch in valid_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(batch)

                predictions = (outputs.squeeze() > 0.5).float()
                valid_correct += (predictions == batch["label"]).sum().item()
                valid_total += batch["label"].size(0)

        # Calculate metrics
        train_accuracy = train_correct / train_total
        valid_accuracy = valid_correct / valid_total

        print(f"\nEpoch {epoch+1}")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Validation Accuracy: {valid_accuracy:.4f}")

        # Save best model
        if valid_accuracy > best_valid_acc:
            best_valid_acc = valid_accuracy
            torch.save(model.state_dict(), "models/best_model.pt")

    return best_valid_acc, model, char_to_idx


if __name__ == "__main__":
    best_accuracy, model, char_to_idx = train_password_model()
    print(f"Best validation accuracy: {best_accuracy:.4f}")
