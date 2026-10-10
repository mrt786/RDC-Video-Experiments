import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from moving_mnist import MovingMNIST


# =========================
# 1. Set random seeds
# =========================
def set_seed(seed_value: int):
    """Set seed for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)


# =========================
# 2. Domain-matched dataset:
#    single rendered MovingMNIST frames + digit labels
# =========================
class MovingMNISTFrames(torch.utils.data.Dataset):
    """
    Wraps MovingMNIST to yield single (1, H, W) frames with their digit label.

    Uses the SAME rendering pipeline as the codec training in train.py
    (same image size, digit scale, random placement), so the classifier is
    domain-matched to the reconstructions it scores during RDC training.
    """

    def __init__(self, train, data_root="./data/", image_size=64):
        self.mm = MovingMNIST(
            train=train,
            data_root=data_root,
            seq_len=1,
            num_digits=1,
            image_size=image_size,
        )

    def __len__(self):
        return len(self.mm)

    def __getitem__(self, index):
        x, label = self.mm[index]                     # x: (1, H, W, 1) float32
        frame = torch.from_numpy(x[0])                # (H, W, 1)
        frame = frame.permute(2, 0, 1).contiguous()   # (1, H, W)
        return frame, int(label)


def get_data_loaders(data_root="./data/", image_size=64, batch_size=128, num_workers=2):
    """Rendered-frame DataLoaders (train + test)."""
    train_ds = MovingMNISTFrames(train=True, data_root=data_root, image_size=image_size)
    test_ds = MovingMNISTFrames(train=False, data_root=data_root, image_size=image_size)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


# =========================
# 3. CNN classifier (native 64x64 input)
# =========================
class Net(nn.Module):
    """
    Same topology as before (conv-pool-conv-pool-FC-FC), but sized for
    `image_size` x `image_size` inputs (default 64x64, matching the codec).

    The flattened feature dimension is computed automatically, so changing
    image_size never silently breaks the FC layers again.
    """

    def __init__(self, image_size=64):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dynamically determine flattened feature size for this resolution
        with torch.no_grad():
            dummy = torch.zeros(1, 1, image_size, image_size)
            dummy = self.pool(F.relu(self.conv1(dummy)))
            dummy = self.pool(F.relu(self.conv2(dummy)))
            flat_dim = dummy.numel()  # e.g. 13*13*10 = 1690 for 64x64

        self.fc1 = nn.Linear(flat_dim, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))       # (N, 10, H-4,     W-4)
        x = self.pool(x)                # (N, 10, (H-4)/2, (W-4)/2)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)     # safe flatten (never merges samples)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ===============================
# 4. Train the model
# ===============================
def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    train_losses, train_accuracies = [], []

    print("Training the classifier on rendered MovingMNIST frames")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_acc = 0
        total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.max(outputs, 1)[1]
            total_acc += (preds == labels).sum().item()
            total_samples += images.size(0)

        epoch_loss = total_loss / total_samples
        epoch_acc = total_acc / total_samples

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(
            f"Epoch [{epoch + 1}/{epochs}]  "
            f"Train Loss: {epoch_loss:.4f}  Train Acc: {epoch_acc:.4f}"
        )

    print("Training complete!")
    return train_losses, train_accuracies


# ===============================
# 5. Test the model
# ===============================
def test_model(model, test_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.max(outputs, 1)[1]
            total_correct += preds.eq(labels).sum().item()
            total_samples += labels.size(0)

    return total_correct / total_samples


# ===============================
# 6. Main
# ===============================
CKPT_PATH = "trained_classifier.pth"


def classifier(data_root="./data/", image_size=64, batch_size=128, epochs=10,
               ckpt_path=CKPT_PATH):
    set_seed(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Net(image_size=image_size).to(device)

    # Reuse an existing checkpoint ONLY if it is architecturally compatible
    if os.path.exists(ckpt_path):
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"Using existing classifier: {ckpt_path}")
            return model
        except RuntimeError as e:
            print(
                f"Existing checkpoint '{ckpt_path}' is incompatible "
                f"(e.g. old 28x28 static-MNIST classifier): {e}\n"
                "Retraining a new 64x64 classifier on rendered MovingMNIST frames."
            )

    train_loader, test_loader = get_data_loaders(
        data_root=data_root, image_size=image_size, batch_size=batch_size
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_losses, train_accuracies = train_model(
        model, train_loader, criterion, optimizer, device, epochs=epochs
    )

    torch.save(model.state_dict(), ckpt_path)
    print(f"Model saved to {ckpt_path}.")

    test_acc = test_model(model, test_loader, device)
    print(f"Test Accuracy (rendered MovingMNIST frames): {test_acc * 100:.2f}%")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.legend()
    plt.title("Loss over epochs")

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.legend()
    plt.title("Accuracy over epochs")
    plt.savefig("training_curves_classifier.png")
    plt.show()

    return model


if __name__ == "__main__":
    classifier()
