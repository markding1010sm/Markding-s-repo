from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torchvision import datasets, transforms

from helper_lib.model import get_model
from helper_lib.trainer import train_model
from helper_lib.utils import get_device, set_seed


def get_cifar10_loaders(data_dir="data", batch_size=64):
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


def main():
    set_seed(42)
    device = get_device()
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = get_cifar10_loaders()
    model = get_model("CNN", num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trained_model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device=device,
        epochs=10,
        checkpoint_dir=checkpoint_dir,
    )

    torch.save(
        {
            "model_state_dict": trained_model.state_dict(),
            "classes": train_loader.dataset.dataset.classes,
            "history": history,
        },
        checkpoint_dir / "cifar10_cnn.pth",
    )

    print(f"Training history: {history}")
    print("Saved classifier checkpoint to checkpoints/cifar10_cnn.pth")


if __name__ == "__main__":
    main()
