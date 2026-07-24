import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from helper_lib.generator import generate_samples
from helper_lib.model import get_model
from helper_lib.trainer import (
    load_diffusion_training_checkpoint,
    train_diffusion,
)
from helper_lib.utils import get_device, set_seed

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Diffusion Model on CIFAR-10."
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    return parser.parse_args()


def get_cifar10_loaders(data_dir, batch_size):
    transform = transforms.ToTensor()
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    validation_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )
    generator = torch.Generator().manual_seed(42)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, validation_loader


def main():
    args = parse_args()
    set_seed(42)
    device = get_device()
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader, validation_loader = get_cifar10_loaders(
        args.data_dir,
        args.batch_size,
    )
    model = get_model(
        "Diffusion",
        image_channels=3,
        image_size=32,
        ema_decay=args.ema_decay,
    )
    model.set_normalizer(CIFAR10_MEAN, CIFAR10_STD)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.network.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    start_epoch = 0

    if args.resume:
        checkpoint = load_diffusion_training_checkpoint(
            model,
            optimizer,
            args.resume,
            device=device,
        )
        start_epoch = checkpoint["epoch"]
        print(f"Resuming Diffusion Model from epoch {start_epoch}.")

    trained_model, history = train_diffusion(
        model,
        train_loader,
        nn.L1Loss(),
        optimizer,
        device=device,
        epochs=args.epochs,
        val_loader=validation_loader,
        checkpoint_dir=checkpoint_dir,
        start_epoch=start_epoch,
        max_batches=args.max_batches,
        max_val_batches=args.max_val_batches,
    )
    sample_path = checkpoint_dir / "cifar10_diffusion_samples.png"
    generate_samples(
        trained_model,
        device=device,
        num_samples=8,
        output_path=sample_path,
        diffusion_steps=50,
    )

    print(f"Training history: {history}")
    print(
        "Saved API checkpoint to "
        f"{checkpoint_dir / 'cifar10_diffusion.pth'}"
    )
    print(f"Saved samples to {sample_path}")


if __name__ == "__main__":
    main()
