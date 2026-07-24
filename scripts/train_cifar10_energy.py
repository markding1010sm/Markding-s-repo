import argparse
from pathlib import Path

import torch
from torchvision import datasets, transforms

from helper_lib.generator import generate_samples
from helper_lib.model import get_model
from helper_lib.trainer import (
    EnergyReplayBuffer,
    load_energy_training_checkpoint,
    train_energy_model,
)
from helper_lib.utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an Energy-Based Model on CIFAR-10."
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--langevin-steps", type=int, default=60)
    parser.add_argument("--step-size", type=float, default=10.0)
    parser.add_argument("--noise-std", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str)
    parser.add_argument("--max-batches", type=int)
    return parser.parse_args()


def get_cifar10_loader(data_dir, batch_size):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    generator = torch.Generator().manual_seed(42)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )


def main():
    args = parse_args()
    set_seed(42)
    device = get_device()
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    data_loader = get_cifar10_loader(args.data_dir, args.batch_size)
    model = get_model("Energy", image_channels=3)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.0, 0.999),
    )
    replay_buffer = EnergyReplayBuffer(
        model,
        device=device,
        image_channels=3,
    )
    start_epoch = 0

    if args.resume:
        checkpoint = load_energy_training_checkpoint(
            model,
            optimizer,
            args.resume,
            device=device,
            replay_buffer=replay_buffer,
        )
        start_epoch = checkpoint["epoch"]
        print(f"Resuming Energy Model from epoch {start_epoch}.")

    trained_model, history, _ = train_energy_model(
        model,
        data_loader,
        optimizer,
        device=device,
        epochs=args.epochs,
        langevin_steps=args.langevin_steps,
        step_size=args.step_size,
        noise_std=args.noise_std,
        alpha=args.alpha,
        checkpoint_dir=checkpoint_dir,
        replay_buffer=replay_buffer,
        start_epoch=start_epoch,
        max_batches=args.max_batches,
    )
    sample_path = checkpoint_dir / "cifar10_energy_samples.png"
    generate_samples(
        trained_model,
        device=device,
        num_samples=8,
        output_path=sample_path,
        energy_steps=args.langevin_steps,
    )

    print(f"Training history: {history}")
    print(
        "Saved API checkpoint to "
        f"{checkpoint_dir / 'cifar10_energy.pth'}"
    )
    print(f"Saved samples to {sample_path}")


if __name__ == "__main__":
    main()
