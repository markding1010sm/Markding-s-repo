import argparse
from pathlib import Path

import torch
from torchvision import datasets, transforms

from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train a WGAN on MNIST digits.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--n-critic", type=int, default=5)
    parser.add_argument("--clip-value", type=float, default=0.01)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return parser.parse_args()


def get_mnist_loader(data_dir, batch_size):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )


def main():
    args = parse_args()
    set_seed(42)
    device = get_device()
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader = get_mnist_loader(args.data_dir, args.batch_size)
    model = get_model("MNISTGAN", latent_dim=args.latent_dim)

    trained_model, history = train_gan(
        model,
        train_loader,
        device=device,
        epochs=args.epochs,
        lr=args.learning_rate,
        n_critic=args.n_critic,
        clip_value=args.clip_value,
        checkpoint_dir=checkpoint_dir,
    )

    torch.save(
        {
            "model_state_dict": trained_model.state_dict(),
            "latent_dim": args.latent_dim,
            "history": history,
        },
        checkpoint_dir / "mnist_gan.pth",
    )

    print(f"Training history: {history}")
    print("Saved MNIST GAN checkpoint to checkpoints/mnist_gan.pth")


if __name__ == "__main__":
    main()
