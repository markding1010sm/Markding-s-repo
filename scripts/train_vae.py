from helper_lib.data_loader import get_data_loader
from helper_lib.generator import generate_samples
from helper_lib.model import get_model
from helper_lib.trainer import train_vae_model
from helper_lib.utils import get_device, set_seed
import torch.optim as optim


def main():
    set_seed(42)
    device = get_device()

    train_loader = get_data_loader("data/train", batch_size=64)
    vae = get_model("VAE")
    optimizer = optim.Adam(vae.parameters(), lr=0.001)

    trained_vae, history = train_vae_model(
        vae,
        train_loader,
        optimizer,
        device=device,
        epochs=5,
        checkpoint_dir="checkpoints",
    )

    samples = generate_samples(
        trained_vae,
        device,
        num_samples=10,
        output_path="checkpoints/vae_samples.png",
    )

    print(f"Training history: {history}")
    print(f"Generated samples shape: {tuple(samples.shape)}")


if __name__ == "__main__":
    main()
