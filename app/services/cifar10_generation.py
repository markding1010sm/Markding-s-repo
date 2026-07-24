from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import make_grid

from helper_lib.generator import generate_energy_samples
from helper_lib.model import get_model


def _samples_to_png(samples):
    grid = make_grid(
        samples,
        nrow=min(samples.size(0), 8),
        normalize=False,
        padding=2,
    )
    image_tensor = grid.mul(255).clamp(0, 255).byte().cpu()
    image = Image.fromarray(image_tensor.permute(1, 2, 0).numpy())
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class CIFAR10EnergyGenerator:
    def __init__(
        self,
        checkpoint_path="checkpoints/cifar10_energy.pth",
        langevin_steps=60,
        step_size=10.0,
        noise_std=0.005,
        device="cpu",
    ):
        project_root = Path(__file__).resolve().parents[2]
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = project_root / checkpoint_path

        self.checkpoint_path = checkpoint_path
        self.langevin_steps = langevin_steps
        self.step_size = step_size
        self.noise_std = noise_std
        self.device = torch.device(device)
        self.model = get_model("Energy", image_channels=3)
        self.is_loaded = False

    def load_checkpoint(self):
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

    def generate_images(self, num_images=8):
        if not self.is_loaded:
            if not self.checkpoint_path.exists():
                raise RuntimeError(
                    "CIFAR-10 Energy Model checkpoint not found. Run "
                    "`uv run python -m scripts.train_cifar10_energy` first."
                )
            try:
                self.load_checkpoint()
            except (KeyError, RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    "CIFAR-10 Energy Model checkpoint is incompatible. "
                    "Re-run `uv run python -m "
                    "scripts.train_cifar10_energy`."
                ) from exc

        initial_images = torch.rand(
            num_images,
            3,
            32,
            32,
            device=self.device,
        ) * 2 - 1
        samples = generate_energy_samples(
            self.model,
            initial_images,
            steps=self.langevin_steps,
            step_size=self.step_size,
            noise_std=self.noise_std,
        )
        samples = (samples + 1) / 2
        return _samples_to_png(samples)


class CIFAR10DiffusionGenerator:
    def __init__(
        self,
        checkpoint_path="checkpoints/cifar10_diffusion.pth",
        diffusion_steps=50,
        device="cpu",
    ):
        project_root = Path(__file__).resolve().parents[2]
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = project_root / checkpoint_path

        self.checkpoint_path = checkpoint_path
        self.diffusion_steps = diffusion_steps
        self.device = torch.device(device)
        self.model = get_model(
            "Diffusion",
            image_channels=3,
            image_size=32,
        )
        self.is_loaded = False

    def load_checkpoint(self):
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
        )
        self.model.network.load_state_dict(checkpoint["model_state_dict"])
        self.model.ema_network.load_state_dict(
            checkpoint["ema_model_state_dict"]
        )
        self.model.normalizer_mean.copy_(checkpoint["normalizer_mean"])
        self.model.normalizer_std.copy_(checkpoint["normalizer_std"])
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

    def generate_images(self, num_images=8):
        if not self.is_loaded:
            if not self.checkpoint_path.exists():
                raise RuntimeError(
                    "CIFAR-10 Diffusion checkpoint not found. Run "
                    "`uv run python -m scripts.train_cifar10_diffusion` "
                    "first."
                )
            try:
                self.load_checkpoint()
            except (KeyError, RuntimeError, ValueError) as exc:
                raise RuntimeError(
                    "CIFAR-10 Diffusion checkpoint is incompatible. "
                    "Re-run `uv run python -m "
                    "scripts.train_cifar10_diffusion`."
                ) from exc

        samples = self.model.generate(
            num_images=num_images,
            diffusion_steps=self.diffusion_steps,
        )
        return _samples_to_png(samples)
