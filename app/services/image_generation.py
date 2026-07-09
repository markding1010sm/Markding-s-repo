from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import make_grid

from helper_lib.model import get_model


class MNISTImageGenerator:
    def __init__(
        self,
        checkpoint_path="checkpoints/mnist_gan.pth",
        latent_dim=100,
        device="cpu",
    ):
        project_root = Path(__file__).resolve().parents[2]
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = project_root / checkpoint_path

        self.checkpoint_path = checkpoint_path
        self.latent_dim = latent_dim
        self.device = device
        self.model = get_model("MNISTGAN", latent_dim=latent_dim)
        self.is_loaded = False

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        checkpoint_latent_dim = checkpoint.get("latent_dim", self.latent_dim)
        if checkpoint_latent_dim != self.latent_dim:
            self.latent_dim = checkpoint_latent_dim
            self.model = get_model("MNISTGAN", latent_dim=self.latent_dim)

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

    def generate_digits(self, num_images=16):
        if not self.is_loaded:
            if not self.checkpoint_path.exists():
                raise RuntimeError(
                    "MNIST GAN checkpoint not found. Run "
                    "`uv run python -m scripts.train_mnist_gan` first."
                )

            try:
                self.load_checkpoint()
            except RuntimeError as exc:
                raise RuntimeError(
                    "MNIST GAN checkpoint is incompatible with the current "
                    "assignment architecture. Re-run "
                    "`uv run python -m scripts.train_mnist_gan` to regenerate it."
                ) from exc

        with torch.no_grad():
            noise = torch.randn(num_images, self.latent_dim, 1, 1, device=self.device)
            samples = self.model.generator(noise).cpu()

        samples = (samples + 1) / 2
        grid = make_grid(
            samples,
            nrow=min(num_images, 8),
            normalize=False,
            padding=2,
        )
        image_tensor = grid.mul(255).clamp(0, 255).byte()
        buffer = BytesIO()
        if image_tensor.size(0) == 1:
            image = Image.fromarray(image_tensor.squeeze(0).numpy())
        else:
            image = Image.fromarray(image_tensor.permute(1, 2, 0).numpy())
        image.save(buffer, format="PNG")

        return buffer.getvalue()
