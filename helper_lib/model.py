import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNN(nn.Module):
    def __init__(self, num_classes=10, image_size=64):
        super().__init__()
        input_features = 3 * image_size * image_size
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class EnhancedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class VAE(nn.Module):
    def __init__(self, latent_dim=128, image_channels=3, image_size=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_channels = image_channels
        self.image_size = image_size
        self.flattened_size = image_channels * image_size * image_size

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, 512)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.flattened_size),
            nn.Sigmoid(),
        )

    def encode(self, x):
        encoded = self.encoder(x)
        return self.fc_mu(encoded), self.fc_logvar(encoded)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        decoded = self.decoder_input(z)
        decoded = self.decoder(decoded)
        return decoded.view(
            -1,
            self.image_channels,
            self.image_size,
            self.image_size,
        )

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.network(x)


class Generator(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.z_dim = latent_dim
        self.network = nn.Sequential(
            nn.ConvTranspose2d(
                latent_dim,
                512,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(512, momentum=0.9),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        return self.network(z)


class GAN(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.z_dim = latent_dim
        self.generator = Generator(latent_dim=latent_dim)
        self.critic = Critic()

    def forward(self, z):
        return self.generator(z)


class MNISTCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class MNISTGenerator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.z_dim = latent_dim
        self.project = nn.Linear(latent_dim, 7 * 7 * 128)
        self.network = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        z = z.view(z.size(0), self.latent_dim)
        z = self.project(z)
        z = z.view(z.size(0), 128, 7, 7)
        return self.network(z)


class MNISTGAN(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.z_dim = latent_dim
        self.generator = MNISTGenerator(latent_dim=latent_dim)
        self.critic = MNISTCritic()

    def forward(self, z):
        return self.generator(z)


class EnergyModel(nn.Module):
    def __init__(self, image_channels=3):
        super().__init__()
        self.image_channels = image_channels
        self.conv1 = nn.Conv2d(
            image_channels,
            16,
            kernel_size=5,
            stride=2,
            padding=2,
        )
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    @staticmethod
    def swish(x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        x = self.swish(self.conv1(x))
        x = self.swish(self.conv2(x))
        x = self.swish(self.conv3(x))
        x = self.swish(self.conv4(x))
        x = self.flatten(x)
        x = self.swish(self.fc1(x))
        return self.fc2(x)


class SinusoidalEmbedding(nn.Module):
    def __init__(self, num_frequencies=16):
        super().__init__()
        frequencies = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(1000.0),
                num_frequencies,
            )
        )
        self.register_buffer(
            "angular_speeds",
            2.0 * math.pi * frequencies.view(1, -1, 1, 1),
        )

    def forward(self, x):
        x = x.expand(-1, self.angular_speeds.size(1), -1, -1)
        return torch.cat(
            [
                torch.sin(self.angular_speeds * x),
                torch.cos(self.angular_speeds * x),
            ],
            dim=1,
        )


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.projection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    @staticmethod
    def swish(x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        residual = self.projection(x)
        x = self.swish(self.conv1(x))
        x = self.conv2(x)
        return x + residual


class DownBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        blocks = []
        for _ in range(block_depth):
            blocks.append(ResidualBlock(in_channels, width))
            in_channels = width
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        return self.pool(x)


class UpBlock(nn.Module):
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        blocks = []
        for _ in range(block_depth):
            blocks.append(ResidualBlock(in_channels + width, width))
            in_channels = width
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, skips):
        x = F.interpolate(
            x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        for block in self.blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x


class DiffusionUNet(nn.Module):
    def __init__(self, image_size=32, num_channels=3):
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.embedding = SinusoidalEmbedding(num_frequencies=16)

        self.down1 = DownBlock(32, block_depth=2, in_channels=64)
        self.down2 = DownBlock(64, block_depth=2, in_channels=32)
        self.down3 = DownBlock(96, block_depth=2, in_channels=64)
        self.mid1 = ResidualBlock(96, 128)
        self.mid2 = ResidualBlock(128, 128)
        self.up1 = UpBlock(96, block_depth=2, in_channels=128)
        self.up2 = UpBlock(64, block_depth=2, in_channels=96)
        self.up3 = UpBlock(32, block_depth=2, in_channels=64)
        self.final = nn.Conv2d(32, num_channels, kernel_size=1)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, noisy_images, noise_variances):
        skips = []
        x = self.initial(noisy_images)
        noise_embedding = self.embedding(noise_variances)
        noise_embedding = F.interpolate(
            noise_embedding,
            size=(self.image_size, self.image_size),
            mode="nearest",
        )
        x = torch.cat([x, noise_embedding], dim=1)
        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)
        x = self.mid1(x)
        x = self.mid2(x)
        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)
        return self.final(x)


def offset_cosine_diffusion_schedule(
    diffusion_times,
    min_signal_rate=0.02,
    max_signal_rate=0.95,
):
    start_angle = torch.acos(
        torch.tensor(
            max_signal_rate,
            dtype=diffusion_times.dtype,
            device=diffusion_times.device,
        )
    )
    end_angle = torch.acos(
        torch.tensor(
            min_signal_rate,
            dtype=diffusion_times.dtype,
            device=diffusion_times.device,
        )
    )
    diffusion_angles = start_angle + diffusion_times * (
        end_angle - start_angle
    )
    signal_rates = torch.cos(diffusion_angles)
    noise_rates = torch.sin(diffusion_angles)
    return noise_rates, signal_rates


class DiffusionModel(nn.Module):
    def __init__(
        self,
        image_size=32,
        num_channels=3,
        ema_decay=0.999,
    ):
        super().__init__()
        self.network = DiffusionUNet(
            image_size=image_size,
            num_channels=num_channels,
        )
        self.ema_network = copy.deepcopy(self.network)
        self.ema_network.requires_grad_(False)
        self.ema_network.eval()
        self.image_size = image_size
        self.num_channels = num_channels
        self.ema_decay = ema_decay
        self.register_buffer(
            "normalizer_mean",
            torch.zeros(1, num_channels, 1, 1),
        )
        self.register_buffer(
            "normalizer_std",
            torch.ones(1, num_channels, 1, 1),
        )

    def set_normalizer(self, mean, std):
        mean = torch.as_tensor(mean).reshape(1, self.num_channels, 1, 1)
        std = torch.as_tensor(std).reshape(1, self.num_channels, 1, 1)
        self.normalizer_mean.copy_(mean)
        self.normalizer_std.copy_(std)

    def normalize(self, images):
        return (images - self.normalizer_mean) / self.normalizer_std

    def denormalize(self, images):
        images = images * self.normalizer_std + self.normalizer_mean
        return images.clamp(0.0, 1.0)

    def denoise(
        self,
        noisy_images,
        noise_rates,
        signal_rates,
        use_ema=False,
    ):
        network = self.ema_network if use_ema else self.network
        predicted_noise = network(noisy_images, noise_rates.square())
        predicted_images = (
            noisy_images - noise_rates * predicted_noise
        ) / signal_rates
        return predicted_noise, predicted_images

    @torch.no_grad()
    def update_ema(self):
        for ema_parameter, parameter in zip(
            self.ema_network.parameters(),
            self.network.parameters(),
        ):
            ema_parameter.lerp_(parameter, 1.0 - self.ema_decay)
        for ema_buffer, buffer in zip(
            self.ema_network.buffers(),
            self.network.buffers(),
        ):
            ema_buffer.copy_(buffer)

    def forward(self, noisy_images, noise_variances):
        return self.network(noisy_images, noise_variances)

    @torch.no_grad()
    def generate(self, num_images=8, diffusion_steps=50):
        device = next(self.parameters()).device
        current_images = torch.randn(
            num_images,
            self.num_channels,
            self.image_size,
            self.image_size,
            device=device,
        )
        step_size = 1.0 / diffusion_steps
        predicted_images = current_images

        for step in range(diffusion_steps):
            diffusion_times = torch.full(
                (num_images, 1, 1, 1),
                1.0 - step * step_size,
                device=device,
            )
            noise_rates, signal_rates = offset_cosine_diffusion_schedule(
                diffusion_times
            )
            predicted_noise, predicted_images = self.denoise(
                current_images,
                noise_rates,
                signal_rates,
                use_ema=True,
            )
            next_times = (diffusion_times - step_size).clamp(min=0.0)
            next_noise_rates, next_signal_rates = (
                offset_cosine_diffusion_schedule(next_times)
            )
            current_images = (
                next_signal_rates * predicted_images
                + next_noise_rates * predicted_noise
            )

        return self.denormalize(predicted_images)


def get_model(
    model_name,
    num_classes=10,
    latent_dim=128,
    image_channels=3,
    image_size=32,
    ema_decay=0.999,
):
    normalized_name = model_name.strip().lower()

    if normalized_name == "fcnn":
        return FCNN(num_classes=num_classes)
    if normalized_name == "cnn":
        return CNN(num_classes=num_classes)
    if normalized_name == "enhancedcnn":
        return EnhancedCNN(num_classes=num_classes)
    if normalized_name == "vae":
        return VAE(latent_dim=latent_dim)
    if normalized_name == "gan":
        return GAN(latent_dim=latent_dim)
    if normalized_name == "mnistgan":
        return MNISTGAN(latent_dim=latent_dim)
    if normalized_name == "energy":
        return EnergyModel(image_channels=image_channels)
    if normalized_name == "diffusion":
        return DiffusionModel(
            image_size=image_size,
            num_channels=image_channels,
            ema_decay=ema_decay,
        )

    raise ValueError(
        "Unknown model_name. Expected one of: FCNN, CNN, EnhancedCNN, VAE, GAN, "
        "MNISTGAN, Energy, Diffusion."
    )
