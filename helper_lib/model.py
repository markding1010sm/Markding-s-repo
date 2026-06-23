import torch
import torch.nn as nn


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


def get_model(model_name, num_classes=10, latent_dim=128):
    normalized_name = model_name.strip().lower()

    if normalized_name == "fcnn":
        return FCNN(num_classes=num_classes)
    if normalized_name == "cnn":
        return CNN(num_classes=num_classes)
    if normalized_name == "enhancedcnn":
        return EnhancedCNN(num_classes=num_classes)
    if normalized_name == "vae":
        return VAE(latent_dim=latent_dim)

    raise ValueError(
        "Unknown model_name. Expected one of: FCNN, CNN, EnhancedCNN, VAE."
    )
