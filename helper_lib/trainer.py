import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from helper_lib.checkpoints import save_checkpoint
from helper_lib.evaluator import evaluate_model


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10,
    checkpoint_dir="checkpoints",
):
    model.to(device)
    best_accuracy = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = outputs.max(dim=1)
            correct += (predicted == labels).sum().item()
            total += batch_size

        train_loss = running_loss / total if total else 0.0
        train_accuracy = (correct / total) * 100 if total else 0.0
        val_loss, val_accuracy = evaluate_model(
            model,
            val_loader,
            criterion,
            device=device,
        )

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(metrics)

        save_checkpoint(
            model,
            optimizer,
            epoch,
            val_loss,
            val_accuracy,
            checkpoint_dir=checkpoint_dir,
        )

        if val_accuracy >= best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                val_accuracy,
                checkpoint_dir=checkpoint_dir,
                filename="best_model.pth",
            )

    return model, history


def vae_loss_function(reconstructed, original, mu, logvar):
    reconstruction_loss = F.mse_loss(
        reconstructed,
        original,
        reduction="sum",
    )
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence


def train_vae_model(
    model,
    data_loader,
    optimizer,
    device="cpu",
    epochs=10,
    checkpoint_dir="checkpoints",
):
    model.to(device)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0

        for inputs, _ in tqdm(data_loader, desc=f"VAE Epoch {epoch}/{epochs}"):
            inputs = inputs.to(device)
            optimizer.zero_grad()

            reconstructed, mu, logvar = model(inputs)
            loss = vae_loss_function(reconstructed, inputs, mu, logvar)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item()
            total += batch_size

        avg_loss = running_loss / total if total else 0.0
        metrics = {"epoch": epoch, "loss": avg_loss}
        history.append(metrics)

        save_checkpoint(
            model,
            optimizer,
            epoch,
            avg_loss,
            accuracy=0.0,
            checkpoint_dir=checkpoint_dir,
        )

    return model, history


class _GANOptimizers:
    def __init__(self, generator_optimizer, critic_optimizer):
        self.generator_optimizer = generator_optimizer
        self.critic_optimizer = critic_optimizer

    def state_dict(self):
        return {
            "generator_optimizer": self.generator_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }


def train_gan(
    model,
    data_loader,
    criterion=None,
    optimizer=None,
    device="cpu",
    epochs=10,
    lr=5e-5,
    n_critic=5,
    clip_value=0.01,
    checkpoint_dir="checkpoints",
):
    model.to(device)
    model.train()
    history = []

    if optimizer is None:
        generator_optimizer = optim.RMSprop(model.generator.parameters(), lr=lr)
        critic_optimizer = optim.RMSprop(model.critic.parameters(), lr=lr)
    elif isinstance(optimizer, dict):
        generator_optimizer = optimizer["generator"]
        critic_optimizer = optimizer["critic"]
    else:
        generator_optimizer, critic_optimizer = optimizer

    for epoch in range(1, epochs + 1):
        running_critic_loss = 0.0
        running_generator_loss = 0.0
        batches = 0

        for real, _ in tqdm(data_loader, desc=f"GAN Epoch {epoch}/{epochs}"):
            real = real.to(device)
            batch_size = real.size(0)

            for _ in range(n_critic):
                noise = torch.randn(
                    batch_size,
                    model.latent_dim,
                    1,
                    1,
                    device=device,
                )
                fake = model.generator(noise).detach()
                critic_real = model.critic(real).mean()
                critic_fake = model.critic(fake).mean()
                critic_loss = -(critic_real - critic_fake)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                for parameter in model.critic.parameters():
                    parameter.data.clamp_(-clip_value, clip_value)

            noise = torch.randn(
                batch_size,
                model.latent_dim,
                1,
                1,
                device=device,
            )
            fake = model.generator(noise)
            generator_loss = -model.critic(fake).mean()

            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

            running_critic_loss += critic_loss.item()
            running_generator_loss += generator_loss.item()
            batches += 1

        avg_critic_loss = running_critic_loss / batches if batches else 0.0
        avg_generator_loss = running_generator_loss / batches if batches else 0.0
        metrics = {
            "epoch": epoch,
            "critic_loss": avg_critic_loss,
            "generator_loss": avg_generator_loss,
        }
        history.append(metrics)

        save_checkpoint(
            model,
            _GANOptimizers(generator_optimizer, critic_optimizer),
            epoch,
            avg_generator_loss,
            accuracy=0.0,
            checkpoint_dir=checkpoint_dir,
        )

    return model, history
