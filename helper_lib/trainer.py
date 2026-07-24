from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from helper_lib.checkpoints import save_checkpoint
from helper_lib.evaluator import evaluate_model
from helper_lib.generator import generate_energy_samples
from helper_lib.model import offset_cosine_diffusion_schedule


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


class EnergyReplayBuffer:
    def __init__(
        self,
        model,
        device,
        image_channels=3,
        image_size=32,
        initial_size=128,
        max_size=8192,
        fresh_ratio=0.05,
    ):
        self.model = model
        self.device = torch.device(device)
        self.image_channels = image_channels
        self.image_size = image_size
        self.max_size = max_size
        self.fresh_ratio = fresh_ratio
        self.examples = torch.rand(
            initial_size,
            image_channels,
            image_size,
            image_size,
            device=self.device,
        ) * 2 - 1

    def state_dict(self, max_examples=512):
        return {
            "examples": self.examples[-max_examples:].detach().cpu(),
            "max_size": self.max_size,
            "fresh_ratio": self.fresh_ratio,
        }

    def load_state_dict(self, state):
        examples = state.get("examples")
        if examples is not None and len(examples):
            self.examples = examples.to(self.device)
        self.max_size = state.get("max_size", self.max_size)
        self.fresh_ratio = state.get("fresh_ratio", self.fresh_ratio)

    def sample(
        self,
        batch_size,
        steps=60,
        step_size=10.0,
        noise_std=0.005,
    ):
        num_fresh = max(1, round(batch_size * self.fresh_ratio))
        num_replayed = batch_size - num_fresh
        fresh_images = torch.rand(
            num_fresh,
            self.image_channels,
            self.image_size,
            self.image_size,
            device=self.device,
        ) * 2 - 1

        if num_replayed:
            indices = torch.randint(
                len(self.examples),
                (num_replayed,),
                device=self.device,
            )
            replayed_images = self.examples[indices]
            initial_images = torch.cat([fresh_images, replayed_images], dim=0)
        else:
            initial_images = fresh_images

        generated_images = generate_energy_samples(
            self.model,
            initial_images,
            steps=steps,
            step_size=step_size,
            noise_std=noise_std,
        )
        self.examples = torch.cat(
            [generated_images.detach(), self.examples],
            dim=0,
        )[: self.max_size]
        return generated_images


def _save_energy_training_checkpoint(
    model,
    optimizer,
    replay_buffer,
    epoch,
    metrics,
    checkpoint_dir,
    config,
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "replay_buffer_state": replay_buffer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "config": config,
    }
    epoch_path = checkpoint_dir / f"energy_epoch_{epoch:03d}.pth"
    api_path = checkpoint_dir / "cifar10_energy.pth"
    torch.save(checkpoint, epoch_path)
    torch.save(checkpoint, api_path)
    return epoch_path


def train_energy_model(
    model,
    data_loader,
    optimizer,
    device="cpu",
    epochs=10,
    langevin_steps=60,
    step_size=10.0,
    noise_std=0.005,
    alpha=0.1,
    checkpoint_dir="checkpoints",
    replay_buffer=None,
    start_epoch=0,
    max_batches=None,
):
    device = torch.device(device)
    model.to(device)
    replay_buffer = replay_buffer or EnergyReplayBuffer(
        model,
        device=device,
        image_channels=model.image_channels,
    )
    history = []
    config = {
        "image_channels": model.image_channels,
        "image_size": 32,
        "langevin_steps": langevin_steps,
        "step_size": step_size,
        "noise_std": noise_std,
        "alpha": alpha,
    }

    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        totals = {
            "loss": 0.0,
            "contrastive_divergence": 0.0,
            "regularization": 0.0,
            "real_energy": 0.0,
            "fake_energy": 0.0,
        }
        batches = 0
        progress = tqdm(data_loader, desc=f"Energy Epoch {epoch}/{epochs}")

        for inputs, _ in progress:
            real_images = inputs.to(device)
            real_images = (
                real_images + torch.randn_like(real_images) * noise_std
            ).clamp(-1.0, 1.0)
            fake_images = replay_buffer.sample(
                batch_size=real_images.size(0),
                steps=langevin_steps,
                step_size=step_size,
                noise_std=noise_std,
            )

            energies = model(torch.cat([real_images, fake_images], dim=0))
            real_energy, fake_energy = torch.split(
                energies,
                [real_images.size(0), fake_images.size(0)],
            )
            contrastive_divergence = (
                real_energy.mean() - fake_energy.mean()
            )
            regularization = alpha * (
                real_energy.square().mean() + fake_energy.square().mean()
            )
            loss = contrastive_divergence + regularization

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            values = {
                "loss": loss.item(),
                "contrastive_divergence": contrastive_divergence.item(),
                "regularization": regularization.item(),
                "real_energy": real_energy.mean().item(),
                "fake_energy": fake_energy.mean().item(),
            }
            for name, value in values.items():
                totals[name] += value
            batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

            if max_batches is not None and batches >= max_batches:
                break

        metrics = {
            "epoch": epoch,
            **{
                name: value / batches if batches else 0.0
                for name, value in totals.items()
            },
        }
        history.append(metrics)
        _save_energy_training_checkpoint(
            model,
            optimizer,
            replay_buffer,
            epoch,
            metrics,
            checkpoint_dir,
            config,
        )
        print(
            f"Energy epoch {epoch}: loss={metrics['loss']:.4f}, "
            f"real={metrics['real_energy']:.4f}, "
            f"fake={metrics['fake_energy']:.4f}"
        )

    return model, history, replay_buffer


def load_energy_training_checkpoint(
    model,
    optimizer,
    checkpoint_path,
    device="cpu",
    replay_buffer=None,
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if replay_buffer is not None and "replay_buffer_state" in checkpoint:
        replay_buffer.load_state_dict(checkpoint["replay_buffer_state"])
    return checkpoint


def _save_diffusion_training_checkpoint(
    model,
    optimizer,
    epoch,
    metrics,
    checkpoint_dir,
    config,
):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.network.state_dict(),
        "ema_model_state_dict": model.ema_network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "normalizer_mean": model.normalizer_mean.detach().cpu(),
        "normalizer_std": model.normalizer_std.detach().cpu(),
        "epoch": epoch,
        "metrics": metrics,
        "config": config,
    }
    epoch_path = checkpoint_dir / f"diffusion_epoch_{epoch:03d}.pth"
    api_path = checkpoint_dir / "cifar10_diffusion.pth"
    torch.save(checkpoint, epoch_path)
    torch.save(checkpoint, api_path)
    return epoch_path


def train_diffusion(
    model,
    data_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10,
    val_loader=None,
    checkpoint_dir="checkpoints",
    start_epoch=0,
    max_batches=None,
    max_val_batches=None,
):
    device = torch.device(device)
    model.to(device)
    history = []
    config = {
        "image_channels": model.num_channels,
        "image_size": model.image_size,
        "ema_decay": model.ema_decay,
    }

    for epoch in range(start_epoch + 1, epochs + 1):
        model.network.train()
        model.ema_network.eval()
        train_loss = 0.0
        train_batches = 0
        progress = tqdm(data_loader, desc=f"Diffusion Epoch {epoch}/{epochs}")

        for images, _ in progress:
            images = model.normalize(images.to(device))
            noise = torch.randn_like(images)
            diffusion_times = torch.rand(
                images.size(0),
                1,
                1,
                1,
                device=device,
            )
            noise_rates, signal_rates = offset_cosine_diffusion_schedule(
                diffusion_times
            )
            noisy_images = (
                signal_rates * images + noise_rates * noise
            )
            predicted_noise, _ = model.denoise(
                noisy_images,
                noise_rates,
                signal_rates,
                use_ema=False,
            )
            loss = criterion(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_ema()

            train_loss += loss.item()
            train_batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

            if max_batches is not None and train_batches >= max_batches:
                break

        validation_loss = None
        if val_loader is not None:
            model.network.eval()
            validation_total = 0.0
            validation_batches = 0
            with torch.no_grad():
                for images, _ in val_loader:
                    images = model.normalize(images.to(device))
                    noise = torch.randn_like(images)
                    diffusion_times = torch.rand(
                        images.size(0),
                        1,
                        1,
                        1,
                        device=device,
                    )
                    noise_rates, signal_rates = (
                        offset_cosine_diffusion_schedule(diffusion_times)
                    )
                    noisy_images = (
                        signal_rates * images + noise_rates * noise
                    )
                    predicted_noise, _ = model.denoise(
                        noisy_images,
                        noise_rates,
                        signal_rates,
                        use_ema=True,
                    )
                    validation_total += criterion(
                        predicted_noise,
                        noise,
                    ).item()
                    validation_batches += 1
                    if (
                        max_val_batches is not None
                        and validation_batches >= max_val_batches
                    ):
                        break
            validation_loss = (
                validation_total / validation_batches
                if validation_batches
                else 0.0
            )

        metrics = {
            "epoch": epoch,
            "train_loss": (
                train_loss / train_batches if train_batches else 0.0
            ),
            "validation_loss": validation_loss,
        }
        history.append(metrics)
        _save_diffusion_training_checkpoint(
            model,
            optimizer,
            epoch,
            metrics,
            checkpoint_dir,
            config,
        )
        validation_text = (
            f"{validation_loss:.4f}"
            if validation_loss is not None
            else "n/a"
        )
        print(
            f"Diffusion epoch {epoch}: "
            f"train_loss={metrics['train_loss']:.4f}, "
            f"validation_loss={validation_text}"
        )

    return model, history


def load_diffusion_training_checkpoint(
    model,
    optimizer,
    checkpoint_path,
    device="cpu",
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.network.load_state_dict(checkpoint["model_state_dict"])
    model.ema_network.load_state_dict(checkpoint["ema_model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.normalizer_mean.copy_(checkpoint["normalizer_mean"])
    model.normalizer_std.copy_(checkpoint["normalizer_std"])
    return checkpoint
