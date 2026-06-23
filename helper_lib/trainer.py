import torch
import torch.nn.functional as F
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
