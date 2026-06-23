from pathlib import Path

import torch


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    accuracy,
    checkpoint_dir="checkpoints",
    filename=None,
):
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"model_epoch_{epoch:03d}.pth"

    save_path = checkpoint_path / filename
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy,
        },
        save_path,
    )

    return save_path


def load_checkpoint(model, optimizer, checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "epoch": checkpoint["epoch"],
        "loss": checkpoint["loss"],
        "accuracy": checkpoint["accuracy"],
    }
