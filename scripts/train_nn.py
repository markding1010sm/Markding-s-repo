from helper_lib.checkpoints import load_checkpoint
from helper_lib.data_loader import get_data_loader
from helper_lib.evaluator import evaluate_model
from helper_lib.model import get_model
from helper_lib.trainer import train_model
from helper_lib.utils import get_device, set_seed
import torch.nn as nn
import torch.optim as optim


def main():
    set_seed(42)
    device = get_device()

    train_loader = get_data_loader("data/train", batch_size=64)
    val_loader = get_data_loader("data/val", batch_size=64, train=False)
    test_loader = get_data_loader("data/test", batch_size=64, train=False)

    num_classes = len(train_loader.dataset.classes)
    model = get_model("CNN", num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    trained_model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device=device,
        epochs=10,
        checkpoint_dir="checkpoints",
    )

    load_checkpoint(
        trained_model,
        optimizer,
        "checkpoints/best_model.pth",
        device=device,
    )

    avg_loss, accuracy = evaluate_model(
        trained_model,
        test_loader,
        criterion,
        device=device,
    )

    print(f"Training history: {history}")
    print(f"Test loss: {avg_loss:.4f}")
    print(f"Test accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
