import torch
import torch.nn as nn
import torch.optim as optim
import pytest
from torch.utils.data import DataLoader, TensorDataset

from helper_lib.checkpoints import load_checkpoint, save_checkpoint
from helper_lib.generator import generate_samples
from helper_lib.model import get_model
from helper_lib.trainer import train_vae_model, vae_loss_function


@pytest.mark.parametrize("model_name", ["FCNN", "CNN", "EnhancedCNN", "VAE"])
def test_get_model_returns_torch_module(model_name):
    model = get_model(model_name)

    assert isinstance(model, nn.Module)


def test_get_model_rejects_unknown_model_name():
    with pytest.raises(ValueError):
        get_model("Transformer")


def test_cnn_matches_assignment_architecture():
    model = get_model("CNN")

    assert isinstance(model.features[0], nn.Conv2d)
    assert model.features[0].in_channels == 3
    assert model.features[0].out_channels == 16
    assert model.features[0].kernel_size == (3, 3)
    assert model.features[0].stride == (1, 1)
    assert model.features[0].padding == (1, 1)
    assert isinstance(model.features[1], nn.ReLU)
    assert isinstance(model.features[2], nn.MaxPool2d)
    assert model.features[2].kernel_size == 2
    assert model.features[2].stride == 2

    assert isinstance(model.features[3], nn.Conv2d)
    assert model.features[3].in_channels == 16
    assert model.features[3].out_channels == 32
    assert model.features[3].kernel_size == (3, 3)
    assert model.features[3].stride == (1, 1)
    assert model.features[3].padding == (1, 1)
    assert isinstance(model.features[4], nn.ReLU)
    assert isinstance(model.features[5], nn.MaxPool2d)
    assert model.features[5].kernel_size == 2
    assert model.features[5].stride == 2

    assert isinstance(model.classifier[0], nn.Flatten)
    assert isinstance(model.classifier[1], nn.Linear)
    assert model.classifier[1].in_features == 32 * 16 * 16
    assert model.classifier[1].out_features == 100
    assert isinstance(model.classifier[2], nn.ReLU)
    assert isinstance(model.classifier[3], nn.Linear)
    assert model.classifier[3].in_features == 100
    assert model.classifier[3].out_features == 10


def test_checkpoint_save_and_load_restores_training_state(tmp_path):
    model = get_model("CNN", num_classes=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    inputs = torch.randn(2, 3, 64, 64)
    labels = torch.tensor([0, 1])
    criterion = nn.CrossEntropyLoss()

    loss = criterion(model(inputs), labels)
    loss.backward()
    optimizer.step()

    checkpoint_path = save_checkpoint(
        model,
        optimizer,
        epoch=3,
        loss=0.25,
        accuracy=87.5,
        checkpoint_dir=tmp_path,
    )

    restored_model = get_model("CNN", num_classes=2)
    restored_optimizer = optim.Adam(restored_model.parameters(), lr=0.001)

    metadata = load_checkpoint(
        restored_model,
        restored_optimizer,
        checkpoint_path,
        device="cpu",
    )

    assert metadata == {"epoch": 3, "loss": 0.25, "accuracy": 87.5}
    assert restored_optimizer.state_dict()["state"]

    for original, restored in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(original, restored)


def test_vae_forward_returns_reconstruction_and_latent_parameters():
    model = get_model("VAE", latent_dim=16)
    inputs = torch.randn(2, 3, 64, 64)

    reconstructed, mu, logvar = model(inputs)

    assert reconstructed.shape == inputs.shape
    assert mu.shape == (2, 16)
    assert logvar.shape == (2, 16)


def test_vae_loss_function_returns_scalar_with_grad():
    model = get_model("VAE", latent_dim=16)
    inputs = torch.randn(2, 3, 64, 64)
    reconstructed, mu, logvar = model(inputs)

    loss = vae_loss_function(reconstructed, inputs, mu, logvar)

    assert loss.dim() == 0
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_train_vae_model_runs_one_epoch_and_saves_checkpoint(tmp_path):
    model = get_model("VAE", latent_dim=8)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    inputs = torch.randn(4, 3, 64, 64)
    labels = torch.zeros(4, dtype=torch.long)
    data_loader = DataLoader(TensorDataset(inputs, labels), batch_size=2)

    trained_model, history = train_vae_model(
        model,
        data_loader,
        optimizer,
        epochs=1,
        checkpoint_dir=tmp_path,
    )

    assert trained_model is model
    assert history[0]["epoch"] == 1
    assert history[0]["loss"] > 0
    assert (tmp_path / "model_epoch_001.pth").exists()


def test_generate_samples_saves_image_grid(tmp_path):
    model = get_model("VAE", latent_dim=8)
    output_path = tmp_path / "vae_samples.png"

    samples = generate_samples(
        model,
        device="cpu",
        num_samples=4,
        output_path=output_path,
    )

    assert samples.shape == (4, 3, 64, 64)
    assert output_path.exists()
