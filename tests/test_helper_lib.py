import torch
import torch.nn as nn
import torch.optim as optim
import pytest
from torch.utils.data import DataLoader, TensorDataset

from app.services.rnn import RNNTextGenerator
from helper_lib.checkpoints import load_checkpoint, save_checkpoint
from helper_lib.generator import generate_energy_samples, generate_samples
from helper_lib.model import (
    get_model,
    offset_cosine_diffusion_schedule,
)
from helper_lib.trainer import (
    train_diffusion,
    train_energy_model,
    train_gan,
    train_vae_model,
    vae_loss_function,
)


@pytest.mark.parametrize(
    "model_name",
    [
        "FCNN",
        "CNN",
        "EnhancedCNN",
        "VAE",
        "GAN",
        "MNISTGAN",
        "Energy",
        "Diffusion",
    ],
)
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


def test_gan_model_generates_images_and_scores_them():
    model = get_model("GAN", latent_dim=16)
    noise = torch.randn(2, 16)

    generated = model.generator(noise)
    scores = model.critic(generated)

    assert hasattr(model, "generator")
    assert hasattr(model, "critic")
    assert generated.shape == (2, 3, 64, 64)
    assert scores.shape == (2, 1)


def test_train_gan_runs_one_epoch_and_saves_checkpoint(tmp_path):
    model = get_model("GAN", latent_dim=8)
    inputs = (torch.rand(2, 3, 64, 64) * 2) - 1
    labels = torch.zeros(2, dtype=torch.long)
    data_loader = DataLoader(TensorDataset(inputs, labels), batch_size=2)

    trained_model, history = train_gan(
        model,
        data_loader,
        epochs=1,
        n_critic=1,
        checkpoint_dir=tmp_path,
    )

    assert trained_model is model
    assert history[0]["epoch"] == 1
    assert "critic_loss" in history[0]
    assert "generator_loss" in history[0]
    assert (tmp_path / "model_epoch_001.pth").exists()


def test_mnist_gan_model_generates_digits_and_scores_them():
    model = get_model("MNISTGAN", latent_dim=16)
    noise = torch.randn(2, 16)

    generated = model.generator(noise)
    scores = model.critic(generated)

    assert hasattr(model, "generator")
    assert hasattr(model, "critic")
    assert generated.shape == (2, 1, 28, 28)
    assert scores.shape == (2, 1)


def test_mnist_gan_matches_assignment_architecture():
    model = get_model("MNISTGAN", latent_dim=100)

    assert isinstance(model.generator.project, nn.Linear)
    assert model.generator.project.in_features == 100
    assert model.generator.project.out_features == 7 * 7 * 128
    assert isinstance(model.generator.network[0], nn.ConvTranspose2d)
    assert model.generator.network[0].in_channels == 128
    assert model.generator.network[0].out_channels == 64
    assert model.generator.network[0].kernel_size == (4, 4)
    assert model.generator.network[0].stride == (2, 2)
    assert model.generator.network[0].padding == (1, 1)
    assert isinstance(model.generator.network[1], nn.BatchNorm2d)
    assert isinstance(model.generator.network[2], nn.ReLU)
    assert isinstance(model.generator.network[3], nn.ConvTranspose2d)
    assert model.generator.network[3].in_channels == 64
    assert model.generator.network[3].out_channels == 1
    assert isinstance(model.generator.network[4], nn.Tanh)

    assert isinstance(model.critic.features[0], nn.Conv2d)
    assert model.critic.features[0].in_channels == 1
    assert model.critic.features[0].out_channels == 64
    assert model.critic.features[0].kernel_size == (4, 4)
    assert model.critic.features[0].stride == (2, 2)
    assert model.critic.features[0].padding == (1, 1)
    assert isinstance(model.critic.features[1], nn.LeakyReLU)
    assert isinstance(model.critic.features[2], nn.Conv2d)
    assert model.critic.features[2].in_channels == 64
    assert model.critic.features[2].out_channels == 128
    assert isinstance(model.critic.features[3], nn.BatchNorm2d)
    assert isinstance(model.critic.features[4], nn.LeakyReLU)
    assert isinstance(model.critic.classifier[0], nn.Flatten)
    assert isinstance(model.critic.classifier[1], nn.Linear)
    assert model.critic.classifier[1].in_features == 128 * 7 * 7
    assert model.critic.classifier[1].out_features == 1


def test_train_mnist_gan_runs_one_epoch_and_saves_checkpoint(tmp_path):
    model = get_model("MNISTGAN", latent_dim=8)
    inputs = (torch.rand(2, 1, 28, 28) * 2) - 1
    labels = torch.zeros(2, dtype=torch.long)
    data_loader = DataLoader(TensorDataset(inputs, labels), batch_size=2)

    trained_model, history = train_gan(
        model,
        data_loader,
        epochs=1,
        n_critic=1,
        checkpoint_dir=tmp_path,
    )

    assert trained_model is model
    assert history[0]["epoch"] == 1
    assert "critic_loss" in history[0]
    assert "generator_loss" in history[0]
    assert (tmp_path / "model_epoch_001.pth").exists()


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


def test_generate_samples_saves_gan_image_grid(tmp_path):
    model = get_model("GAN", latent_dim=8)
    output_path = tmp_path / "gan_samples.png"

    samples = generate_samples(
        model,
        device="cpu",
        num_samples=4,
        output_path=output_path,
    )

    assert samples.shape == (4, 3, 64, 64)
    assert output_path.exists()


def test_generate_samples_saves_mnist_gan_image_grid(tmp_path):
    model = get_model("MNISTGAN", latent_dim=8)
    output_path = tmp_path / "mnist_gan_samples.png"

    samples = generate_samples(
        model,
        device="cpu",
        num_samples=4,
        output_path=output_path,
    )

    assert samples.shape == (4, 1, 28, 28)
    assert output_path.exists()


def test_rnn_text_generator_returns_requested_word_count():
    model = RNNTextGenerator(
        [
            "the count of monte cristo is a novel",
            "the count seeks revenge",
            "monte cristo tells a story",
        ],
        sequence_length=3,
        epochs=1,
    )

    generated_text = model.generate_text("the", length=4)

    assert len(generated_text.split()) == 4


def test_energy_model_scores_cifar10_images():
    model = get_model("Energy")

    energies = model(torch.randn(2, 3, 32, 32))

    assert energies.shape == (2, 1)


def test_energy_langevin_sampling_preserves_image_shape_and_range():
    model = get_model("Energy")
    initial_images = (torch.rand(2, 3, 32, 32) * 2) - 1

    samples = generate_energy_samples(
        model,
        initial_images,
        steps=2,
        step_size=1.0,
        noise_std=0.001,
    )

    assert samples.shape == initial_images.shape
    assert samples.min() >= -1
    assert samples.max() <= 1


def test_diffusion_model_predicts_noise_and_generates_images():
    model = get_model("Diffusion")
    images = torch.randn(2, 3, 32, 32)
    noise_variances = torch.rand(2, 1, 1, 1)

    predicted_noise = model(images, noise_variances)
    samples = model.generate(num_images=2, diffusion_steps=2)

    assert predicted_noise.shape == images.shape
    assert samples.shape == images.shape
    assert samples.min() >= 0
    assert samples.max() <= 1


def test_offset_cosine_schedule_returns_valid_rates():
    diffusion_times = torch.tensor([0.0, 0.5, 1.0]).view(-1, 1, 1, 1)

    noise_rates, signal_rates = offset_cosine_diffusion_schedule(
        diffusion_times
    )

    assert noise_rates.shape == diffusion_times.shape
    assert signal_rates.shape == diffusion_times.shape
    assert torch.all(noise_rates >= 0)
    assert torch.all(noise_rates <= 1)
    assert torch.all(signal_rates >= 0)
    assert torch.all(signal_rates <= 1)


def test_train_energy_model_runs_one_batch_and_saves_checkpoint(tmp_path):
    model = get_model("Energy")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    inputs = (torch.rand(2, 3, 32, 32) * 2) - 1
    labels = torch.zeros(2, dtype=torch.long)
    data_loader = DataLoader(TensorDataset(inputs, labels), batch_size=2)

    trained_model, history, _ = train_energy_model(
        model,
        data_loader,
        optimizer,
        epochs=1,
        langevin_steps=1,
        checkpoint_dir=tmp_path,
        max_batches=1,
    )

    assert trained_model is model
    assert history[0]["epoch"] == 1
    assert (tmp_path / "energy_epoch_001.pth").exists()
    assert (tmp_path / "cifar10_energy.pth").exists()


def test_train_diffusion_runs_one_batch_and_saves_checkpoint(tmp_path):
    model = get_model("Diffusion")
    model.set_normalizer(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616),
    )
    optimizer = optim.AdamW(model.network.parameters(), lr=1e-3)
    inputs = torch.rand(2, 3, 32, 32)
    labels = torch.zeros(2, dtype=torch.long)
    data_loader = DataLoader(TensorDataset(inputs, labels), batch_size=2)

    trained_model, history = train_diffusion(
        model,
        data_loader,
        nn.L1Loss(),
        optimizer,
        epochs=1,
        checkpoint_dir=tmp_path,
        max_batches=1,
    )

    assert trained_model is model
    assert history[0]["epoch"] == 1
    assert (tmp_path / "diffusion_epoch_001.pth").exists()
    assert (tmp_path / "cifar10_diffusion.pth").exists()
