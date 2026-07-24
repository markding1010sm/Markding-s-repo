import math

import matplotlib.pyplot as plt
import torch


def generate_energy_samples(
    model,
    input_images,
    steps=60,
    step_size=10.0,
    noise_std=0.005,
):
    was_training = model.training
    model.eval()
    images = input_images.detach()

    for _ in range(steps):
        with torch.no_grad():
            noise = torch.randn_like(images) * noise_std
            images = (images + noise).clamp(-1.0, 1.0)

        images.requires_grad_(True)
        energy = model(images)
        (gradients,) = torch.autograd.grad(
            energy,
            images,
            grad_outputs=torch.ones_like(energy),
        )

        with torch.no_grad():
            gradients = gradients.clamp(-0.03, 0.03)
            images = (images - step_size * gradients).clamp(-1.0, 1.0)

    model.train(was_training)
    return images.detach()


def generate_samples(
    model,
    device,
    num_samples=10,
    output_path=None,
    diffusion_steps=50,
    energy_steps=60,
):
    model.to(device)
    model.eval()

    if hasattr(model, "ema_network") and hasattr(model, "generate"):
        samples = model.generate(
            num_images=num_samples,
            diffusion_steps=diffusion_steps,
        ).cpu()
    elif model.__class__.__name__ == "EnergyModel":
        initial_images = torch.rand(
            num_samples,
            model.image_channels,
            32,
            32,
            device=device,
        ) * 2 - 1
        samples = generate_energy_samples(
            model,
            initial_images,
            steps=energy_steps,
        )
        samples = ((samples + 1) / 2).cpu()
    else:
        with torch.no_grad():
            if hasattr(model, "decode"):
                z = torch.randn(num_samples, model.latent_dim, device=device)
                samples = model.decode(z).cpu()
            elif hasattr(model, "generator"):
                z = torch.randn(num_samples, model.latent_dim, 1, 1, device=device)
                samples = model.generator(z).cpu()
                samples = (samples + 1) / 2
            else:
                latent_dim = getattr(model, "latent_dim", getattr(model, "z_dim"))
                z = torch.randn(num_samples, latent_dim, 1, 1, device=device)
                samples = model(z).cpu()
                samples = (samples + 1) / 2

    grid_cols = min(num_samples, 5)
    grid_rows = math.ceil(num_samples / grid_cols)
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 2, grid_rows * 2))

    if not isinstance(axes, (list, tuple)):
        axes = getattr(axes, "flat", [axes])
    else:
        axes = [axis for row in axes for axis in row]

    axes = list(axes)
    for index, axis in enumerate(axes):
        axis.axis("off")
        if index >= num_samples:
            continue

        image = samples[index].clamp(0, 1)
        if image.size(0) == 1:
            axis.imshow(image.squeeze(0).numpy(), cmap="gray")
        else:
            axis.imshow(image.permute(1, 2, 0).numpy())

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    else:
        plt.show()

    return samples
