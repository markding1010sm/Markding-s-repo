import math

import matplotlib.pyplot as plt
import torch


def generate_samples(model, device, num_samples=10, output_path=None):
    model.to(device)
    model.eval()

    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim, device=device)
        samples = model.decode(z).cpu()

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

        image = samples[index].permute(1, 2, 0).clamp(0, 1).numpy()
        axis.imshow(image)

    fig.tight_layout()

    if output_path:
        fig.savefig(output_path)
        plt.close(fig)
    else:
        plt.show()

    return samples
