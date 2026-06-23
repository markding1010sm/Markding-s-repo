import marimo

__generated_with = "0.10.0"
app = marimo.App()


@app.cell
def _():
    from helper_lib.data_loader import get_data_loader
    from helper_lib.generator import generate_samples
    from helper_lib.model import get_model
    from helper_lib.trainer import train_vae_model
    from helper_lib.utils import get_device, set_seed
    import torch.optim as optim

    return (
        generate_samples,
        get_data_loader,
        get_device,
        get_model,
        optim,
        set_seed,
        train_vae_model,
    )


@app.cell
def _(get_device, get_model, optim, set_seed):
    set_seed(42)
    device = get_device()
    vae = get_model("VAE")
    optimizer = optim.Adam(vae.parameters(), lr=0.001)
    device, optimizer, vae

    return device, optimizer, vae


@app.cell
def _(generate_samples, device, vae):
    # Run after training, or use it to inspect untrained VAE output.
    samples = generate_samples(vae, device, num_samples=10)
    samples.shape

    return (samples,)


if __name__ == "__main__":
    app.run()
