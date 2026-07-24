# SPS GenAI

## Running the API

Start the FastAPI app from the project root:

```bash
uv run fastapi dev
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

## Training Commands

Run training scripts as Python modules from the project root so top-level packages
like `helper_lib` are importable:

```bash
uv run python -m scripts.train_cifar10_classifier
uv run python -m scripts.train_nn
uv run python -m scripts.train_vae
uv run python -m scripts.train_mnist_gan
uv run python -m scripts.train_cifar10_energy
uv run python -m scripts.train_cifar10_diffusion
```

The CIFAR10 classifier script saves the API checkpoint to:

```text
checkpoints/cifar10_cnn.pth
```

This API checkpoint is committed with the project.

After that checkpoint exists, `POST /classify` can load it and classify uploaded
images.

The MNIST GAN script saves the image generation API checkpoint to:

```text
checkpoints/mnist_gan.pth
```

This API checkpoint is committed with the project.

After that checkpoint exists, `POST /generate_digit` can generate a PNG grid of
hand-written digit samples as an `image/png` response.

## CIFAR-10 Energy Model

The Energy-Based Model follows the Module 7 class example, adapted from
single-channel MNIST to 3-channel, 32x32 CIFAR-10 images. Its defaults match the
class settings:

```text
epochs=10
batch_size=128
langevin_steps=60
```

Training saves an API-ready checkpoint to:

```text
checkpoints/cifar10_energy.pth
```

This API checkpoint is committed with the project so `/generate_energy` works
after cloning or deployment. Per-epoch checkpoints and generated sample images
remain ignored.

The long-running settings can be overridden or resumed:

```bash
uv run python -m scripts.train_cifar10_energy --epochs 1 --max-batches 2
uv run python -m scripts.train_cifar10_energy \
  --epochs 10 \
  --resume checkpoints/energy_epoch_005.pth
```

After training, `POST /generate_energy` accepts JSON such as
`{"num_images": 8}` and returns a PNG grid. The allowed range is 1 to 16 images.

## CIFAR-10 Diffusion Model

The Diffusion Model follows the Module 7 helper-library activity. It uses a
noise-conditioned U-Net, an offset-cosine diffusion schedule, L1 noise
prediction loss, and exponential moving average weights for inference.
Defaults are:

```text
epochs=10
batch_size=64
diffusion_api_steps=50
```

Training saves an API-ready checkpoint to:

```text
checkpoints/cifar10_diffusion.pth
```

This API checkpoint is committed with the project so `/generate_diffusion`
works after cloning or deployment. Per-epoch checkpoints and generated sample
images remain ignored.

The training can also be shortened or resumed:

```bash
uv run python -m scripts.train_cifar10_diffusion \
  --epochs 1 \
  --max-batches 2 \
  --max-val-batches 1
uv run python -m scripts.train_cifar10_diffusion \
  --epochs 10 \
  --resume checkpoints/diffusion_epoch_005.pth
```

After training, `POST /generate_diffusion` accepts JSON such as
`{"num_images": 8}` and returns a PNG grid. Both CIFAR-10 endpoints return HTTP
503 with the relevant training command when their checkpoint is unavailable or
incompatible.

The full Energy and Diffusion defaults can take several hours on a laptop.
Checkpoints are saved after every epoch so interrupted runs can continue.
