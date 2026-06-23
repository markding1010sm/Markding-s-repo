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
```

The CIFAR10 classifier script saves the API checkpoint to:

```text
checkpoints/cifar10_cnn.pth
```

After that checkpoint exists, `POST /classify` can load it and classify uploaded
images.
