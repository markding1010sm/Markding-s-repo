from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loader(data_dir, batch_size=32, train=True):
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    transform_steps = [
        transforms.Resize((64, 64)),
    ]

    if train:
        transform_steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
            ]
        )

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    dataset = datasets.ImageFolder(
        root=data_path,
        transform=transforms.Compose(transform_steps),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
    )
