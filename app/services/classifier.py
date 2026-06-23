from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from helper_lib.model import get_model

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class ClassifierModel:
    def __init__(
        self,
        checkpoint_path="checkpoints/cifar10_cnn.pth",
        device="cpu",
    ):
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        self.model = get_model("CNN", num_classes=len(CIFAR10_CLASSES))
        self.transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616],
                ),
            ]
        )
        self.is_loaded = False

        if self.checkpoint_path.exists():
            self.load_checkpoint()

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

    def predict(self, image):
        if not self.is_loaded:
            raise RuntimeError(
                f"Classifier checkpoint not found: {self.checkpoint_path}"
            )

        if not isinstance(image, Image.Image):
            image = Image.open(image)

        image = image.convert("RGB")
        inputs = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(inputs)
            probabilities = torch.softmax(logits, dim=1)[0]
            confidence, class_index = torch.max(probabilities, dim=0)

        return {
            "label": CIFAR10_CLASSES[class_index.item()],
            "class_index": class_index.item(),
            "confidence": confidence.item(),
        }
