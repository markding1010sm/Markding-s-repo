from fastapi.testclient import TestClient
from io import BytesIO
from PIL import Image

from app.dependencies import get_classifier_model
from app.dependencies import get_embedding_model
from app.main import app


class FakeEmbeddingModel:
    def get_embedding(self, input_word):
        return [0.1, 0.2, 0.3]


def override_embedding_model():
    return FakeEmbeddingModel()


class FakeClassifierModel:
    def predict(self, image):
        return {
            "label": "cat",
            "class_index": 3,
            "confidence": 0.9,
        }


def override_classifier_model():
    return FakeClassifierModel()


client = TestClient(app)


def test_root_returns_status():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "sps-genai"}


def test_generate_returns_generated_text():
    response = client.post(
        "/generate",
        json={"start_word": "the", "length": 3},
    )

    assert response.status_code == 200
    assert "generated_text" in response.json()


def test_generate_rejects_invalid_length():
    response = client.post(
        "/generate",
        json={"start_word": "the", "length": 0},
    )

    assert response.status_code == 422


def test_embedding_returns_vector_metadata():
    app.dependency_overrides[get_embedding_model] = override_embedding_model

    response = client.post(
        "/embedding",
        json={"word": "example"},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "word": "example",
        "dimension": 3,
        "embedding": [0.1, 0.2, 0.3],
    }


def test_classify_returns_prediction():
    image = Image.new("RGB", (64, 64), color="red")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    app.dependency_overrides[get_classifier_model] = override_classifier_model

    response = client.post(
        "/classify",
        files={"file": ("sample.png", buffer, "image/png")},
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "label": "cat",
        "class_index": 3,
        "confidence": 0.9,
    }
