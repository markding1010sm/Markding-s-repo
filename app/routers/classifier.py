from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.dependencies import get_classifier_model
from app.schemas import ClassificationResponse
from app.services.classifier import ClassifierModel

router = APIRouter(tags=["classification"])


@router.post("/classify", response_model=ClassificationResponse)
def classify_image(
    file: UploadFile = File(...),
    classifier_model: ClassifierModel = Depends(get_classifier_model),
):
    try:
        image = Image.open(file.file)
        prediction = classifier_model.predict(image)
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ClassificationResponse(**prediction)
