from fastapi import APIRouter, Depends, HTTPException, Response

from app.dependencies import get_mnist_image_generator
from app.schemas import ImageGenerationRequest
from app.services.image_generation import MNISTImageGenerator

router = APIRouter(tags=["image generation"])


@router.post(
    "/generate_digit",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
def generate_digit(
    request: ImageGenerationRequest,
    image_generator: MNISTImageGenerator = Depends(get_mnist_image_generator),
):
    try:
        generated_image = image_generator.generate_digits(request.num_images)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return Response(content=generated_image, media_type="image/png")
