from fastapi import APIRouter, Depends, HTTPException, Response

from app.dependencies import (
    get_cifar10_diffusion_generator,
    get_cifar10_energy_generator,
    get_mnist_image_generator,
)
from app.schemas import CIFARImageGenerationRequest, ImageGenerationRequest
from app.services.cifar10_generation import (
    CIFAR10DiffusionGenerator,
    CIFAR10EnergyGenerator,
)
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


@router.post(
    "/generate_energy",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
def generate_energy(
    request: CIFARImageGenerationRequest,
    image_generator: CIFAR10EnergyGenerator = Depends(
        get_cifar10_energy_generator
    ),
):
    try:
        generated_image = image_generator.generate_images(request.num_images)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return Response(content=generated_image, media_type="image/png")


@router.post(
    "/generate_diffusion",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
def generate_diffusion(
    request: CIFARImageGenerationRequest,
    image_generator: CIFAR10DiffusionGenerator = Depends(
        get_cifar10_diffusion_generator
    ),
):
    try:
        generated_image = image_generator.generate_images(request.num_images)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return Response(content=generated_image, media_type="image/png")
