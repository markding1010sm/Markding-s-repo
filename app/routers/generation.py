from fastapi import APIRouter, Depends

from app.dependencies import get_bigram_model
from app.schemas import TextGenerationRequest, TextGenerationResponse
from app.services.bigram import BigramModel

router = APIRouter(tags=["generation"])


@router.post("/generate", response_model=TextGenerationResponse)
def generate_text(
    request: TextGenerationRequest,
    bigram_model: BigramModel = Depends(get_bigram_model),
):
    generated_text = bigram_model.generate_text(
        request.start_word,
        request.length,
    )

    return TextGenerationResponse(generated_text=generated_text)
