from fastapi import APIRouter, Depends

from app.dependencies import get_embedding_model
from app.schemas import EmbeddingRequest, EmbeddingResponse
from app.services.embedding import EmbeddingModel

router = APIRouter(tags=["embedding"])


@router.post("/embedding", response_model=EmbeddingResponse)
def get_embedding(
    request: EmbeddingRequest,
    embedding_model: EmbeddingModel = Depends(get_embedding_model),
):
    embedding = embedding_model.get_embedding(request.word)

    return EmbeddingResponse(
        word=request.word,
        dimension=len(embedding),
        embedding=embedding,
    )
