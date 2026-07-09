from pydantic import BaseModel, Field


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int = Field(ge=1, le=100)


class TextGenerationResponse(BaseModel):
    generated_text: str


class ImageGenerationRequest(BaseModel):
    num_images: int = Field(default=16, ge=1, le=64)


class EmbeddingRequest(BaseModel):
    word: str


class EmbeddingResponse(BaseModel):
    word: str
    dimension: int
    embedding: list[float]


class ClassificationResponse(BaseModel):
    label: str
    class_index: int
    confidence: float
