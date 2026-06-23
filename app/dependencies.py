from functools import lru_cache

from app.services.classifier import ClassifierModel
from app.services.bigram import BigramModel
from app.services.embedding import EmbeddingModel

CORPUS = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantes, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]


@lru_cache
def get_bigram_model() -> BigramModel:
    return BigramModel(CORPUS)


@lru_cache
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()


@lru_cache
def get_classifier_model() -> ClassifierModel:
    return ClassifierModel()
