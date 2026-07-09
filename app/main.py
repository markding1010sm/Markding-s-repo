from fastapi import FastAPI

from app.routers import classifier, embedding, generation, image_generation

app = FastAPI(title="SPS GenAI")

app.include_router(generation.router)
app.include_router(image_generation.router)
app.include_router(embedding.router)
app.include_router(classifier.router)


@app.get("/", tags=["health"])
def read_root():
    return {"status": "ok", "service": "sps-genai"}
