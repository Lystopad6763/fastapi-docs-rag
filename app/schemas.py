from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Тіло запиту /chat/stream. Це Q&A-бот (без історії) — лише одне питання."""
    message: str = Field(..., min_length=1, description="Питання про FastAPI-документацію")