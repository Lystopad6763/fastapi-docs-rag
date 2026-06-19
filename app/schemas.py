from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for /chat/stream. This is a stateless Q&A bot (no history) — a single question."""
    message: str = Field(..., min_length=1, description="Question about the FastAPI documentation")