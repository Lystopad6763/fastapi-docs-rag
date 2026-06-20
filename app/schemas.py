from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for /chat/stream. This is a stateless Q&A bot (no history) — a single question."""
    message: str = Field(..., min_length=1, description="Question about the FastAPI documentation")

    # Pre-fills Swagger UI's "Try it out" body so it can be sent as-is (no hand-typed JSON).
    model_config = {
        "json_schema_extra": {"example": {"message": "How do I upload a file in FastAPI?"}}
    }