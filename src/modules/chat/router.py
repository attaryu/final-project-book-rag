from fastapi import APIRouter

from src.modules.chat.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
async def ask_chatbot(payload: ChatRequest):
    user_message = payload.message
    reply = f"Your question: {user_message}. (RAG not connected yet)"

    return ChatResponse(reply=reply)
