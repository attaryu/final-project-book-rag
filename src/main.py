from fastapi import FastAPI
from src.modules.chat.router import router as chat_router

app = FastAPI(
    title="Final Project", description="Chatbot for my final project", version="1.0.0"
)


@app.get("/")
def root():
    return {"message": "System ready, check the /docs to try!"}


app.include_router(chat_router)
