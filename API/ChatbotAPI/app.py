import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "API"))

if project_root not in sys.path:
    sys.path.append(project_root)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utilities.raven_chatbot import Chatbot

app = FastAPI()
chatbot = Chatbot()

origins = ["http://localhost:4200", "https://localhost:4200"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DataInput(BaseModel):
    prompt: str


@app.get("/")
async def root():
    return "Welcome to our Chatbot API"


@app.post("/send_prompt")
def send_prompt(data: DataInput):
    try:
        response = chatbot.handle_input(data.model_dump()["prompt"])

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=5001)
