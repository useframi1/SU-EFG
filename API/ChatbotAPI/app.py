import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "API"))
if project_root not in sys.path:
    sys.path.append(project_root)

import uuid
from fastapi import FastAPI, HTTPException, Request, Response, Depends, Cookie
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utilities.chatbot import Chatbot


app = FastAPI()

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
    isSinglePrompt: bool


chatbot_sessions = {}


@app.get("/")
async def root(request: Request, response: Response):
    try:
        user_id = request.cookies.get("user_id")
        if not user_id:
            user_id = str(uuid.uuid4())
            response.set_cookie(
                key="user_id",
                value=user_id,
                httponly=True,
                secure=True,
                samesite="None",
            )

        chatbot_sessions[user_id] = Chatbot()
        return user_id
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_user_id(user_id: str = Cookie(None)):
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID not found")
    return user_id


@app.post("/send_prompt")
async def send_prompt(data: DataInput, user_id: str = Depends(get_user_id)):
    if user_id not in chatbot_sessions:
        raise HTTPException(status_code=400, detail="Chatbot session not found")

    chatbot = chatbot_sessions[user_id]

    try:
        response = chatbot.run_conversation(
            data.model_dump()["prompt"], data.model_dump()["isSinglePrompt"]
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=5001)
