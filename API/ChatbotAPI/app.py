import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "API"))

if project_root not in sys.path:
    sys.path.append(project_root)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utilities.raven_chatbot import Chatbot

app = FastAPI()
chatbot = Chatbot()


class DataInput(BaseModel):
    prompt: str


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
