from pydantic import BaseModel

class ChatRequest(BaseModel):
    sessionID: str
    msg: str

class ChatResponse(BaseModel):
    msg: str