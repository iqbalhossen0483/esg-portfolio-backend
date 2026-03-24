from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    user_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


class SessionListItem(BaseModel):
    session_id: str
    title: str
    created_at: str | None = None
    last_updated: str | None = None


class MessageItem(BaseModel):
    role: str
    content: str
    timestamp: str | None = None


class SessionMessages(BaseModel):
    session_id: str
    title: str
    created_at: str | None = None
    messages: list[MessageItem]
