"""Chat API endpoints — powered by ADK DatabaseSessionService."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from google.genai import types

from core.auth.dependencies import get_current_user
from db.models import User
from schemas.chat import ChatRequest

router = APIRouter(prefix="/chat", tags=["chat"])

APP_NAME = "esg_advisor"


def _get_services():
    from core.adk.chat_runner import chat_runner, chat_session_service
    return chat_runner, chat_session_service


@router.post("")
async def send_message(request: ChatRequest, user: User = Depends(get_current_user)):
    """Send a message. Creates new session or continues existing one."""
    runner, session_service = _get_services()
    user_id = str(user.id)

    if request.session_id:
        session_id = request.session_id
    else:
        session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=user_id,
            state={
                "session_title": request.message[:80],
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        session_id = session.id

    response_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(
            parts=[types.Part(text=request.message)]
        ),
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    response_text = part.text

    return {"response": response_text, "session_id": session_id}


@router.get("/sessions")
async def list_sessions(user: User = Depends(get_current_user)):
    """List all chat sessions for the current user (chat history sidebar)."""
    _, session_service = _get_services()
    user_id = str(user.id)

    sessions = await session_service.list_sessions(
        app_name=APP_NAME,
        user_id=user_id,
    )

    result = []
    for s in sessions:
        state = s.state if hasattr(s, "state") and s.state else {}
        result.append({
            "session_id": s.id,
            "title": state.get("session_title", "Untitled Chat"),
            "created_at": state.get("created_at"),
            "last_updated": str(s.update_time) if hasattr(s, "update_time") else None,
        })

    result.sort(key=lambda x: x.get("last_updated") or "", reverse=True)
    return result


@router.get("/sessions/{session_id}")
async def get_session_messages(session_id: str, user: User = Depends(get_current_user)):
    """Get full conversation messages for a session."""
    _, session_service = _get_services()
    user_id = str(user.id)

    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )
    if not session:
        return {"error": "Session not found"}

    messages = []
    for event in session.events:
        if event.content and event.content.parts:
            text = ""
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
            if text:
                author = getattr(event, "author", "unknown")
                messages.append({
                    "role": "user" if author == "user" else "assistant",
                    "content": text,
                    "timestamp": str(event.timestamp) if hasattr(event, "timestamp") else None,
                })

    state = session.state if hasattr(session, "state") and session.state else {}
    return {
        "session_id": session_id,
        "title": state.get("session_title", "Untitled Chat"),
        "created_at": state.get("created_at"),
        "messages": messages,
    }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user: User = Depends(get_current_user)):
    """Delete a session and its conversation history."""
    _, session_service = _get_services()
    user_id = str(user.id)

    await session_service.delete_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )
    return {"status": "deleted", "session_id": session_id}
