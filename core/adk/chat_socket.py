"""Socket.IO handlers for chat: thinking steps + response streaming.

Mounted on the /chat namespace so it doesn't collide with /training.
"""

from datetime import datetime, timezone

from google.genai import types
from socketio.exceptions import ConnectionRefusedError

from core.auth.security import decode_token
from core.logging import get_logger

log = get_logger(__name__)

NAMESPACE = "/chat"


def _extract_token(auth, environ) -> str | None:
    if auth and isinstance(auth, dict):
        token = auth.get("token")
        if token:
            return token
    query = environ.get("QUERY_STRING", "")
    for param in query.split("&"):
        if param.startswith("token="):
            return param.split("=", 1)[1]
    return None


def register_chat_handlers(sio):
    """Register Socket.IO event handlers for chat real-time communication."""

    @sio.on("connect", namespace=NAMESPACE)
    async def on_connect(sid, environ, auth=None):
        token = _extract_token(auth, environ)
        if not token:
            raise ConnectionRefusedError("authentication required")

        try:
            payload = decode_token(token)
        except Exception:
            raise ConnectionRefusedError("invalid token")

        await sio.save_session(sid, {
            "user_id": payload.get("user_id"),
            "email": payload.get("email"),
            "role": payload.get("role"),
            "authenticated": True,
        }, namespace=NAMESPACE)
        log.info("chat socket connected sid=%s user_id=%s",
                 sid, payload.get("user_id"))

    @sio.on("chat:send_message", namespace=NAMESPACE)
    async def handle_chat_message(sid, data):
        """Process a chat message through the multi-agent pipeline with live thinking steps."""
        session_data = await sio.get_session(sid, namespace=NAMESPACE)
        user_id = session_data.get("user_id", "anonymous")
        session_id = data.get("session_id")
        message = data.get("message", "")

        if not message:
            await sio.emit("chat:error", {"error": "Empty message"},
                           room=sid, namespace=NAMESPACE)
            return

        from .chat_runner import chat_runner, chat_session_service

        if not session_id:
            adk_session = await chat_session_service.create_session(
                app_name="esg_advisor",
                user_id=user_id,
                state={
                    "session_title": message[:80],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            session_id = adk_session.id

        step_count = 0
        final_response = ""

        try:
            async for event in chat_runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(
                    parts=[types.Part(text=message)]
                ),
            ):
                if not event.content:
                    continue

                author = getattr(event, "author", None) or ""

                if author and author != "user" and author != "ResponseBeautifier":
                    step_count += 1

                    actions = getattr(event, "actions", None)
                    if actions:
                        tool_calls = getattr(actions, "tool_calls", None) or []
                        for tc in tool_calls:
                            fc = getattr(tc, "function_call", None)
                            if fc:
                                await sio.emit("chat:thinking_step", {
                                    "step": step_count,
                                    "agent": author,
                                    "tool": fc.name,
                                    "status": "calling",
                                    "detail": f"Calling {fc.name}...",
                                }, room=sid, namespace=NAMESPACE)
                    else:
                        status_map = {
                            "InvestmentAdvisorRouter": "Routing to specialist...",
                            "SectorAnalyst": "Analyzing sectors...",
                            "CompanyAnalyst": "Analyzing companies...",
                            "PortfolioOptimizer": "Optimizing portfolio...",
                            "InvestmentEducator": "Searching knowledge base...",
                            "QualityJudge": "Validating response...",
                        }
                        detail = status_map.get(author, f"{author} processing...")
                        await sio.emit("chat:thinking_step", {
                            "step": step_count,
                            "agent": author,
                            "tool": None,
                            "status": "processing",
                            "detail": detail,
                        }, room=sid, namespace=NAMESPACE)

                if author == "ResponseBeautifier" and event.content and event.content.parts:
                    await sio.emit("chat:response_start", {
                        "session_id": session_id,
                    }, room=sid, namespace=NAMESPACE)

                    for part in event.content.parts:
                        if part.text:
                            final_response = part.text
                            for i in range(0, len(part.text), 50):
                                chunk = part.text[i:i + 50]
                                await sio.emit("chat:response_token", {
                                    "token": chunk,
                                }, room=sid, namespace=NAMESPACE)

        except Exception as e:
            log.exception("chat handler failed sid=%s", sid)
            await sio.emit("chat:error", {
                "error": str(e),
                "session_id": session_id,
            }, room=sid, namespace=NAMESPACE)
            return

        await sio.emit("chat:response_end", {
            "session_id": session_id,
            "full_response": final_response,
        }, room=sid, namespace=NAMESPACE)
