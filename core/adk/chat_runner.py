"""Chat Runner setup with DatabaseSessionService for production."""

from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService

from config import settings

chat_session_service: DatabaseSessionService | None = None
chat_runner: Runner | None = None

def init_chat_runner():
    """Called once from lifespan, inside the running event loop."""
    global chat_session_service, chat_runner
    try:
        from .chat_agents import root_agent

        chat_session_service = DatabaseSessionService(
            db_url=settings.DATABASE_URL,
        )
        chat_runner = Runner(
            agent=root_agent,
            app_name="esg_advisor",
            session_service=chat_session_service,
        )
        print("✅ Chat runner initialized successfully")
    except Exception as e:
        print(f"❌ Chat runner initialization FAILED: {e}")
        raise

def get_chat_runner() -> Runner:
    if chat_runner is None:
        raise RuntimeError("Chat runner not initialized. Call init_chat_runner() in lifespan.")
    return chat_runner

def get_chat_session_service() -> DatabaseSessionService:
    if chat_session_service is None:
        raise RuntimeError("Chat session service not initialized. Call init_chat_runner() in lifespan.")
    return chat_session_service

