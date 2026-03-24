"""Chat Runner setup with DatabaseSessionService for production."""

from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService

from config import settings
from .chat_agents import root_agent

# Production: PostgreSQL-backed sessions (auto-creates tables)
chat_session_service = DatabaseSessionService(
    db_url=settings.DATABASE_URL,
)

chat_runner = Runner(
    agent=root_agent,
    app_name="esg_advisor",
    session_service=chat_session_service,
)
