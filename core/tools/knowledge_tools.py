"""ADK FunctionTools for investment knowledge base search. Async + pgvector."""

from sqlalchemy import select

from core.embeddings import generate_embedding
from core.logging import get_logger
from db.crud import search_knowledge
from db.database import async_session
from db.models import KnowledgeBase

log = get_logger(__name__)


async def _keyword_fallback(db, query: str, limit: int) -> list[dict]:
    rows = (await db.execute(
        select(KnowledgeBase).limit(200)
    )).scalars().all()

    qlower = query.lower()
    scored = []
    for entry in rows:
        score = 0
        content = (entry.content or "").lower()
        title = (entry.title or "").lower()
        for word in qlower.split():
            if word in content:
                score += 1
            if word in title:
                score += 2
        if score:
            scored.append((score, entry))
    scored.sort(key=lambda x: -x[0])
    return [
        {"title": e.title, "content": e.content, "topic": e.topic}
        for _, e in scored[:limit]
    ]


async def search_knowledge_base(query: str) -> dict:
    """Searches the investment knowledge base via pgvector semantic similarity.

    Falls back to keyword scoring when embeddings are unavailable.

    Args:
        query: Natural language search query.

    Returns:
        dict with 'results' list (title, content, topic) of the most relevant entries.
    """
    limit = 5
    embedding = await generate_embedding(query)

    async with async_session() as db:
        results: list[dict] = []
        if embedding and any(v != 0.0 for v in embedding):
            entries = await search_knowledge(db, embedding, limit=limit)
            results = [
                {"title": e.title, "content": e.content, "topic": e.topic}
                for e in entries
                if e.embedding is not None
            ]
        if not results:
            log.info("knowledge search using keyword fallback query=%r", query)
            results = await _keyword_fallback(db, query, limit)

    return {"query": query, "results": results, "count": len(results)}
