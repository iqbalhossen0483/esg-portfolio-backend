"""ADK FunctionTools for investment knowledge base search."""

import asyncio

from db.database import async_session
from db.crud import search_knowledge


def search_knowledge_base(query: str) -> dict:
    """Searches the investment knowledge base using pgvector semantic search.

    Use this tool when the user asks about investment concepts, definitions,
    methodology, or how the system works. Covers: Sharpe ratio, Sortino ratio,
    ESG scoring, portfolio diversification, risk metrics, DRL optimization.

    Args:
        query: Natural language search query (e.g., 'what is Sharpe ratio',
               'how does ESG screening work', 'explain diversification').

    Returns:
        dict with 'results' list containing title, content, and topic
        of the most relevant knowledge base entries.
    """
    async def _search():
        async with async_session() as db:
            # For now, use text-based search until embeddings are generated
            # TODO: Replace with embedding-based search
            from sqlalchemy import select
            from db.models import KnowledgeBase

            # Simple keyword search fallback
            query_lower = query.lower()
            result = await db.execute(
                select(KnowledgeBase).limit(50)
            )
            entries = result.scalars().all()

            # Score by keyword overlap
            scored = []
            for entry in entries:
                score = 0
                content_lower = (entry.content or "").lower()
                title_lower = (entry.title or "").lower()
                for word in query_lower.split():
                    if word in content_lower:
                        score += 1
                    if word in title_lower:
                        score += 2
                if score > 0:
                    scored.append((score, entry))

            scored.sort(key=lambda x: -x[0])
            top = scored[:5]

            return [
                {
                    "title": entry.title,
                    "content": entry.content,
                    "topic": entry.topic,
                }
                for _, entry in top
            ]

    results = asyncio.run(_search())
    return {"query": query, "results": results, "count": len(results)}
