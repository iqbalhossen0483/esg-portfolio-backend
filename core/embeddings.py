"""Generate and manage pgvector embeddings for company profiles and knowledge base."""

import asyncio

from sqlalchemy import select, update

from db.database import async_session
from db.models import Company, ComputedMetric, KnowledgeBase


def generate_company_profile_text(company: dict, metrics: dict) -> str:
    """Combine company attributes + metrics into a text description for embedding.

    Args:
        company: Dict with symbol, name, sector, sub_industry.
        metrics: Dict with sharpe, esg, volatility, etc.

    Returns:
        Text description suitable for embedding.
    """
    parts = [
        f"{company.get('name', '')} ({company.get('symbol', '')}).",
        f"Sector: {company.get('sector', 'Unknown')}.",
    ]
    if company.get("sub_industry"):
        parts.append(f"Industry: {company['sub_industry']}.")

    if metrics.get("sharpe"):
        parts.append(f"Sharpe ratio: {metrics['sharpe']:.2f}.")
    if metrics.get("esg"):
        parts.append(f"ESG score: {metrics['esg']:.1f}/100.")
    if metrics.get("volatility"):
        parts.append(f"Volatility: {metrics['volatility']:.2%}.")
    if metrics.get("annual_return"):
        parts.append(f"Annual return: {metrics['annual_return']:.2%}.")
    if metrics.get("max_drawdown"):
        parts.append(f"Max drawdown: {metrics['max_drawdown']:.2%}.")

    if metrics.get("e_score"):
        parts.append(f"Environmental: {metrics['e_score']:.1f}.")
    if metrics.get("s_score"):
        parts.append(f"Social: {metrics['s_score']:.1f}.")
    if metrics.get("g_score"):
        parts.append(f"Governance: {metrics['g_score']:.1f}.")

    return " ".join(parts)


async def generate_embedding(text: str) -> list[float]:
    """Generate embedding using Google text-embedding-004.

    Args:
        text: Text to embed.

    Returns:
        List of 768 floats.
    """
    try:
        import google.genai as genai
        from config import settings

        client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        result = await asyncio.to_thread(
            client.models.embed_content,
            model="gemini-embedding-2",
            contents=text,
        )
        return result.embeddings[0].values
    except Exception as exc:
        import logging
        logging.exception("Failed to generate embedding", exc_info=exc)
        # Return zero vector as fallback
        return [0.0] * 3072


async def update_all_company_embeddings():
    """Generate and store profile embeddings for all companies."""
    async with async_session() as db:
        result = await db.execute(
            select(
                Company.symbol, Company.name, Company.sector, Company.sub_industry,
                ComputedMetric.sharpe_252d, ComputedMetric.avg_esg_composite,
                ComputedMetric.annual_volatility, ComputedMetric.annual_return,
                ComputedMetric.max_drawdown,
                ComputedMetric.avg_e_score, ComputedMetric.avg_s_score, ComputedMetric.avg_g_score,
            ).outerjoin(
                ComputedMetric, Company.symbol == ComputedMetric.symbol
            )
        )
        rows = result.all()

        count = 0
        for row in rows:
            company = {
                "symbol": row[0], "name": row[1],
                "sector": row[2], "sub_industry": row[3],
            }
            metrics = {
                "sharpe": float(row[4]) if row[4] else None,
                "esg": float(row[5]) if row[5] else None,
                "volatility": float(row[6]) if row[6] else None,
                "annual_return": float(row[7]) if row[7] else None,
                "max_drawdown": float(row[8]) if row[8] else None,
                "e_score": float(row[9]) if row[9] else None,
                "s_score": float(row[10]) if row[10] else None,
                "g_score": float(row[11]) if row[11] else None,
            }

            text = generate_company_profile_text(company, metrics)
            embedding = await generate_embedding(text)

            await db.execute(
                update(Company)
                .where(Company.symbol == row[0])
                .values(profile_embedding=embedding)
            )
            count += 1

        await db.commit()
        return {"companies_updated": count}


async def update_knowledge_embeddings():
    """Generate embeddings for knowledge base entries that don't have one."""
    async with async_session() as db:
        result = await db.execute(
            select(KnowledgeBase).where(KnowledgeBase.embedding.is_(None))
        )
        entries = result.scalars().all()

        count = 0
        for entry in entries:
            text = f"{entry.title or ''}: {entry.content}"
            embedding = await generate_embedding(text)

            await db.execute(
                update(KnowledgeBase)
                .where(KnowledgeBase.id == entry.id)
                .values(embedding=embedding)
            )
            count += 1

        await db.commit()
        return {"entries_updated": count}
