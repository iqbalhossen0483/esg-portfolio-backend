"""ADK FunctionTools for the training pipeline.
Each function's docstring is read by the LLM to decide when/how to use it.
All storage tools are async — they run inside the ADK event loop.
"""

import json
from datetime import datetime, date

from core.embeddings import generate_embedding
from core.logging import get_logger
from db.crud import (
    bulk_upsert_companies,
    bulk_upsert_esg_scores,
    bulk_upsert_prices,
    create_knowledge_entry,
)
from db.database import async_session

log = get_logger(__name__)


def extract_tabular_data(raw_content: str, column_mapping: str) -> dict:
    """Extract structured rows from tabular data content.

    Args:
        raw_content: Raw text containing tab-separated or comma-separated rows.
        column_mapping: JSON string mapping detected columns to standard names,
            e.g. '{"col_0": "date", "col_1": "symbol", "col_2": "open"}'

    Returns:
        dict with 'rows' containing list of dicts with standardized column names,
        and 'row_count' with the number of rows extracted.
    """
    mapping = json.loads(column_mapping) if isinstance(column_mapping, str) else column_mapping
    lines = raw_content.strip().split("\n")
    rows = []

    for line in lines:
        if line.startswith("[") or not line.strip():
            continue
        cells = line.split("\t") if "\t" in line else line.split(",")
        row = {}
        for i, cell in enumerate(cells):
            col_key = f"col_{i}"
            if col_key in mapping:
                row[mapping[col_key]] = cell.strip()
        if row:
            rows.append(row)

    return {"rows": rows, "row_count": len(rows)}


def extract_text_content(raw_content: str, title: str = "", topic: str = "") -> dict:
    """Extract narrative text content for the knowledge base.

    Args:
        raw_content: Raw text content from a PDF page or document section.
        title: A short title for this content piece.
        topic: Category — one of 'concept', 'methodology', 'metric', 'strategy'.

    Returns:
        dict with 'title', 'content', and 'topic'.
    """
    content = raw_content
    for prefix in ["[PDF Page", "[Sheet:", "[CSV,"]:
        if content.startswith(prefix):
            first_newline = content.find("\n")
            if first_newline > 0:
                content = content[first_newline + 1:]
            break

    return {"title": title, "content": content.strip(), "topic": topic}


def normalize_dates(date_str: str) -> str:
    """Normalize a date string to YYYY-MM-DD format.

    Args:
        date_str: Date string in any common format (MM/DD/YYYY, DD-MM-YYYY, etc.)

    Returns:
        Date string in YYYY-MM-DD format, or original string if parsing fails.
    """
    formats = [
        "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d",
        "%m-%d-%Y", "%d-%m-%Y", "%Y%m%d",
        "%b %d, %Y", "%d %b %Y", "%B %d, %Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str.strip()


def clean_numeric_values(value: str) -> str:
    """Clean a numeric string: remove commas, handle #N/A, convert to float-safe string.

    Args:
        value: Raw numeric string that may contain commas, currency symbols, or error markers.

    Returns:
        Cleaned numeric string, or empty string if not a valid number.
    """
    if not value or not isinstance(value, str):
        return ""
    v = value.strip()
    invalid = {"#N/A", "#N/A N/A", "N/A", "#VALUE!", "#REF!", "#DIV/0!", "-", "--", "None", "nan"}
    if v in invalid:
        return ""
    v = v.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        float(v)
        return v
    except ValueError:
        return ""


def _coerce_records(records_json):
    """Parse LLM-provided records into a flat list.

    Tolerates three malformed shapes we see in practice:
      - markdown fences around the JSON (```json ... ```)
      - several JSON arrays concatenated, e.g. `[...][...]`
      - JSONL, one object per line
    """
    if not isinstance(records_json, str):
        return records_json

    s = records_json.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl >= 0:
            s = s[nl + 1:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
        s = s.strip()

    if not s:
        return []

    try:
        parsed = json.loads(s)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError as e:
        if "Extra data" not in str(e):
            raise

    decoder = json.JSONDecoder()
    records: list = []
    idx, n = 0, len(s)
    while idx < n:
        while idx < n and s[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = decoder.raw_decode(s, idx)
        if isinstance(obj, list):
            records.extend(obj)
        else:
            records.append(obj)
        idx = end
    log.warning("coerced %d records from multi-value JSON payload", len(records))
    return records


async def store_prices(records_json: str) -> dict:
    """Store price records to the prices_daily PostgreSQL table using upsert.

    Args:
        records_json: JSON string containing a list of price records.
            Each record must have: symbol, date, open, high, low, close, volume.

    Returns:
        dict with 'records_stored' count and 'status'.
    """
    records = _coerce_records(records_json)
    cleaned = []
    for r in records:
        try:
            cleaned.append({
                "symbol": str(r["symbol"]).strip(),
                "date": r["date"],
                "open": float(r["open"]) if r.get("open") else None,
                "high": float(r["high"]) if r.get("high") else None,
                "low": float(r["low"]) if r.get("low") else None,
                "close": float(r["close"]) if r.get("close") else None,
                "volume": int(float(r["volume"])) if r.get("volume") else None,
            })
        except (ValueError, KeyError):
            continue

    if not cleaned:
        return {"records_stored": 0, "status": "empty"}

    async with async_session() as db:
        await bulk_upsert_prices(db, cleaned)
    log.info("stored prices count=%d", len(cleaned))
    return {"records_stored": len(cleaned), "status": "ok"}


async def store_esg_scores(records_json: str) -> dict:
    """Store ESG score records to the esg_scores PostgreSQL table using upsert.

    Args:
        records_json: JSON string containing a list of ESG records.
            Each record must have: symbol, date, provider, e_score, s_score, g_score, composite.

    Returns:
        dict with 'records_stored' count and 'status'.
    """
    records = _coerce_records(records_json)
    cleaned = []
    for r in records:
        try:
            raw_date = r.get("date")
            if isinstance(raw_date, str):
                parsed_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
            elif isinstance(raw_date, date):
                parsed_date = raw_date
            else:
                continue  # skip invalid date

            composite_raw = r.get("composite", r.get("composite_score"))

            cleaned.append({
                "symbol": str(r["symbol"]).strip(),
                "date": parsed_date,  
                "provider": str(r.get("provider", "unknown")).strip(),
                "e_score": float(r["e_score"]) if r.get("e_score") is not None else None,
                "s_score": float(r["s_score"]) if r.get("s_score") is not None else None,
                "g_score": float(r["g_score"]) if r.get("g_score") is not None else None,
                "composite_score": float(composite_raw) if composite_raw is not None else None,
            })
        except (ValueError, KeyError):
            continue

    if not cleaned:
        return {"records_stored": 0, "status": "empty"}

    async with async_session() as db:
        await bulk_upsert_esg_scores(db, cleaned)
    log.info("stored esg_scores count=%d", len(cleaned))
    return {"records_stored": len(cleaned), "status": "ok"}


async def store_company_metadata(records_json: str) -> dict:
    """Store company metadata to the companies PostgreSQL table using upsert.

    Args:
        records_json: JSON string containing a list of company records.
            Each record must have: symbol, name. Optional: sector, sub_industry,
            restricted_business, severe_controversy.

    Returns:
        dict with 'records_stored' count and 'status'.
    """
    records = _coerce_records(records_json)
    cleaned = []
    for r in records:
        if not r.get("symbol") or not r.get("name"):
            continue
        cleaned.append({
            "symbol": str(r["symbol"]).strip(),
            "name": str(r["name"]).strip(),
            "sector": r.get("sector"),
            "sub_industry": r.get("sub_industry"),
            "restricted_business": bool(r.get("restricted_business", False)),
            "severe_controversy": bool(r.get("severe_controversy", False)),
        })

    if not cleaned:
        return {"records_stored": 0, "status": "empty"}

    async with async_session() as db:
        await bulk_upsert_companies(db, cleaned)
    log.info("stored companies count=%d", len(cleaned))
    return {"records_stored": len(cleaned), "status": "ok"}


async def store_knowledge_embedding(title: str, content: str, topic: str) -> dict:
    """Store research/narrative content to the knowledge_base table with an
    embedding generated from text-embedding-004.

    Args:
        title: Short title for this knowledge entry.
        content: The text content to store.
        topic: Category — one of 'concept', 'methodology', 'metric', 'strategy'.

    Returns:
        dict with 'status' and 'entry_id'.
    """
    text_for_embedding = f"{title}: {content}" if title else content
    embedding = await generate_embedding(text_for_embedding)

    async with async_session() as db:
        entry = await create_knowledge_entry(db, {
            "title": title,
            "content": content,
            "topic": topic,
            "embedding": embedding,
        })
        entry_id = entry.id

    log.info("stored knowledge entry_id=%s topic=%s", entry_id, topic)
    return {"status": "ok", "entry_id": entry_id}


def trigger_metric_recomputation() -> dict:
    """Trigger recomputation of all financial and ESG metrics.

    This runs the metric pipeline which computes Sharpe, Sortino, Calmar,
    volatility, drawdown, momentum for all companies, then aggregates
    sector rankings.

    Returns:
        dict with 'status'.
    """
    from tasks.pipeline_task import recompute_metrics
    recompute_metrics.delay()
    return {"status": "triggered"}
