"""ADK Training Pipeline Agents.
File parsing + chunking is done in pure Python BEFORE passing to these agents.
Each chunk (500-600 tokens) is processed by this pipeline one at a time.
"""

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import FunctionTool

from core.tools.training_tools import (
    extract_tabular_data,
    extract_text_content,
    normalize_dates,
    clean_numeric_values,
    store_prices,
    store_esg_scores,
    store_company_metadata,
    store_knowledge_embedding,
    trigger_metric_recomputation,
)

# ── Step 1: Classify this chunk ──
classifier_agent = LlmAgent(
    name="DataClassifier",
    model="gemini-2.5-flash",
    description="Classifies a single data chunk into its data type.",
    instruction="""You receive a single chunk of data (500-600 tokens).
    Classify it as ONE of:
    - "price_data": contains OHLCV columns, dates, ticker symbols, numeric prices
    - "esg_scores": contains ESG/Environmental/Social/Governance scores, ratings
    - "company_meta": contains company names, ticker symbols, sectors, industry classifications
    - "research_text": contains narrative text, methodology, analysis, definitions
    - "unknown": does not match any known format

    Also identify:
    - provider: "bloomberg", "lesg", or "unknown"
    - header_row: which row contains column headers (if tabular)
    - date_format: detected date format pattern
    - column_mapping: map detected columns to standard field names

    Output JSON: {type, provider, header_row, date_format, column_mapping, raw_content}""",
    output_key="classified_chunk",
)

# ── Step 2: Extract structured data from this chunk ──
extractor_agent = LlmAgent(
    name="DataExtractor",
    model="gemini-2.5-flash",
    description="Extracts structured records from a classified chunk.",
    instruction="""Read the classified chunk from state key 'classified_chunk'.
    Based on the classification type, extract structured records:

    For "price_data": extract rows as [{symbol, date, open, high, low, close, volume}]
    For "esg_scores": extract as [{symbol, date, provider, e_score, s_score, g_score, composite}]
    For "company_meta": extract as [{symbol, name, sector, sub_industry, restricted_business, severe_controversy}]
    For "research_text": extract as [{title, content, topic}]

    Handle Bloomberg-style Excel where:
    - Row 4 might contain ticker symbols spread across columns
    - Row 6+ contains dates in column A and prices in subsequent columns

    Use the column_mapping from classification to align fields correctly.
    Normalize all dates to YYYY-MM-DD format using normalize_dates tool.
    Clean numeric values using clean_numeric_values tool.

    Output JSON: {type, records: [...], record_count}""",
    tools=[
        FunctionTool(extract_tabular_data),
        FunctionTool(extract_text_content),
        FunctionTool(normalize_dates),
        FunctionTool(clean_numeric_values),
    ],
    output_key="extracted_data",
)

# ── Step 3: Validate extracted data ──
validation_agent = LlmAgent(
    name="DataValidator",
    model="gemini-2.5-pro",
    description="Validates extracted records for accuracy and consistency.",
    instruction="""Read extracted data from state key 'extracted_data'.

    VALIDATE based on data type:
    1. PRICES: Open <= High, Low <= Close, all > 0, dates not in future, volume >= 0
    2. ESG SCORES: within valid range (0-10 for Bloomberg, 0-100 for LESG)
    3. SYMBOLS: should be valid ticker format (1-5 uppercase letters)
    4. DATES: reasonable range (2000-present), proper YYYY-MM-DD format
    5. COMPLETENESS: flag records missing required fields

    Output JSON:
    {
      type,
      valid_records: [...],
      rejected_records: [...],
      warnings: [...],
      stats: {total, valid, rejected}
    }

    Reject records with critical errors. Pass records with minor warnings.""",
    output_key="validated_data",
)

# ── Step 4: Store to correct target ──
storage_agent = LlmAgent(
    name="DataStorer",
    model="gemini-2.5-flash",
    description="Stores validated data to PostgreSQL or pgvector based on data type.",
    instruction="""Read validated data from state key 'validated_data'.

    Route to correct storage based on type:
    - "price_data"    -> use store_prices tool (pass valid_records as JSON string)
    - "esg_scores"    -> use store_esg_scores tool (pass valid_records as JSON string)
    - "company_meta"  -> use store_company_metadata tool (pass valid_records as JSON string)
    - "research_text" -> use store_knowledge_embedding tool for each record

    Use upsert — re-uploading the same data is safe.
    Output JSON: {type, records_stored, records_failed, storage_target}""",
    tools=[
        FunctionTool(store_prices),
        FunctionTool(store_esg_scores),
        FunctionTool(store_company_metadata),
        FunctionTool(store_knowledge_embedding),
    ],
    output_key="storage_result",
)

# ── Per-chunk pipeline ──
chunk_pipeline = SequentialAgent(
    name="ChunkProcessor",
    description="Processes a single chunk: classify → extract → validate → store",
    sub_agents=[classifier_agent, extractor_agent, validation_agent, storage_agent],
)
