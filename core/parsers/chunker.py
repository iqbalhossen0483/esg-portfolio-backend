import tiktoken

encoder = tiktoken.get_encoding("cl100k_base")

CHUNK_MIN_TOKENS = 500
CHUNK_MAX_TOKENS = 600


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken (cl100k_base encoding)."""
    return len(encoder.encode(text))


def chunk_pages(pages: list[str]) -> list[str]:
    """Merge small pages and split large pages into 500-600 token chunks.

    Rules:
    1. Page 500-600 tokens → keep as-is (ideal chunk)
    2. Page < 500 tokens → concatenate with next page(s) until 500-600
    3. Page > 600 tokens → split at sentence boundaries

    Context is preserved — small chunks merge with neighbors.
    """
    chunks = []
    buffer = ""
    buffer_tokens = 0

    for page in pages:
        page_tokens = count_tokens(page)

        # If buffer + page fits in range → add to buffer
        if buffer_tokens + page_tokens <= CHUNK_MAX_TOKENS:
            buffer += ("\n\n" + page if buffer else page)
            buffer_tokens += page_tokens
            continue

        # If buffer is already in range → flush it
        if buffer_tokens >= CHUNK_MIN_TOKENS:
            chunks.append(buffer)
            buffer = ""
            buffer_tokens = 0

        # If page alone is too large → split it
        if page_tokens > CHUNK_MAX_TOKENS:
            if buffer:
                page = buffer + "\n\n" + page
                buffer = ""
                buffer_tokens = 0
            sub_chunks = split_large_text(page, CHUNK_MIN_TOKENS, CHUNK_MAX_TOKENS)
            chunks.extend(sub_chunks[:-1])
            buffer = sub_chunks[-1]
            buffer_tokens = count_tokens(buffer)
        else:
            buffer = (buffer + "\n\n" + page if buffer else page)
            buffer_tokens = count_tokens(buffer)

    # Flush remaining buffer
    if buffer:
        chunks.append(buffer)

    return chunks


def split_large_text(text: str, min_tokens: int, max_tokens: int) -> list[str]:
    """Split text larger than max_tokens at sentence boundaries."""
    sentences = text.replace("\n", " \n ").split(". ")
    chunks = []
    current = ""
    current_tokens = 0

    for sentence in sentences:
        s = sentence.strip() + ". "
        s_tokens = count_tokens(s)

        if current_tokens + s_tokens > max_tokens and current_tokens >= min_tokens:
            chunks.append(current.strip())
            current = s
            current_tokens = s_tokens
        else:
            current += s
            current_tokens += s_tokens

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text]
