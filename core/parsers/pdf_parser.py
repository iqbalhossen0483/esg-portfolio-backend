import pdfplumber


def parse_pdf(file_path: str) -> list[str]:
    """Parse a PDF file. Each PDF page becomes one string.
    Preserves table structures where possible.
    Returns list of strings — one per page.
    """
    pages = []

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text_parts = []

            # Try to extract tables first
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    for row in table:
                        row_str = "\t".join(
                            str(cell) if cell else "" for cell in row
                        )
                        text_parts.append(row_str)

            # Extract remaining text
            text = page.extract_text()
            if text:
                text_parts.append(text)

            page_text = "\n".join(text_parts).strip()
            if page_text:
                pages.append(f"[PDF Page {i + 1}]\n{page_text}")

    return pages
