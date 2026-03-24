import csv


ROWS_PER_PAGE = 100


def parse_csv(file_path: str) -> list[str]:
    """Parse a CSV file. Split into pages of ~100 rows.
    Header row is included in every page for context.
    Returns list of strings — one per page.
    """
    pages = []

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        all_rows = []
        for row in reader:
            row_str = "\t".join(row)
            if row_str.strip():
                all_rows.append(row_str)

    if not all_rows:
        return []

    header = all_rows[0]
    data_rows = all_rows[1:]

    if not data_rows:
        pages.append(header)
        return pages

    for i in range(0, len(data_rows), ROWS_PER_PAGE):
        batch = data_rows[i : i + ROWS_PER_PAGE]
        page_text = (
            f"[CSV, Rows {i + 2}-{i + 1 + len(batch)}]\n"
            + header
            + "\n"
            + "\n".join(batch)
        )
        pages.append(page_text)

    return pages
