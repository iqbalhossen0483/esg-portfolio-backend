from pathlib import Path


def parse_file(file_path: str) -> list[str]:
    """Detect file format by extension and dispatch to the correct parser.
    Returns a list of strings — one per page/sheet.
    """
    ext = Path(file_path).suffix.lower()

    if ext in (".xlsx", ".xls"):
        from .excel_parser import parse_excel
        return parse_excel(file_path)
    elif ext == ".csv":
        from .csv_parser import parse_csv
        return parse_csv(file_path)
    elif ext == ".pdf":
        from .pdf_parser import parse_pdf
        return parse_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported: .xlsx, .xls, .csv, .pdf")
