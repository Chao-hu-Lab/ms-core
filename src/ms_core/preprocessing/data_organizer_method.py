"""Method-file sample type mapping helpers for Step1 data organization."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from ms_core.preprocessing.method_sequence import extract_docx_tables_fallback

logger = logging.getLogger(__name__)


def parse_method_file(file_path: str | Path) -> dict[str, str]:
    """Parse method file tables into sample-id to sample-type mapping."""
    mapping: dict[str, str] = {}
    path = Path(file_path)

    if not path.exists() or path.suffix.lower() not in [".docx", ".doc"]:
        return mapping

    tables = _load_method_tables(path)
    for table_rows in tables:
        for row in table_rows:
            cells = [str(cell).strip() for cell in row]
            for cell_text in cells:
                sample_type = _sample_type_from_text(cell_text)
                if sample_type is None:
                    continue
                sample_id = extract_sample_id(cell_text)
                if sample_id:
                    mapping[sample_id] = sample_type

    return mapping


def extract_sample_id(text: str) -> Optional[str]:
    """Extract a sample ID from free text."""
    patterns = [
        r"BC\d+_\w+",
        r"[A-Z]{2,}\d+",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group()
    return None


def _load_method_tables(path: Path) -> list[list[list[str]]]:
    try:
        from docx import Document

        doc = Document(path)
        tables: list[list[list[str]]] = []
        for table in doc.tables:
            table_rows: list[list[str]] = []
            for row in table.rows:
                table_rows.append([cell.text.strip() for cell in row.cells])
            if table_rows:
                tables.append(table_rows)
        return tables
    except ImportError:
        logger.warning("python-docx not installed; using fallback DOCX parser for method file")
        return extract_docx_tables_fallback(path)
    except Exception as exc:
        logger.warning(
            "python-docx parse failed for %s (%s); using fallback parser",
            path,
            exc,
        )
        return extract_docx_tables_fallback(path)


def _sample_type_from_text(text: str) -> str | None:
    cell_lower = text.lower()
    if "tumor" in cell_lower or "cancer" in cell_lower:
        return "tumor"
    if "normal" in cell_lower:
        return "normal"
    if "benign" in cell_lower:
        return "benign"
    if "qc" in cell_lower or "pool" in cell_lower:
        return "qc"
    if "blank" in cell_lower:
        return "blank"
    if "std" in cell_lower:
        return "standard"
    return None
