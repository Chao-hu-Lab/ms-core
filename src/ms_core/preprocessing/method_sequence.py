"""Method sequence parsing helpers for Step1 preprocessing.

This module owns method-file injection parsing so DataOrganizer can orchestrate
Step1 without carrying DOCX table parsing and injection row extraction details.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any, Sequence
import zipfile
import xml.etree.ElementTree as ET

from ms_core.preprocessing.sample_identity import (
    extract_primary_sample_token,
    is_likely_sample_name,
    normalize_sample_key,
    simplify_method_sample_name,
)

logger = logging.getLogger(__name__)


@dataclass
class InjectionInfo:
    """Information about a sample injection from method file."""

    injection_order: int
    file_name: str
    sample_name: str
    injection_volume: float
    instrument_method: str = ""


def extract_docx_tables_fallback(file_path: str | Path) -> list[list[list[str]]]:
    """Extract DOCX table cell texts without python-docx."""
    tables: list[list[list[str]]] = []
    path = Path(file_path)

    if path.suffix.lower() != ".docx" or not path.exists():
        return tables

    try:
        with zipfile.ZipFile(path, "r") as zf:
            with zf.open("word/document.xml") as fp:
                tree = ET.parse(fp)
        root = tree.getroot()
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}

        for tbl in root.findall(".//w:tbl", ns):
            parsed_rows: list[list[str]] = []
            for tr in tbl.findall("./w:tr", ns):
                parsed_cells: list[str] = []
                for tc in tr.findall("./w:tc", ns):
                    texts = [t.text for t in tc.findall(".//w:t", ns) if t.text]
                    parsed_cells.append("".join(texts).strip())
                if parsed_cells:
                    parsed_rows.append(parsed_cells)
            if parsed_rows:
                tables.append(parsed_rows)
    except (OSError, KeyError, zipfile.BadZipFile, ET.ParseError) as exc:
        logger.warning("Fallback DOCX parse failed for %s: %s", path, exc)

    return tables


def parse_injection_volume_from_cells(cells: Sequence[Any]) -> float:
    """Parse likely injection volume from a table row."""
    for cell in reversed(cells):
        cell_text = str(cell).strip().replace(",", "")
        if not cell_text:
            continue
        try:
            value = float(cell_text)
        except ValueError:
            continue
        if 0 < value <= 100:
            return value
    return 0.0


def extract_injection_rows_from_table(
    table_rows: Sequence[Sequence[Any]],
    preserve_source_order: bool = False,
) -> list[InjectionInfo]:
    """Extract injection rows from one method sequence table."""
    injections: list[InjectionInfo] = []
    candidate_pairs = [(0, 1), (3, 4)]

    for row in table_rows:
        cells = [str(cell).strip() for cell in row]
        if len(cells) < 2:
            continue

        for order_idx, sample_idx in candidate_pairs:
            if len(cells) <= sample_idx:
                continue
            order_text = cells[order_idx].strip() if len(cells) > order_idx else ""
            if not re.fullmatch(r"\d{1,4}", order_text):
                continue

            sample_cell = cells[sample_idx]
            sample_text = re.sub(r"\s+", " ", sample_cell).strip()
            if not sample_text:
                continue
            sample_token = extract_primary_sample_token(sample_cell)
            if not sample_token and not is_likely_sample_name(sample_cell):
                continue

            instrument_method = ""
            next_col = sample_idx + 1
            if len(cells) > next_col:
                candidate = re.sub(r"\s+", " ", cells[next_col]).strip()
                if candidate and not candidate.isdigit():
                    instrument_method = candidate

            injections.append(
                InjectionInfo(
                    injection_order=int(order_text),
                    file_name=sample_text,
                    sample_name=simplify_method_sample_name(sample_text),
                    injection_volume=parse_injection_volume_from_cells(cells),
                    instrument_method=instrument_method,
                )
            )

    if not injections:
        return injections

    if preserve_source_order:
        return _dedupe_and_renumber_by_source_order(injections)

    deduped: list[InjectionInfo] = []
    seen = set()
    for info in sorted(injections, key=lambda x: x.injection_order):
        dedupe_key = (info.injection_order, normalize_sample_key(info.file_name))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(info)
    return deduped


def parse_injection_tables(tables: Sequence[Sequence[Sequence[Any]]]) -> list[InjectionInfo]:
    """Parse injection sequence records from already-loaded DOCX tables."""
    target_table = _find_injection_sequence_table(tables)

    if target_table is not None:
        parsed_from_target = extract_injection_rows_from_table(target_table)
        if parsed_from_target:
            order_values = [info.injection_order for info in parsed_from_target]
            duplicate_orders = len(order_values) - len(set(order_values))
            bc_like_count = sum(
                1 for info in parsed_from_target if re.search(r"bc\d+", info.file_name, re.IGNORECASE)
            )
            if duplicate_orders > 0 and bc_like_count >= 5:
                parsed_by_rows = extract_injection_rows_from_table(
                    target_table,
                    preserve_source_order=True,
                )
                if parsed_by_rows:
                    return parsed_by_rows
            return parsed_from_target

    row_order_records = _parse_target_table_by_row_order(target_table)
    if row_order_records:
        return row_order_records

    return _parse_best_fallback_table(tables)


def parse_injection_sequence(file_path: str | Path) -> list[InjectionInfo]:
    """Parse injection sequence records from a Word method file."""
    path = Path(file_path)
    if not path.exists() or path.suffix.lower() not in {".docx", ".doc"}:
        return []

    tables = _load_docx_tables(path)
    return parse_injection_tables(tables)


def _load_docx_tables(file_path: Path) -> list[list[list[str]]]:
    tables: list[list[list[str]]] = []
    try:
        from docx import Document
        from docx.opc.exceptions import PackageNotFoundError
    except ImportError:
        logger.warning(
            "python-docx not installed; using fallback DOCX parser for injection sequence"
        )
        return extract_docx_tables_fallback(file_path)

    try:
        doc = Document(file_path)
        for table in doc.tables:
            table_rows: list[list[str]] = []
            for row in table.rows:
                table_rows.append([cell.text.strip() for cell in row.cells])
            if table_rows:
                tables.append(table_rows)
    except (OSError, RuntimeError, ValueError, PackageNotFoundError) as exc:
        logger.warning(
            "python-docx parse failed for %s (%s); using fallback parser", file_path, exc
        )
        tables = extract_docx_tables_fallback(file_path)

    return tables


def _find_injection_sequence_table(
    tables: Sequence[Sequence[Sequence[Any]]],
) -> Sequence[Sequence[Any]] | None:
    for table_rows in tables:
        if len(table_rows) <= 10:
            continue
        header_cells = [str(cell).strip().lower() for cell in table_rows[0]]
        if any(
            ("file" in h and "name" in h)
            or ("filename" in h)
            or ("檔" in h)
            or ("樣本" in h)
            for h in header_cells
        ):
            return table_rows
    return None


def _parse_target_table_by_row_order(
    target_table: Sequence[Sequence[Any]] | None,
) -> list[InjectionInfo]:
    if target_table is None:
        return []

    injection_list: list[InjectionInfo] = []
    row_order = 0
    for row_idx, row in enumerate(target_table):
        if row_idx == 0:
            continue

        cells = [str(cell).strip() for cell in row]
        if len(cells) < 2:
            continue

        file_name = re.sub(r"\s+", " ", cells[1] if len(cells) > 1 else "").strip()
        if not file_name:
            continue

        row_order += 1
        instrument_method = ""
        for idx in [2, 3]:
            if len(cells) > idx and "method" not in cells[idx].lower():
                if cells[idx] and not cells[idx].isdigit():
                    instrument_method = cells[idx]
                    break

        injection_list.append(
            InjectionInfo(
                injection_order=row_order,
                file_name=file_name,
                sample_name=simplify_method_sample_name(file_name),
                injection_volume=parse_injection_volume_from_cells(cells),
                instrument_method=instrument_method,
            )
        )
    return injection_list


def _parse_best_fallback_table(
    tables: Sequence[Sequence[Sequence[Any]]],
) -> list[InjectionInfo]:
    best_list: list[InjectionInfo] = []
    best_score: tuple[int, int, int, int] = (-1, -1, -1, -1)
    for table_rows in tables:
        parsed = extract_injection_rows_from_table(table_rows)
        if len(parsed) < 5:
            continue

        unique_orders = len({info.injection_order for info in parsed})
        unique_samples = len({normalize_sample_key(info.file_name) for info in parsed})
        duplicate_orders = len(parsed) - unique_orders
        score = (unique_samples, unique_orders, -duplicate_orders, len(parsed))
        if score > best_score:
            best_score = score
            best_list = parsed

    return best_list


def _dedupe_and_renumber_by_source_order(injections: Sequence[InjectionInfo]) -> list[InjectionInfo]:
    deduped_source: list[InjectionInfo] = []
    seen_keys = set()
    for info in injections:
        key = normalize_sample_key(info.file_name)
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        deduped_source.append(info)
    for idx, info in enumerate(deduped_source, start=1):
        info.injection_order = idx
    return deduped_source
