"""Sample identity helpers shared by preprocessing steps.

This module keeps matching keys separate from exported sample names. Matching
keys may normalize dataset-specific spelling variants, while SampleInfo names
must preserve the RawIntensity column names used by downstream joins.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

SAMPLE_TOKEN_REGEX = re.compile(
    r"(?:EC\d{2,4}(?:_\d+)?|U\d{5}ZBEE|ZBEE\d{6}|"
    r"pooled[_\s-]*QC[_\s-]*\d+|QC[_\s-]*sample[_\s-]*\d+|"
    r"QC[_\s-]*\d+|blank)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SampleInfoIdentity:
    """Exported SampleInfo identity columns for one raw data column."""

    sample_name: str
    method_sample_name: str


def extract_primary_sample_token(text: Any) -> str | None:
    """Extract a canonical sample token from free text for matching."""
    if text is None:
        return None
    cleaned = str(text).strip()
    if not cleaned:
        return None
    match = SAMPLE_TOKEN_REGEX.search(cleaned)
    if match:
        token = re.sub(r"\s+", "", match.group(0))
        return token.replace("-", "_")

    compact = re.sub(r"\s+", "", cleaned.lower())
    if re.search(r"bc\d+_dna(?:\+rna|andrna)", compact):
        return cleaned
    if re.search(
        r"\bbc\d+_(?:dna\s*(?:\+\s*|and\s*)rna|dnaandrna|dna\+rna|dna|rna)\b",
        cleaned,
        re.IGNORECASE,
    ):
        return cleaned
    if re.search(
        r"\b(?:dna_)?program\d+_[a-z0-9_]*bc\d+_(?:dnaandrna|dna|rna)\b",
        cleaned,
        re.IGNORECASE,
    ):
        return cleaned
    return None


def extract_raw_sample_name(header: Any) -> str:
    """Extract the raw sample token from a matrix header or file path."""
    header_text = str(header)
    if header_text.lower().startswith("intensity of "):
        header_text = header_text[13:]

    try:
        filename = Path(header_text).stem
    except (OSError, ValueError):
        filename = header_text

    patterns_to_remove = [
        r"^.+_program\d+_",
        r"^program\d+_(?:dna|rna)_program\d+_",
        r"^(?:dna|rna)_program\d+_",
        r"^program\d+_program\d+_",
        r"^program\d+_\d+_",
        r"^program\d+_",
    ]

    result = filename
    for pattern in patterns_to_remove:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)

    result = re.sub(r"(qc)[ _-]?(\d+)", r"\1_\2", result, flags=re.IGNORECASE)
    result = result.strip("_")
    return result if result else filename


def normalize_sample_key(sample_name: Any) -> str:
    """Normalize a matrix or method-file sample name for matching only."""
    if sample_name is None:
        return ""
    token = str(sample_name).strip()
    if not token:
        return ""
    return normalize_sample_token(extract_raw_sample_name(token))


def is_likely_sample_name(text: Any) -> bool:
    """Return True when a method-file entry looks like a real sample."""
    if text is None:
        return False
    name_lower = str(text).lower()
    if "blank" in name_lower or "sdolek" in name_lower or "std" in name_lower:
        return False
    if extract_primary_sample_token(text):
        return True
    return (
        re.search(r"bc\d+", name_lower) is not None
        or "qc" in name_lower
        or "tissue" in name_lower
        or "pooled" in name_lower
    )


def simplify_method_sample_name(file_name: Any) -> str:
    """Simplify a method-file sample name while preserving traceability."""
    name = str(file_name).strip()

    if "pooled" in name.lower() and "qc" in name.lower():
        match = re.search(r"pooled_?QC_?\d+", name, re.IGNORECASE)
        if match:
            return match.group().replace(" ", "")

    tissue_patterns = [
        (r"tumor\s*tissue\s*", "Tumor"),
        (r"normal\s*tissue\s*", "Normal"),
        (r"benign\s*tissue\s*(fat\s*)?", "Benign"),
    ]

    for pattern, prefix in tissue_patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            bc_match = re.search(r"BC\d+_\w+", name, re.IGNORECASE)
            if bc_match:
                return f"{prefix}{bc_match.group()}"

    if "blank" in name.lower():
        match = re.search(r"blank_?\d*", name, re.IGNORECASE)
        if match:
            return match.group()

    if "sdolek" in name.lower() or "std" in name.lower():
        return re.sub(r"\s+", "_", name)

    return name


def normalize_sample_token(sample_name: Any) -> str:
    """Normalize a sample token for method-file matching only."""
    if sample_name is None:
        return ""

    token = str(sample_name).strip()
    if not token:
        return ""

    lower = token.lower()
    lower = re.sub(r"[^a-z0-9]+", "_", lower).strip("_")
    lower = re.sub(r"_+", "_", lower)

    # Strip 14-digit datetime stamps appended by instrument software (YYYYMMDDHHMMSS).
    lower = re.sub(r"_\d{14}$", "", lower)

    # Normalize QC naming variants.
    lower = re.sub(r"qc[_\s-]*sample[_\s-]*(\d+)", r"qc_sample_\1", lower)
    lower = re.sub(r"^qc[_\s-]*(\d+)$", r"qc_sample_\1", lower)
    lower = re.sub(r"pooled[_\s-]*qc[_\s-]*(\d+)", r"pooled_qc_\1", lower)

    # Normalize technical prefixes in column exports. Some providers stack
    # program prefixes, e.g. program2_DNA_program1_TumorBC2257_DNA.
    while True:
        stripped = re.sub(r"^(?:dna|rna)_program\d+_", "", lower)
        stripped = re.sub(r"^program\d+_", "", stripped)
        if stripped == lower:
            break
        lower = stripped
    lower = re.sub(r"dna_(?:and_)?rna", "dnaandrna", lower)
    lower = re.sub(r"dna_rna", "dnaandrna", lower)

    # Normalize ZBEE000070 -> U00070ZBEE. Rerun suffixes are not identity.
    zbee_match = re.fullmatch(r"zbee(\d{6})(?:_\d+)?", lower)
    if zbee_match:
        lower = f"u{int(zbee_match.group(1)):05d}zbee"
    u_zbee_match = re.fullmatch(r"(u\d{5}zbee)(?:_\d+)?", lower)
    if u_zbee_match:
        lower = u_zbee_match.group(1)

    # Rerun suffixes are method-file matching noise, not RawIntensity names.
    lower = re.sub(r"(bc\d+)_\d+(_(?:dna|rna|dnaandrna))?$", r"\1\2", lower)

    # Normalize EC301 -> EC0301 and EC013_2 -> EC013.
    ec_suffix_match = re.fullmatch(r"(ec\d{2,4})_\d+", lower)
    if ec_suffix_match:
        lower = ec_suffix_match.group(1)
    ec_match = re.fullmatch(r"ec(\d{2,4})", lower)
    if ec_match:
        digits = ec_match.group(1)
        if len(digits) == 3 and int(digits) >= 300:
            lower = f"ec0{digits}"
        elif len(digits) == 2:
            lower = f"ec0{digits}"
        else:
            lower = f"ec{digits}"

    return lower


def build_sample_info_identity(
    raw_column_name: Any,
    method_sample_name: Any | None,
) -> SampleInfoIdentity:
    """Build exported identity values without overwriting the raw column name."""
    sample_name = "" if raw_column_name is None else str(raw_column_name)
    method_name = "" if method_sample_name is None else str(method_sample_name).strip()

    return SampleInfoIdentity(
        sample_name=sample_name,
        method_sample_name=method_name,
    )
