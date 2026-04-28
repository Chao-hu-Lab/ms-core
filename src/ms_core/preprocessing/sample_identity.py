"""Sample identity helpers shared by preprocessing steps.

This module keeps matching keys separate from exported sample names. Matching
keys may normalize dataset-specific spelling variants, while SampleInfo names
must preserve the RawIntensity column names used by downstream joins.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class SampleInfoIdentity:
    """Exported SampleInfo identity columns for one raw data column."""

    sample_name: str
    method_sample_name: str


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

    # Normalize technical prefixes in column exports.
    lower = re.sub(r"^(?:dna|rna)_program\d+_", "", lower)
    lower = re.sub(r"^program\d+_", "", lower)
    lower = re.sub(r"dna_(?:and_)?rna", "dnaandrna", lower)
    lower = re.sub(r"dna_rna", "dnaandrna", lower)

    # Normalize ZBEE000070 -> U00070ZBEE.
    zbee_match = re.fullmatch(r"zbee(\d{6})", lower)
    if zbee_match:
        lower = f"u{int(zbee_match.group(1)):05d}zbee"

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
