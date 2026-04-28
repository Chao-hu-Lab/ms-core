"""Raw matrix layout helpers for Step1 data organization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from ms_core.preprocessing.sample_info_builder import is_non_sample_column


@dataclass(frozen=True)
class SampleTypeRowExtraction:
    """Result of extracting an input-provided Sample Type row."""

    data: pd.DataFrame
    sample_types: dict[str, str]
    stats: dict[str, Any]


def is_pre_merged_mz_rt_header(col_name: str) -> bool:
    """Return True when a column header indicates an already-combined Mz/RT column."""
    normalized = re.sub(r"[\s_]", "", col_name.lower())
    return normalized in {"mz/rt", "m/z/rt", "mzrt"}


def expand_pre_merged_mz_rt(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Expand a leading combined Mz/RT column into separate numeric Mz and RT columns."""
    if df.empty:
        return df, False

    first_col = str(df.columns[0]).strip().lower()
    if not is_pre_merged_mz_rt_header(first_col):
        return df, False

    mz_values: list[float] = []
    rt_values: list[float] = []
    any_parsed = False

    for val in df.iloc[:, 0]:
        val_str = str(val).strip()
        parts = val_str.split("/")
        if len(parts) == 2:
            try:
                mz_values.append(float(parts[0].strip()))
                rt_values.append(float(parts[1].strip()))
                any_parsed = True
                continue
            except ValueError:
                pass
        mz_values.append(np.nan)
        rt_values.append(np.nan)

    if not any_parsed:
        return df, False

    rest_df = df.iloc[:, 1:].reset_index(drop=True)
    expanded = pd.DataFrame({"Mz": mz_values, "RT": rt_values})
    return pd.concat([expanded, rest_df], axis=1), True


def normalize_sample_type_value(value: Any) -> Optional[str]:
    """Normalize sample type labels to toolkit canonical values."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None

    key = re.sub(r"[^a-z0-9]+", "", text.lower())
    mapping = {
        "qc": "QC",
        "qualitycontrol": "QC",
        "pooledqc": "QC",
        "exposure": "Exposure",
        "tumor": "Exposure",
        "tumour": "Exposure",
        "cancer": "Exposure",
        "case": "Exposure",
        "normal": "Normal",
        "control": "Control",
        "benign": "Control",
        "benignfat": "Control",
        "sample": "sample",
        "blank": "blank",
        "std": "standard",
        "standard": "standard",
        "sdolek": "standard",
        "na": "na",
    }
    return mapping.get(key, text)


def extract_sample_type_row_from_input(df: pd.DataFrame) -> SampleTypeRowExtraction:
    """Extract a user-provided Sample Type row from raw input when present."""
    stats: dict[str, Any] = {
        "sample_types_from_input": False,
        "input_sample_type_count": 0,
    }
    if df.empty:
        return SampleTypeRowExtraction(df, {}, stats)

    marker = str(df.iloc[0, 0]).strip().lower()
    marker_compact = re.sub(r"[^a-z0-9]+", "", marker)
    if marker_compact != "sampletype":
        return SampleTypeRowExtraction(df, {}, stats)

    provided_types: dict[str, str] = {}
    for col in df.columns[2:]:
        col_str = str(col)
        if is_non_sample_column(col_str):
            continue
        normalized = normalize_sample_type_value(df.iloc[0][col])
        if normalized is None:
            continue
        provided_types[col_str] = normalized

    cleaned_df = df.iloc[1:].reset_index(drop=True)
    stats["sample_types_from_input"] = True
    stats["input_sample_type_count"] = len(provided_types)
    return SampleTypeRowExtraction(cleaned_df, provided_types, stats)


def move_leading_metadata_to_end(df: pd.DataFrame) -> pd.DataFrame:
    """Move leading non-Mz/RT metadata columns to the end of the matrix."""
    if df.empty or len(df.columns) < 3:
        return df

    leading_meta: list[Any] = []
    for col in df.columns:
        col_lower = str(col).strip().lower()
        if _looks_like_mz_or_rt(col_lower):
            break
        if is_non_sample_column(str(col)):
            leading_meta.append(col)
            continue
        break
    if not leading_meta:
        return df
    rest = [c for c in df.columns if c not in leading_meta]
    return df[rest + leading_meta]


def _looks_like_mz_or_rt(col_lower: str) -> bool:
    return bool(
        "m/z" in col_lower
        or re.search(r"\bmz\b", col_lower)
        or re.search(r"\bmass\b", col_lower)
        or re.search(r"\brt\b", col_lower)
        or "retention" in col_lower
    )
