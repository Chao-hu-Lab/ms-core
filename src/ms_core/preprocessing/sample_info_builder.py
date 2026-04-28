"""SampleInfo construction helpers for Step1 preprocessing.

This module owns the contract that maps RawIntensity sample columns to method
file injection records without renaming the RawIntensity join keys.
"""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ms_core.preprocessing.method_sequence import InjectionInfo
from ms_core.preprocessing.sample_identity import (
    build_sample_info_identity,
    extract_primary_sample_token,
    is_likely_sample_name,
    normalize_sample_key,
    simplify_method_sample_name,
)
from ms_core.utils.validators import detect_fixed_columns

DISPLAY_COLUMNS = [
    "Sample_Name",
    "Method_Sample_Name",
    "Sample_Type",
    "Injection_Order",
    "Batch",
    "Injection_Volume",
    "DNA_mg/20uL",
]


def is_non_sample_column(column_name: Any) -> bool:
    """Return True when a column is metadata and should not be treated as a sample."""
    name = str(column_name).strip().lower()
    compact = re.sub(r"[^a-z0-9]+", "", name)
    if compact in {"rowid", "featureid", "feature", "id", "mzmineid", "mzminertmin", "z"}:
        return True
    if name.startswith("unnamed:"):
        return True
    if "mzmine rt" in name:
        return True
    return False


class SampleInfoBuilder:
    """Build SampleInfo rows from matrix columns and method-file injection records."""

    def build(
        self,
        df: pd.DataFrame,
        injection_info_list: Sequence[InjectionInfo],
    ) -> pd.DataFrame:
        """Build a SampleInfo DataFrame using RawIntensity column names as join keys."""
        _, num_fixed = detect_fixed_columns(df)
        sample_cols = [
            col for col in list(df.columns[num_fixed:]) if not is_non_sample_column(col)
        ]
        sample_type_row = df.iloc[0] if not df.empty else pd.Series(dtype=object)

        filtered_injection_list = self._renumber_real_samples(injection_info_list)
        info_by_key = self._build_info_by_key(filtered_injection_list)

        sample_info_data = []
        for col in sample_cols:
            matched_info = self._match_column_to_injection(
                str(col),
                filtered_injection_list,
                info_by_key,
            )
            sample_type = sample_type_row.get(col, "sample")
            identity = build_sample_info_identity(
                col,
                matched_info.file_name if matched_info else None,
            )
            sample_info_data.append(
                {
                    "Sample_Name": identity.sample_name,
                    "Sample_Type": sample_type,
                    "Injection_Order": matched_info.injection_order if matched_info else 999,
                    "Injection_Volume": matched_info.injection_volume if matched_info else 0,
                    "Method_Sample_Name": identity.method_sample_name,
                    "_col_name": col,
                }
            )

        sample_info_df = pd.DataFrame(sample_info_data)
        if not sample_info_df.empty:
            sample_info_df = sample_info_df.sort_values("Injection_Order").reset_index(drop=True)

        return self._ensure_display_columns(sample_info_df)

    def _renumber_real_samples(
        self,
        injection_info_list: Sequence[InjectionInfo],
    ) -> list[InjectionInfo]:
        filtered = [
            info for info in injection_info_list if is_likely_sample_name(info.file_name)
        ]
        sorted_info = sorted(filtered, key=lambda info: info.injection_order)
        return [
            replace(info, injection_order=new_order)
            for new_order, info in enumerate(sorted_info, start=1)
        ]

    def _build_info_by_key(
        self,
        injection_info_list: Sequence[InjectionInfo],
    ) -> dict[str, InjectionInfo]:
        info_by_key: dict[str, InjectionInfo] = {}
        for info in injection_info_list:
            keys = {
                normalize_sample_key(info.file_name),
                normalize_sample_key(info.sample_name),
                normalize_sample_key(simplify_method_sample_name(info.file_name)),
                normalize_sample_key(extract_primary_sample_token(info.file_name) or ""),
            }
            for key in keys:
                if key and key not in info_by_key:
                    info_by_key[key] = info
        return info_by_key

    def _match_column_to_injection(
        self,
        column_name: str,
        injection_info_list: Sequence[InjectionInfo],
        info_by_key: dict[str, InjectionInfo],
    ) -> InjectionInfo | None:
        col_lower = column_name.lower()
        col_keys = {
            normalize_sample_key(column_name),
            normalize_sample_key(extract_primary_sample_token(column_name) or ""),
        }
        for key in {key for key in col_keys if key}:
            if key in info_by_key:
                return info_by_key[key]

        for info in injection_info_list:
            file_lower = info.file_name.lower().replace("\n", " ")
            if _matches_bc_identity(col_lower, file_lower):
                return info
            if _matches_qc_identity(col_lower, file_lower):
                return info

        return None

    def _ensure_display_columns(self, sample_info_df: pd.DataFrame) -> pd.DataFrame:
        for col in DISPLAY_COLUMNS:
            if col not in sample_info_df.columns:
                sample_info_df[col] = np.nan

        extra_cols = [
            col for col in sample_info_df.columns if col not in DISPLAY_COLUMNS and col != "_col_name"
        ]
        ordered_cols = DISPLAY_COLUMNS + extra_cols
        if "_col_name" in sample_info_df.columns:
            ordered_cols.append("_col_name")
        return sample_info_df[ordered_cols]


def _detect_variant(text: str) -> str:
    text = text.lower().replace("\n", " ").replace("*", " ")
    if "dna" in text and "rna" in text and "+" in text:
        return "dna+rna"
    if "dnaandrna" in text.replace(" ", ""):
        return "dna+rna"
    if "_rna" in text or " rna" in text or text.endswith("rna"):
        return "rna"
    return "dna"


def _matches_bc_identity(column_lower: str, file_lower: str) -> bool:
    bc_match_col = re.search(r"(tumor|normal|benign|benignfat)?(bc\d+)", column_lower)
    bc_match_file = re.search(
        r"(tumor|normal|benign)\s*(tissue)?\s*(fat\s*)?(bc\d+)",
        file_lower,
    )
    if not bc_match_col or not bc_match_file:
        return False

    col_prefix = bc_match_col.group(1) or ""
    if "benign" in col_prefix:
        col_prefix = "benign"

    file_prefix = bc_match_file.group(1) or ""
    return (
        bc_match_col.group(2) == bc_match_file.group(4)
        and col_prefix == file_prefix
        and _detect_variant(column_lower) == _detect_variant(file_lower)
    )


def _matches_qc_identity(column_lower: str, file_lower: str) -> bool:
    qc_match_col = re.search(r"(pooled_?)?qc_?(\d+)", column_lower)
    qc_match_file = re.search(r"(pooled_?)?qc_?(\d+)", file_lower)
    return bool(
        qc_match_col
        and qc_match_file
        and qc_match_col.group(2) == qc_match_file.group(2)
    )
