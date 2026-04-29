"""Matrix transformation helpers for Step1 data organization."""

from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ms_core.preprocessing.method_sequence import InjectionInfo
from ms_core.preprocessing.data_organizer_layout import normalize_sample_type_value
from ms_core.preprocessing.sample_identity import (
    extract_primary_sample_token,
    extract_raw_sample_name,
    is_likely_sample_name,
    normalize_sample_key,
    simplify_method_sample_name,
)
from ms_core.preprocessing.sample_info_builder import is_non_sample_column
from ms_core.utils.validators import detect_fixed_columns


def validate_statistics_input(df: pd.DataFrame) -> tuple[bool, str]:
    """Validate input data for statistics mode."""
    if df is None or df.empty:
        return False, "Input data is empty"
    if len(df.columns) < 3:
        return False, "Input data must have at least 3 columns"

    _, num_fixed = detect_fixed_columns_for_statistics(df)
    if num_fixed >= 1:
        return True, ""

    first_col = str(df.columns[0]).lower()
    second_col = str(df.columns[1]).lower()
    if ("mz" in first_col or "m/z" in first_col or "mass" in first_col) and (
        "rt" in second_col or "time" in second_col or "retention" in second_col
    ):
        return True, ""

    return False, (
        "Statistics mode expects either normalized fixed columns "
        "(Mz/RT) or raw Mz/RT leading columns"
    )


def detect_fixed_columns_for_statistics(df: pd.DataFrame) -> tuple[list[str], int]:
    """Detect fixed columns for statistics-mode sorting."""
    fixed_cols, num_fixed = detect_fixed_columns(df)
    if num_fixed == 0:
        return fixed_cols, num_fixed

    if num_fixed < len(df.columns):
        next_col = str(df.columns[num_fixed]).lower()
        if "tolerance" in next_col:
            fixed_cols = fixed_cols + [df.columns[num_fixed]]
            num_fixed += 1

    return fixed_cols, num_fixed


def reorder_columns_statistics_mode(
    df: pd.DataFrame,
    injection_info_list: Sequence[InjectionInfo],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reorder sample columns for statistics mode without creating SampleInfo."""
    stats: dict[str, Any] = {
        "columns_reordered": 0,
        "columns_unmatched": 0,
    }
    if df.empty:
        return df, stats

    fixed_cols, num_fixed = detect_fixed_columns_for_statistics(df)
    fixed_positions = list(range(num_fixed))
    sample_positions = [
        idx
        for idx in range(num_fixed, len(df.columns))
        if not is_non_sample_column(str(df.columns[idx]))
    ]
    metadata_positions = [
        idx
        for idx in range(num_fixed, len(df.columns))
        if is_non_sample_column(str(df.columns[idx]))
    ]

    if not sample_positions:
        return df, stats

    filtered_injection_list = [
        info for info in injection_info_list if is_likely_sample_name(info.file_name)
    ]
    if not filtered_injection_list:
        stats["columns_unmatched"] = len(sample_positions)
        return df, stats

    sorted_injection_list = sorted(filtered_injection_list, key=lambda x: x.injection_order)
    available_positions = list(sample_positions)
    ordered_positions: list[int] = []

    for info in sorted_injection_list:
        match_pos = find_matching_sample_column_position(
            df,
            available_positions,
            info.file_name,
        )
        if match_pos is None:
            continue
        ordered_positions.append(match_pos)
        available_positions.remove(match_pos)

    ordered_positions.extend(available_positions)
    stats["columns_reordered"] = len(ordered_positions) - len(available_positions)
    stats["columns_unmatched"] = len(available_positions)

    new_positions = fixed_positions + ordered_positions + metadata_positions
    reordered_df = df.iloc[:, new_positions]
    reordered_df.columns = (
        fixed_cols
        + [df.columns[i] for i in ordered_positions]
        + [df.columns[i] for i in metadata_positions]
    )
    return reordered_df, stats


def find_matching_sample_column_position(
    df: pd.DataFrame,
    candidate_positions: Sequence[int],
    file_name: str,
) -> int | None:
    """Find a matching sample column index for a method-file sample name."""
    file_lower = file_name.lower().replace("\n", " ")
    file_simplified = simplify_method_sample_name(file_name).lower()
    file_keys = {
        normalize_sample_key(file_name),
        normalize_sample_key(file_simplified),
        normalize_sample_key(extract_primary_sample_token(file_name) or ""),
    }
    file_keys = {k for k in file_keys if k}

    for pos in candidate_positions:
        col_raw = str(df.columns[pos])
        col_lower = col_raw.lower()
        col_key = extract_raw_sample_name(col_raw).lower()
        col_keys = {
            normalize_sample_key(col_raw),
            normalize_sample_key(col_key),
            normalize_sample_key(extract_primary_sample_token(col_raw) or ""),
        }
        col_keys = {k for k in col_keys if k}

        if file_keys and col_keys and file_keys.intersection(col_keys):
            return pos

        if _matches_bc_identity(col_key, file_lower):
            return pos

        if _matches_qc_identity(col_key, file_lower):
            return pos

        if file_simplified and (file_simplified in col_key or col_key in file_simplified):
            return pos

        col_token = _sanitize(col_key)
        file_token = _sanitize(file_simplified)
        if (
            col_token
            and file_token
            and (col_token == file_token or col_token in file_token or file_token in col_token)
        ):
            return pos

        if file_simplified and (file_simplified in col_lower or col_lower in file_simplified):
            return pos

    return None


def merge_mz_rt(
    df: pd.DataFrame,
    mz_decimals: int = 4,
    rt_decimals: int = 2,
    include_tolerance_col: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge Mz and RT columns into a single Mz/RT column."""
    stats = {"mz_rt_merged": 0, "invalid_values": 0}

    mz_series = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    rt_series = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    valid_mask = mz_series.notna() & rt_series.notna()

    mz_np = mz_series.to_numpy()
    rt_np = rt_series.to_numpy()
    fmt_mz = f"%.{mz_decimals}f"
    fmt_rt = f"%.{rt_decimals}f"
    mz_str_list = [fmt_mz % v if not np.isnan(v) else "nan" for v in mz_np]
    rt_str_list = [fmt_rt % v if not np.isnan(v) else "nan" for v in rt_np]
    merged_list = [f"{m}/{r}" for m, r in zip(mz_str_list, rt_str_list)]

    orig_mz = df.iloc[:, 0].astype(str).tolist()
    orig_rt = df.iloc[:, 1].astype(str).tolist()
    fallback_list = [f"{m}/{r}" for m, r in zip(orig_mz, orig_rt)]
    valid_arr = valid_mask.to_numpy()
    mz_rt_values = [
        merged_list[i] if valid_arr[i] else fallback_list[i] for i in range(len(valid_arr))
    ]

    stats["mz_rt_merged"] = int(valid_mask.sum())
    stats["invalid_values"] = int(len(df) - valid_mask.sum())

    front_cols = {"Mz/RT": mz_rt_values}
    if include_tolerance_col:
        front_cols["m/z Tolerance( ppm)/RT Tolerance"] = "na"
    leading_df = pd.DataFrame(front_cols)
    trailing_df = df.iloc[:, 2:].reset_index(drop=True)
    return pd.concat([leading_df, trailing_df], axis=1), stats


def simplify_headers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Simplify sample headers by extracting sample names from paths."""
    header_mapping: dict[str, str] = {}
    fixed_cols = ["Mz/RT", "FeatureID", "m/z Tolerance( ppm)/RT Tolerance"]

    new_columns: list[str] = []
    for col in df.columns:
        col_str = str(col)
        if col in fixed_cols:
            new_columns.append(col)
            continue
        new_name = extract_raw_sample_name(col_str)
        header_mapping[col_str] = new_name
        new_columns.append(new_name)

    result_df = df.copy()
    result_df.columns = new_columns
    return result_df, header_mapping


def insert_sample_type_row(
    df: pd.DataFrame,
    sample_mapping: dict[str, str],
    sample_type_patterns: dict[str, list[str]],
    sample_type_overrides: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Insert Sample_Type row at the top of the data."""
    stats: dict[str, Any] = {"types_detected": {}, "types_from_input": 0}
    _, num_fixed = detect_fixed_columns(df)
    sample_types = ["Sample_Type"] + ["na"] * (num_fixed - 1)

    override_exact: dict[str, str] = {}
    override_by_key: dict[str, str] = {}
    if sample_type_overrides:
        for col_name, sample_type in sample_type_overrides.items():
            normalized = normalize_sample_type_value(sample_type)
            if normalized is None:
                continue
            exact_key = str(col_name)
            override_exact[exact_key] = normalized
            normalized_col_key = normalize_sample_key(exact_key)
            if normalized_col_key and normalized_col_key not in override_by_key:
                override_by_key[normalized_col_key] = normalized

    for col in df.columns[num_fixed:]:
        if is_non_sample_column(str(col)):
            sample_types.append("na")
            continue
        sample_type = override_exact.get(str(col))
        if sample_type is None:
            col_key = normalize_sample_key(str(col))
            if col_key:
                sample_type = override_by_key.get(col_key)
        if sample_type is None:
            sample_type = detect_sample_type(str(col), sample_mapping, sample_type_patterns)
        else:
            stats["types_from_input"] += 1
        sample_types.append(sample_type)

        if sample_type not in stats["types_detected"]:
            stats["types_detected"][sample_type] = 0
        stats["types_detected"][sample_type] += 1

    sample_type_row = pd.DataFrame([sample_types], columns=df.columns)
    return pd.concat([sample_type_row, df], ignore_index=True), stats


def detect_sample_type(
    column_name: str,
    sample_mapping: dict[str, str],
    sample_type_patterns: dict[str, list[str]],
) -> str:
    """Detect sample type from column name and method-file mapping."""
    col_lower = column_name.lower()
    for sample_type, patterns in sample_type_patterns.items():
        for pattern in patterns:
            if re.search(pattern, col_lower, re.IGNORECASE):
                return sample_type

    for pattern, sample_type in sample_mapping.items():
        if pattern.lower() in col_lower:
            return sample_type

    return "sample"


def finalize_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Finalize numeric sample columns while preserving the Sample_Type row."""
    _, num_fixed = detect_fixed_columns(df)
    if len(df) > 1 and num_fixed < len(df.columns):
        sample_col_positions = list(range(num_fixed, len(df.columns)))
        data_block = df.iloc[1:, sample_col_positions]
        converted = data_block.apply(pd.to_numeric, errors="coerce")
        conv_values = converted.to_numpy()
        result_df = df.copy()
        for j, col_pos in enumerate(sample_col_positions):
            col_name = result_df.columns[col_pos]
            result_df[col_name] = [result_df.iat[0, col_pos]] + conv_values[:, j].tolist()
        return result_df

    return df


def reorder_columns_by_injection(
    df: pd.DataFrame,
    sample_info_df: pd.DataFrame,
) -> pd.DataFrame:
    """Reorder DataFrame columns based on Injection_Order from SampleInfo."""
    if sample_info_df.empty:
        return df

    fixed_cols, num_fixed = detect_fixed_columns(df)
    trailing_cols = list(df.columns[num_fixed:])
    sample_cols = [col for col in trailing_cols if not is_non_sample_column(str(col))]
    metadata_cols = [col for col in trailing_cols if is_non_sample_column(str(col))]

    if "_col_name" in sample_info_df.columns:
        sorted_info = sample_info_df.sort_values("Injection_Order")
        ordered_sample_cols = sorted_info["_col_name"].tolist()
        ordered_sample_cols = [c for c in ordered_sample_cols if c in sample_cols]
        for col in sample_cols:
            if col not in ordered_sample_cols:
                ordered_sample_cols.append(col)
    else:
        ordered_sample_cols = _order_sample_columns_by_display_name(sample_cols, sample_info_df)

    new_column_order = fixed_cols + ordered_sample_cols + metadata_cols
    return df[new_column_order]


def auto_detect_sample_types(
    column_names: Sequence[str],
    sample_type_patterns: dict[str, list[str]],
    patterns: dict[str, str] | None = None,
) -> dict[str, str]:
    """Auto-detect sample types from column names."""
    return {
        col: detect_sample_type(col, patterns or {}, sample_type_patterns)
        for col in column_names
    }


def _order_sample_columns_by_display_name(
    sample_cols: list[str],
    sample_info_df: pd.DataFrame,
) -> list[str]:
    ordered_sample_cols: list[str] = []
    sorted_info = sample_info_df.sort_values("Injection_Order")

    for _, row in sorted_info.iterrows():
        sample_name = row["Sample_Name"]
        for col in sample_cols:
            col_lower = col.lower()
            sample_lower = sample_name.lower()

            matched = _matches_bc_display_name(col_lower, sample_lower)
            if _matches_qc_identity(col_lower, sample_lower):
                matched = True

            if matched and col not in ordered_sample_cols:
                ordered_sample_cols.append(col)
                break

    for col in sample_cols:
        if col not in ordered_sample_cols:
            ordered_sample_cols.append(col)

    return ordered_sample_cols


def _matches_bc_display_name(col_lower: str, sample_lower: str) -> bool:
    bc_match_col = re.search(r"(tumor|normal|benign|benignfat)?(bc\d+)", col_lower)
    bc_match_sample = re.search(
        r"(tumor|normal|benign)?\s*tissue\s*(fat\s*)?(bc\d+)",
        sample_lower,
    )
    if not bc_match_col or not bc_match_sample:
        return False

    col_type = bc_match_col.group(1) or ""
    if "benign" in col_type:
        col_type = "benign"
    sample_type = bc_match_sample.group(1) or ""
    return bc_match_col.group(2) == bc_match_sample.group(3) and col_type == sample_type


def _matches_bc_identity(column_key: str, file_lower: str) -> bool:
    bc_match_col = re.search(r"(tumor|normal|benign|benignfat)?(bc\d+)", column_key)
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
        and _detect_variant(column_key) == _detect_variant(file_lower)
    )


def _matches_qc_identity(column_lower: str, file_lower: str) -> bool:
    qc_match_col = re.search(r"(pooled_?)?qc_?(\d+)", column_lower)
    qc_match_file = re.search(r"(pooled_?)?qc_?(\d+)", file_lower)
    return bool(
        qc_match_col
        and qc_match_file
        and qc_match_col.group(2) == qc_match_file.group(2)
    )


def _detect_variant(text: str) -> str:
    text = text.lower().replace("\n", " ").replace("*", " ")
    if "dna" in text and "rna" in text and "+" in text:
        return "dna+rna"
    if "dnaandrna" in text.replace(" ", ""):
        return "dna+rna"
    if "_rna" in text or " rna" in text or text.endswith("rna"):
        return "rna"
    return "dna"


def _sanitize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())
