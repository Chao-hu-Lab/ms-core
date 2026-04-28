"""Shared Step1 facade preparation and output assembly helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union

import pandas as pd

from ms_core.preprocessing.data_organizer_layout import (
    expand_pre_merged_mz_rt,
    extract_sample_type_row_from_input,
    normalize_sample_type_value,
)
from ms_core.preprocessing.method_sequence import InjectionInfo


ParseMethodFile = Callable[[Union[str, Path]], dict[str, str]]
ParseInjectionSequence = Callable[[Union[str, Path]], list[InjectionInfo]]
ExtractSampleName = Callable[[str], str]
BuildSampleInfo = Callable[[pd.DataFrame, list[InjectionInfo]], pd.DataFrame]
ReorderColumnsByInjection = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]
FinalizeStructure = Callable[[pd.DataFrame], pd.DataFrame]
ProgressCallback = Callable[[int, str], None]
CancellationChecker = Callable[[], bool]


@dataclass(frozen=True)
class StatisticsMzRtRestore:
    """Original Mz/RT columns captured for statistics-mode output restoration."""

    mz_column_name: str
    rt_column_name: str
    mz_values: list[Any]
    rt_values: list[Any]


@dataclass(frozen=True)
class PreparedStep1Input:
    """Common state prepared before Step1 merge/header/type assembly."""

    data: pd.DataFrame
    statistics: dict[str, Any]
    sample_mapping: dict[str, str]
    injection_info_list: list[InjectionInfo]
    input_sample_types: dict[str, str]
    statistics_restore: Optional[StatisticsMzRtRestore]


@dataclass(frozen=True)
class Step1OutputAssembly:
    """Final RawIntensity/SampleInfo output assembled by shared Step1 helpers."""

    data: pd.DataFrame
    sample_info: pd.DataFrame
    statistics: dict[str, Any]
    cancelled: bool = False


def prepare_step1_input(
    df: pd.DataFrame,
    *,
    method_file: Optional[Union[str, Path]],
    sample_type_mapping: Optional[dict[str, str]],
    mode: str,
    parse_method_file: ParseMethodFile,
    parse_injection_sequence: ParseInjectionSequence,
) -> PreparedStep1Input:
    """Prepare common normal/statistics Step1 state after validation."""
    mode_name = str(mode or "normalization").strip().lower()
    result_df = df.copy()
    stats: dict[str, Any] = {
        "original_rows": len(df),
        "original_cols": len(df.columns),
    }
    if mode_name == "statistics":
        stats["mode"] = "statistics"

    extraction = extract_sample_type_row_from_input(result_df)
    result_df = extraction.data
    stats.update(extraction.stats)

    result_df, _was_expanded = expand_pre_merged_mz_rt(result_df)
    restore = _capture_statistics_restore(result_df) if mode_name == "statistics" else None
    sample_mapping, injection_info_list = _load_method_state(
        method_file,
        sample_type_mapping,
        parse_method_file,
        parse_injection_sequence,
    )

    if mode_name == "statistics" or method_file:
        stats["method_file_samples"] = len(sample_mapping)
        stats["injection_sequence_count"] = len(injection_info_list)

    return PreparedStep1Input(
        data=result_df,
        statistics=stats,
        sample_mapping=sample_mapping,
        injection_info_list=injection_info_list,
        input_sample_types=extraction.sample_types,
        statistics_restore=restore,
    )


def simplify_input_sample_type_overrides(
    input_sample_types: dict[str, str],
    header_mapping: dict[str, str],
    extract_sample_name: ExtractSampleName,
) -> dict[str, str]:
    """Map input Sample Type row values onto simplified output sample names."""
    simplified_types: dict[str, str] = {}
    for raw_col, sample_type in input_sample_types.items():
        simplified = header_mapping.get(raw_col, extract_sample_name(str(raw_col)))
        normalized = normalize_sample_type_value(sample_type)
        if simplified and normalized and simplified not in simplified_types:
            simplified_types[simplified] = normalized
    return simplified_types


def assemble_step1_output(
    df: pd.DataFrame,
    *,
    injection_info_list: list[InjectionInfo],
    build_sample_info: BuildSampleInfo,
    reorder_columns_by_injection: ReorderColumnsByInjection,
    finalize_structure: FinalizeStructure,
    statistics_restore: Optional[StatisticsMzRtRestore] = None,
    update_progress: Optional[ProgressCallback] = None,
    cancellation_requested: Optional[CancellationChecker] = None,
) -> Step1OutputAssembly:
    """Build SampleInfo, reorder columns, finalize numeric data, and restore stats output."""
    if update_progress is not None:
        update_progress(70, "Building SampleInfo...")
    sample_info_df = build_sample_info(df, injection_info_list)
    if _is_cancelled(cancellation_requested):
        return Step1OutputAssembly(
            data=df,
            sample_info=sample_info_df,
            statistics={"sample_info_rows": len(sample_info_df)},
            cancelled=True,
        )

    if update_progress is not None:
        update_progress(80, "Reordering columns by injection order...")
    result_df = reorder_columns_by_injection(df, sample_info_df)

    if "_col_name" in sample_info_df.columns:
        sample_info_df = sample_info_df.drop(columns=["_col_name"])
    if _is_cancelled(cancellation_requested):
        return Step1OutputAssembly(
            data=result_df,
            sample_info=sample_info_df,
            statistics={"sample_info_rows": len(sample_info_df)},
            cancelled=True,
        )

    if update_progress is not None:
        update_progress(90, "Finalizing...")
    result_df = finalize_structure(result_df)
    if statistics_restore is not None:
        if update_progress is not None:
            update_progress(95, "Restoring separate Mz and RT columns...")
        result_df = restore_statistics_mz_rt_columns(result_df, statistics_restore)

    return Step1OutputAssembly(
        data=result_df,
        sample_info=sample_info_df,
        statistics={
            "sample_info_rows": len(sample_info_df),
            "final_rows": len(result_df),
            "final_cols": len(result_df.columns),
        },
    )


def restore_statistics_mz_rt_columns(
    df: pd.DataFrame,
    restore: StatisticsMzRtRestore,
) -> pd.DataFrame:
    """Restore separate Mz and RT columns and remove the internal Mz/RT column."""
    result_df = df.copy()
    mz_column_values = ["Sample_Type"] + restore.mz_values
    rt_column_values = ["na"] + restore.rt_values
    result_df.insert(0, restore.mz_column_name, mz_column_values)
    result_df.insert(1, restore.rt_column_name, rt_column_values)

    mzrt_positions = [idx for idx, col in enumerate(result_df.columns) if col == "Mz/RT"]
    if not mzrt_positions:
        return result_df

    drop_idx = mzrt_positions[0]
    keep_idx = [idx for idx in range(len(result_df.columns)) if idx != drop_idx]
    return result_df.iloc[:, keep_idx]


def _capture_statistics_restore(df: pd.DataFrame) -> StatisticsMzRtRestore:
    return StatisticsMzRtRestore(
        mz_column_name=str(df.columns[0]),
        rt_column_name=str(df.columns[1]),
        mz_values=df.iloc[:, 0].tolist(),
        rt_values=df.iloc[:, 1].tolist(),
    )


def _load_method_state(
    method_file: Optional[Union[str, Path]],
    sample_type_mapping: Optional[dict[str, str]],
    parse_method_file: ParseMethodFile,
    parse_injection_sequence: ParseInjectionSequence,
) -> tuple[dict[str, str], list[InjectionInfo]]:
    sample_mapping: dict[str, str] = {}
    injection_info_list: list[InjectionInfo] = []

    if method_file:
        sample_mapping = parse_method_file(method_file)
        injection_info_list = parse_injection_sequence(method_file)

    if sample_type_mapping:
        sample_mapping.update(sample_type_mapping)

    return sample_mapping, injection_info_list


def _is_cancelled(cancellation_requested: Optional[CancellationChecker]) -> bool:
    return bool(cancellation_requested is not None and cancellation_requested())
