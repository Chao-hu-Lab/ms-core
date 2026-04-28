from __future__ import annotations

import pandas as pd

from ms_core.preprocessing.method_sequence import InjectionInfo
from ms_core.preprocessing.sample_info_builder import SampleInfoBuilder, is_non_sample_column


def test_builder_preserves_unmatched_raw_column_as_sample_name() -> None:
    df = pd.DataFrame(
        [
            ["Sample_Type", "sample"],
            ["100.1000/1.00", 1.0],
        ],
        columns=["Mz/RT", "BC9999_2"],
    )

    sample_info = SampleInfoBuilder().build(df, [])

    assert sample_info.loc[0, "Sample_Name"] == "BC9999_2"
    assert sample_info.loc[0, "Method_Sample_Name"] == ""
    assert sample_info.loc[0, "Injection_Order"] == 999
    assert sample_info.loc[0, "Injection_Volume"] == 0
    assert sample_info.loc[0, "_col_name"] == "BC9999_2"


def test_builder_uses_earliest_method_row_for_duplicate_matching_keys() -> None:
    df = pd.DataFrame(
        [
            ["Sample_Type", "sample"],
            ["100.1000/1.00", 1.0],
        ],
        columns=["Mz/RT", "BC2286_2"],
    )
    injection_info = [
        InjectionInfo(7, "BC2286_2", "BC2286_2", 11.0, "late method"),
        InjectionInfo(3, "BC2286", "BC2286", 5.0, "early method"),
    ]

    sample_info = SampleInfoBuilder().build(df, injection_info)

    assert len(sample_info) == 1
    assert sample_info.loc[0, "Sample_Name"] == "BC2286_2"
    assert sample_info.loc[0, "Method_Sample_Name"] == "BC2286"
    assert sample_info.loc[0, "Injection_Order"] == 1
    assert sample_info.loc[0, "Injection_Volume"] == 5.0


def test_builder_matches_ambiguous_bc_rows_by_variant() -> None:
    df = pd.DataFrame(
        [
            ["Sample_Type", "sample"],
            ["100.1000/1.00", 1.0],
        ],
        columns=["Mz/RT", "TumorBC2257_RNA"],
    )
    injection_info = [
        InjectionInfo(1, "Tumor tissue BC2257_DNA", "TumorBC2257_DNA", 5.0),
        InjectionInfo(2, "Tumor tissue BC2257_RNA", "TumorBC2257_RNA", 7.5),
    ]

    sample_info = SampleInfoBuilder().build(df, injection_info)

    assert sample_info.loc[0, "Sample_Name"] == "TumorBC2257_RNA"
    assert sample_info.loc[0, "Method_Sample_Name"] == "Tumor tissue BC2257_RNA"
    assert sample_info.loc[0, "Injection_Order"] == 2
    assert sample_info.loc[0, "Injection_Volume"] == 7.5


def test_builder_does_not_mutate_injection_records_when_renumbering() -> None:
    df = pd.DataFrame(
        [
            ["Sample_Type", "sample"],
            ["100.1000/1.00", 1.0],
        ],
        columns=["Mz/RT", "BC2286"],
    )
    injection_info = [InjectionInfo(3, "BC2286", "BC2286", 5.0)]

    SampleInfoBuilder().build(df, injection_info)

    assert injection_info[0].injection_order == 3


def test_is_non_sample_column_identifies_metadata_columns() -> None:
    assert is_non_sample_column("MZmine RT [min]")
    assert is_non_sample_column("row ID")
    assert is_non_sample_column("Unnamed: 42")
    assert not is_non_sample_column("BC2286")
