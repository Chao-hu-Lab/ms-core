from __future__ import annotations

import pandas as pd

from ms_core.preprocessing.data_organizer_layout import (
    expand_pre_merged_mz_rt,
    extract_sample_type_row_from_input,
    is_pre_merged_mz_rt_header,
    move_leading_metadata_to_end,
    normalize_sample_type_value,
)


def test_layout_helpers_expand_pre_merged_mz_rt_column() -> None:
    df = pd.DataFrame(
        {
            "Mz/RT": ["100.1234/1.23", "not-a-feature"],
            "SampleA": [10, 20],
        }
    )

    result, expanded = expand_pre_merged_mz_rt(df)

    assert expanded is True
    assert list(result.columns) == ["Mz", "RT", "SampleA"]
    assert result.loc[0, "Mz"] == 100.1234
    assert result.loc[0, "RT"] == 1.23
    assert pd.isna(result.loc[1, "Mz"])
    assert pd.isna(result.loc[1, "RT"])


def test_layout_helpers_extract_sample_type_row_with_normalized_values() -> None:
    df = pd.DataFrame(
        [
            ["Sample Type", "na", "tumor", "quality control"],
            [100.0, 1.0, 10, 20],
        ],
        columns=["Mz", "RT", "TumorBC1_DNA", "pooled_QC1"],
    )

    extraction = extract_sample_type_row_from_input(df)

    assert extraction.sample_types == {
        "TumorBC1_DNA": "Exposure",
        "pooled_QC1": "QC",
    }
    assert extraction.stats == {
        "sample_types_from_input": True,
        "input_sample_type_count": 2,
    }
    assert extraction.data.shape == (1, 4)
    assert extraction.data.iloc[0, 0] == 100.0


def test_layout_helpers_move_leading_metadata_after_sample_columns() -> None:
    df = pd.DataFrame(
        {
            "MZmine ID": ["f1"],
            "Mz": [100.0],
            "RT": [1.0],
            "SampleA": [10],
        }
    )

    result = move_leading_metadata_to_end(df)

    assert list(result.columns) == ["Mz", "RT", "SampleA", "MZmine ID"]


def test_layout_helper_normalizers_match_facade_contract() -> None:
    assert is_pre_merged_mz_rt_header("m/z RT") is False
    assert is_pre_merged_mz_rt_header("m/z/rt") is True
    assert normalize_sample_type_value("Benign Fat") == "Control"
