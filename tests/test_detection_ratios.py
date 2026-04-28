from __future__ import annotations

import numpy as np
import pandas as pd

from ms_core.preprocessing.detection_ratios import DetectionRatioCalculator
from ms_core.preprocessing.feature_groups import FeatureGroupDetector
from ms_core.preprocessing.ms_quality_filter import FeatureFilter
from ms_core.preprocessing.settings import FeatureFilterConfig


def _ratio_df(include_qc: bool = True) -> pd.DataFrame:
    data = {
        "Mz/RT": ["Sample_Type", "100/1", "200/2"],
        "A1": ["case", 6000, 0],
        "A2": ["case", 4000, 7000],
        "B1": ["control", 8000, 0],
    }
    if include_qc:
        data["QC1"] = ["qc", 9000, 0]
    return pd.DataFrame(data)


def test_creates_group_ratio_columns() -> None:
    df = _ratio_df()
    group_info = FeatureGroupDetector().detect(df)

    result_df, ratio_cols, _ = DetectionRatioCalculator().calculate(df.copy(), group_info)

    assert ratio_cols["case"] == "case_ratio"
    assert ratio_cols["control"] == "control_ratio"
    assert result_df["case_ratio"].tolist() == ["na", 0.5, 0.5]
    assert result_df["control_ratio"].tolist() == ["na", 1.0, 0.0]


def test_creates_qc_ratio_column_when_qc_exists() -> None:
    df = _ratio_df(include_qc=True)
    group_info = FeatureGroupDetector().detect(df)

    result_df, ratio_cols, _ = DetectionRatioCalculator().calculate(df.copy(), group_info)

    assert ratio_cols["QC"] == "QC_ratio"
    assert result_df["QC_ratio"].tolist() == ["na", 1.0, 0.0]


def test_omits_qc_ratio_column_when_qc_does_not_exist() -> None:
    df = _ratio_df(include_qc=False)
    group_info = FeatureGroupDetector().detect(df)

    result_df, ratio_cols, _ = DetectionRatioCalculator().calculate(df.copy(), group_info)

    assert "QC" not in ratio_cols
    assert "QC_ratio" not in result_df.columns


def test_returns_numeric_block_with_values_all_columns_and_mapping() -> None:
    df = _ratio_df()
    group_info = FeatureGroupDetector().detect(df)

    _, _, numeric_block = DetectionRatioCalculator().calculate(df.copy(), group_info)

    assert numeric_block["all_cols"] == [1, 2, 3, 4]
    assert numeric_block["col_pos"] == {1: 0, 2: 1, 3: 2, 4: 3}
    np.testing.assert_array_equal(
        numeric_block["values"],
        np.array([[6000, 4000, 8000, 9000], [0, 7000, 0, 0]]),
    )


def test_preserves_sample_type_header_row_values_as_na() -> None:
    df = _ratio_df()
    group_info = FeatureGroupDetector().detect(df)

    result_df, ratio_cols, _ = FeatureFilter()._calculate_ratios(df.copy(), group_info)

    for ratio_col in ratio_cols.values():
        assert result_df.at[0, ratio_col] == "na"


def test_uses_configured_signal_threshold() -> None:
    df = _ratio_df()
    group_info = FeatureGroupDetector().detect(df)
    config = FeatureFilterConfig(signal_threshold=8000)

    result_df, _, _ = DetectionRatioCalculator(config).calculate(df.copy(), group_info)

    assert result_df["case_ratio"].tolist() == ["na", 0.0, 0.0]
    assert result_df["control_ratio"].tolist() == ["na", 1.0, 0.0]
