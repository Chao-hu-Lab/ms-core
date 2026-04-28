from __future__ import annotations

import pandas as pd

from ms_core.preprocessing.feature_groups import FeatureGroupDetector
from ms_core.preprocessing.ms_quality_filter import FeatureFilter
from ms_core.preprocessing.settings import FeatureFilterConfig


def test_detects_analysis_groups_from_sample_type_row() -> None:
    df = pd.DataFrame(
        {
            "Mz/RT": ["Sample_Type", "100/1"],
            "A1": ["case", 10],
            "A2": ["case", 20],
            "B1": ["control", 30],
        }
    )

    info = FeatureGroupDetector().detect(df)

    assert info["groups"] == {"case": [1, 2], "control": [3]}
    assert info["has_qc"] is False


def test_detects_qc_columns() -> None:
    df = pd.DataFrame(
        {
            "Mz/RT": ["Sample_Type", "100/1"],
            "A1": ["case", 10],
            "QC1": ["qc", 20],
            "QC2": ["QC", 30],
        }
    )

    info = FeatureGroupDetector().detect(df)

    assert info["groups"] == {"case": [1]}
    assert info["qc_cols"] == [2, 3]
    assert info["has_qc"] is True


def test_respects_excluded_types_from_config() -> None:
    config = FeatureFilterConfig(excluded_types=["blank", "standard"])
    df = pd.DataFrame(
        {
            "Mz/RT": ["Sample_Type", "100/1"],
            "A1": ["case", 10],
            "Blank1": ["blank", 0],
            "Std1": ["standard", 100],
        }
    )

    info = FeatureGroupDetector(config).detect(df)

    assert info["groups"] == {"case": [1]}
    assert info["excluded_cols"] == [2, 3]


def test_handles_missing_fixed_columns_consistently_with_current_step4() -> None:
    df = pd.DataFrame(
        {
            "feature": ["Sample_Type", "f1"],
            "A1": ["case", 10],
            "QC1": ["qc", 20],
        }
    )

    info = FeatureGroupDetector().detect(df)

    assert info["groups"] == {"case": [1]}
    assert info["qc_cols"] == [2]


def test_count_analysis_groups_compatibility() -> None:
    df = pd.DataFrame(
        {
            "feature": ["Sample_Type", "f1"],
            "A1": ["case", 10],
            "B1": ["control", 20],
            "QC1": ["qc", 30],
        }
    )

    assert FeatureFilter().count_analysis_groups(df) == 2
