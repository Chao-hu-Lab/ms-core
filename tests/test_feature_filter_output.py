from __future__ import annotations

import numpy as np
import pandas as pd

from ms_core.preprocessing.feature_filter_decisions import FeatureFilterDecisionResult
from ms_core.preprocessing.feature_filter_output import FeatureFilterOutputBuilder


def _decision(keep_mask: list[bool], mnar_keep: list[bool] | None = None) -> FeatureFilterDecisionResult:
    n_features = len(keep_mask)
    return FeatureFilterDecisionResult(
        keep_mask=np.array(keep_mask, dtype=bool),
        stable_keep=np.zeros(n_features, dtype=bool),
        mnar_keep=np.array(mnar_keep or [False] * n_features, dtype=bool),
        intensity_fc_keep=np.zeros(n_features, dtype=bool),
        protected_mask=np.zeros(n_features, dtype=bool),
        qc_zero=np.zeros(n_features, dtype=bool),
        qc_low=np.zeros(n_features, dtype=bool),
        qc_force_delete=np.zeros(n_features, dtype=bool),
        stats={
            "kept_count": int(sum(keep_mask)),
            "deleted_count": int(n_features - sum(keep_mask)),
        },
    )


def test_zero_to_nan_touches_only_sample_qc_data_columns_after_header_row() -> None:
    df = pd.DataFrame(
        {
            "feature": ["Sample_Type", "f1"],
            "A1": ["a", 0],
            "QC1": ["qc", 0],
            "a_ratio": ["na", 0],
            "QC_ratio": ["na", 0],
        }
    )
    group_info = {"groups": {"a": [1]}, "qc_cols": [2], "has_qc": True}

    result_df, _, stats = FeatureFilterOutputBuilder().build(
        df,
        group_info,
        _decision([True]),
        protected_rows=set(),
    )

    assert result_df.at[0, "A1"] == "a"
    assert result_df.at[0, "QC1"] == "qc"
    assert pd.isna(result_df.at[1, "A1"])
    assert pd.isna(result_df.at[1, "QC1"])
    assert result_df.at[1, "feature"] == "f1"
    assert result_df.at[1, "a_ratio"] == 0
    assert result_df.at[1, "QC_ratio"] == 0
    assert stats["zeros_converted_to_nan"] == 2


def test_deleted_features_preserve_expected_row_shape() -> None:
    df = pd.DataFrame(
        {
            "feature": ["Sample_Type", "kept", "deleted"],
            "A1": ["a", 10, 20],
            "a_ratio": ["na", 1.0, 1.0],
        }
    )
    group_info = {"groups": {"a": [1]}, "qc_cols": [], "has_qc": False}

    result_df, deleted_features, _ = FeatureFilterOutputBuilder().build(
        df,
        group_info,
        _decision([True, False]),
        protected_rows=set(),
    )

    assert result_df["feature"].tolist() == ["Sample_Type", "kept"]
    assert len(deleted_features) == 1
    assert deleted_features[0].index.tolist() == df.columns.tolist()
    assert deleted_features[0]["feature"] == "deleted"


def test_presence_absence_marker_column_follows_mnar_mask() -> None:
    df = pd.DataFrame(
        {
            "feature": ["Sample_Type", "mnar", "ordinary"],
            "A1": ["a", 10, 20],
            "a_ratio": ["na", 1.0, 1.0],
        }
    )
    group_info = {"groups": {"a": [1]}, "qc_cols": [], "has_qc": False}

    result_df, _, _ = FeatureFilterOutputBuilder().build(
        df,
        group_info,
        _decision([True, True], mnar_keep=[True, False]),
        protected_rows=set(),
    )

    assert result_df["is_Presence_Absence_Marker"].tolist() == [
        "is_Presence_Absence_Marker",
        True,
        False,
    ]


def test_protected_row_remapping_remains_stable() -> None:
    df = pd.DataFrame(
        {
            "feature": ["Sample_Type", "deleted", "protected"],
            "A1": ["a", 10, 20],
            "a_ratio": ["na", 1.0, 1.0],
        }
    )
    group_info = {"groups": {"a": [1]}, "qc_cols": [], "has_qc": False}

    result_df, _, stats = FeatureFilterOutputBuilder().build(
        df,
        group_info,
        _decision([False, True]),
        protected_rows={2},
    )

    assert result_df["feature"].tolist() == ["Sample_Type", "protected"]
    assert stats["red_font_rows"] == [1]
