from __future__ import annotations

import numpy as np
import pandas as pd

from ms_core.preprocessing.feature_filter_decisions import FeatureFilterDecisionResult
from ms_core.preprocessing.feature_filter_output import FeatureFilterOutputBuilder


def _decision(
    keep_mask: list[bool],
    stable_keep: list[bool] | None = None,
    mnar_keep: list[bool] | None = None,
    intensity_fc_keep: list[bool] | None = None,
    ratio_rescue_keep: list[bool] | None = None,
    protected_mask: list[bool] | None = None,
    qc_zero: list[bool] | None = None,
    qc_low: list[bool] | None = None,
    qc_force_delete: list[bool] | None = None,
    imputation_tag: list[bool] | None = None,
    tag_reason_structural: list[bool] | None = None,
    tag_reason_low_overall: list[bool] | None = None,
    unfiltered_keep: list[bool] | None = None,
) -> FeatureFilterDecisionResult:
    n_features = len(keep_mask)
    return FeatureFilterDecisionResult(
        keep_mask=np.array(keep_mask, dtype=bool),
        stable_keep=np.array(stable_keep or [False] * n_features, dtype=bool),
        mnar_keep=np.array(mnar_keep or [False] * n_features, dtype=bool),
        intensity_fc_keep=np.array(intensity_fc_keep or [False] * n_features, dtype=bool),
        ratio_rescue_keep=np.array(
            ratio_rescue_keep or [False] * n_features, dtype=bool
        ),
        protected_mask=np.array(protected_mask or [False] * n_features, dtype=bool),
        qc_zero=np.array(qc_zero or [False] * n_features, dtype=bool),
        qc_low=np.array(qc_low or [False] * n_features, dtype=bool),
        qc_force_delete=np.array(qc_force_delete or [False] * n_features, dtype=bool),
        all_groups_pass_background=np.ones(n_features, dtype=bool),
        any_group_zero=np.zeros(n_features, dtype=bool),
        any_group_nonzero=np.ones(n_features, dtype=bool),
        imputation_tag=np.array(imputation_tag or [False] * n_features, dtype=bool),
        tag_reason_structural=np.array(
            tag_reason_structural or [False] * n_features, dtype=bool
        ),
        tag_reason_low_overall=np.array(
            tag_reason_low_overall or [False] * n_features, dtype=bool
        ),
        unfiltered_keep=np.array(unfiltered_keep or [False] * n_features, dtype=bool),
        stats={
            "kept_count": int(sum(keep_mask)),
            "deleted_count": int(n_features - sum(keep_mask)),
        },
    )


def _simple_df(feature_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": ["Sample_Type", *feature_names],
            "A1": ["a", *range(10, 10 + len(feature_names))],
            "a_ratio": ["na", *([1.0] * len(feature_names))],
        }
    )


def test_output_emits_two_new_metadata_columns_after_marker() -> None:
    df = _simple_df(["f1"])
    group_info = {"groups": {"a": [1]}, "qc_cols": [], "has_qc": False}

    result_df, _, _ = FeatureFilterOutputBuilder().build(
        df,
        group_info,
        _decision([True]),
        protected_rows=set(),
    )

    marker_idx = result_df.columns.get_loc("is_Presence_Absence_Marker")
    assert result_df.columns[marker_idx + 1 : marker_idx + 3].tolist() == [
        "Feature_Filter_Keep_Reasons",
        "Imputation_Tag_Reasons",
    ]
    assert "Detection_Profile" not in result_df.columns


def test_output_tag_value_uses_decision_imputation_tag_not_legacy_or() -> None:
    df = _simple_df(["legacy_true_contract_false", "legacy_false_contract_true"])
    group_info = {"groups": {"a": [1]}, "qc_cols": [], "has_qc": False}

    result_df, _, _ = FeatureFilterOutputBuilder().build(
        df,
        group_info,
        _decision(
            [True, True],
            ratio_rescue_keep=[True, False],
            imputation_tag=[False, True],
        ),
        protected_rows=set(),
    )

    assert result_df["is_Presence_Absence_Marker"].tolist() == [
        "is_Presence_Absence_Marker",
        False,
        True,
    ]


def test_output_keep_reasons_token_order_matches_contract() -> None:
    df = _simple_df(["all_positive", "all_positive_protected", "protected", "unfiltered"])
    group_info = {"groups": {"a": [1]}, "qc_cols": [], "has_qc": False}

    result_df, _, _ = FeatureFilterOutputBuilder().build(
        df,
        group_info,
        _decision(
            [True, True, True, True],
            stable_keep=[True, True, False, False],
            mnar_keep=[True, True, False, False],
            intensity_fc_keep=[True, True, False, False],
            ratio_rescue_keep=[True, True, False, False],
            protected_mask=[False, True, True, False],
            unfiltered_keep=[False, False, False, True],
        ),
        protected_rows={2, 3},
    )

    assert result_df["Feature_Filter_Keep_Reasons"].tolist() == [
        "Feature_Filter_Keep_Reasons",
        "stable|mnar|intensity_fc|ratio_rescue",
        "stable|mnar|intensity_fc|ratio_rescue|protected",
        "protected",
        "unfiltered",
    ]


def test_output_tag_reasons_token_order_matches_contract() -> None:
    df = _simple_df(["low", "structural_low", "model_imputable"])
    group_info = {"groups": {"a": [1]}, "qc_cols": [], "has_qc": False}

    result_df, _, _ = FeatureFilterOutputBuilder().build(
        df,
        group_info,
        _decision(
            [True, True, True],
            imputation_tag=[True, True, False],
            tag_reason_structural=[False, True, False],
            tag_reason_low_overall=[True, True, False],
        ),
        protected_rows=set(),
    )

    assert result_df["Imputation_Tag_Reasons"].tolist() == [
        "Imputation_Tag_Reasons",
        "low_overall_detection",
        "structural_absence|low_overall_detection",
        "",
    ]


def test_output_preserves_numeric_ratio_columns_as_detection_source_of_truth() -> None:
    df = pd.DataFrame(
        {
            "feature": ["Sample_Type", "f1"],
            "A1": ["a", 10],
            "QC1": ["qc", 0],
            "a_ratio": ["na", 1.0],
            "QC_ratio": ["na", 0.0],
        }
    )
    group_info = {"groups": {"a": [1]}, "qc_cols": [2], "has_qc": True}

    result_df, _, _ = FeatureFilterOutputBuilder().build(
        df,
        group_info,
        _decision([True]),
        protected_rows=set(),
    )

    assert result_df["a_ratio"].tolist() == ["na", 1.0]
    assert "Detection_Profile" not in result_df.columns


def test_deleted_features_carry_delete_reason_and_existing_ratio_columns_only() -> None:
    df = _simple_df(["deleted"])
    group_info = {"groups": {"a": [1]}, "qc_cols": [], "has_qc": False}

    _, deleted_features, _ = FeatureFilterOutputBuilder().build(
        df,
        group_info,
        _decision([False]),
        protected_rows=set(),
    )

    assert deleted_features[0]["Feature_Filter_Delete_Reasons"] == "no_keep_rule"
    assert deleted_features[0]["a_ratio"] == 1.0
    assert "Detection_Profile" not in deleted_features[0].index
    assert "Feature_Filter_Keep_Reasons" not in deleted_features[0].index
    assert "Imputation_Tag_Reasons" not in deleted_features[0].index


def test_deleted_features_capture_qc_force_delete_reason() -> None:
    df = _simple_df(["deleted_by_qc"])
    group_info = {"groups": {"a": [1]}, "qc_cols": [], "has_qc": False}

    _, deleted_features, _ = FeatureFilterOutputBuilder().build(
        df,
        group_info,
        _decision(
            [False],
            stable_keep=[True],
            qc_zero=[True],
            qc_force_delete=[True],
        ),
        protected_rows=set(),
    )

    assert deleted_features[0]["Feature_Filter_Delete_Reasons"] == "qc_zero"


def test_deleted_features_capture_qc_and_no_keep_rule_combined_reason() -> None:
    df = _simple_df(["deleted_by_qc_and_no_keep"])
    group_info = {"groups": {"a": [1]}, "qc_cols": [], "has_qc": False}

    _, deleted_features, _ = FeatureFilterOutputBuilder().build(
        df,
        group_info,
        _decision(
            [False],
            qc_low=[True],
            qc_force_delete=[True],
        ),
        protected_rows=set(),
    )

    assert deleted_features[0]["Feature_Filter_Delete_Reasons"] == "qc_low|no_keep_rule"


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
    assert deleted_features[0].index.tolist() == [
        *df.columns.tolist(),
        "Feature_Filter_Delete_Reasons",
    ]
    assert deleted_features[0]["feature"] == "deleted"


def test_presence_absence_marker_column_follows_decision_tag_mask() -> None:
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
        _decision([True, True], mnar_keep=[True, False], imputation_tag=[True, False]),
        protected_rows=set(),
    )

    assert result_df["is_Presence_Absence_Marker"].tolist() == [
        "is_Presence_Absence_Marker",
        True,
        False,
    ]


def test_presence_absence_marker_can_include_ratio_rescue_rows_from_decision_tag() -> None:
    df = pd.DataFrame(
        {
            "feature": ["Sample_Type", "mnar_only", "rescue_only", "ordinary"],
            "A1": ["a", 10, 20, 30],
            "a_ratio": ["na", 1.0, 1.0, 1.0],
        }
    )
    group_info = {"groups": {"a": [1]}, "qc_cols": [], "has_qc": False}

    result_df, _, _ = FeatureFilterOutputBuilder().build(
        df,
        group_info,
        _decision(
            [True, True, True],
            mnar_keep=[True, False, False],
            ratio_rescue_keep=[False, True, False],
            imputation_tag=[True, True, False],
        ),
        protected_rows=set(),
    )

    assert result_df["is_Presence_Absence_Marker"].tolist() == [
        "is_Presence_Absence_Marker",
        True,
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
