from __future__ import annotations

import numpy as np
import pandas as pd

from ms_core.preprocessing.detection_ratios import DetectionRatioCalculator
from ms_core.preprocessing.feature_filter_decisions import (
    FeatureFilterDecisionTable,
    FeatureFilterOptions,
    FeatureFilterThresholds,
)
from ms_core.preprocessing.feature_groups import FeatureGroupDetector


def _thresholds(
    background: float = 0.8,
    high_det: float = 0.8,
    low_det: float = 0.2,
    qc_ratio: float = 0.5,
    intensity_fc: float = 2.0,
    ratio_rescue: float = 2.0,
) -> FeatureFilterThresholds:
    return FeatureFilterThresholds(
        background=background,
        high_det=high_det,
        low_det=low_det,
        qc_ratio=qc_ratio,
        intensity_fc=intensity_fc,
        ratio_rescue=ratio_rescue,
    )


def _options(
    *,
    background: bool = True,
    qc_ratio: bool = True,
    intensity_fc: bool = False,
    mnar: bool = True,
    single_group: bool = False,
    ratio_rescue: bool = True,
) -> FeatureFilterOptions:
    return FeatureFilterOptions(
        enable_background=background,
        enable_qc_ratio=qc_ratio,
        enable_intensity_fc=intensity_fc,
        enable_mnar=mnar,
        allow_single_group_stable=single_group,
        enable_ratio_rescue=ratio_rescue,
    )


def _two_group_df(
    a_values: list[int],
    b_values: list[int],
    qc_values: list[int] | None = None,
) -> pd.DataFrame:
    data: dict[str, list[object]] = {"feature": ["Sample_Type", "f1"]}
    for i, value in enumerate(a_values, start=1):
        data[f"A{i}"] = ["a", value]
    for i, value in enumerate(b_values, start=1):
        data[f"B{i}"] = ["b", value]
    if qc_values is not None:
        for i, value in enumerate(qc_values, start=1):
            data[f"QC{i}"] = ["qc", value]
    return pd.DataFrame(data)


def _prepared(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object], dict[str, str], dict[str, object]]:
    group_info = FeatureGroupDetector().detect(df)
    ratio_df, ratio_cols, numeric_block = DetectionRatioCalculator().calculate(df.copy(), group_info)
    return ratio_df, group_info, ratio_cols, numeric_block


def _decide(
    df: pd.DataFrame,
    options: FeatureFilterOptions,
    thresholds: FeatureFilterThresholds | None = None,
    protected_rows: set[int] | None = None,
):
    ratio_df, group_info, ratio_cols, numeric_block = _prepared(df)
    return FeatureFilterDecisionTable().decide(
        ratio_df,
        group_info,
        ratio_cols,
        thresholds or _thresholds(),
        options,
        protected_rows or set(),
        numeric_block,
    )


def test_small_n_groups_use_observed_detection_ratio() -> None:
    df = _two_group_df([6000, 6000, 6000, 6000, 0], [6000, 6000, 6000, 6000, 0])

    result = _decide(df, _options(qc_ratio=False, mnar=False))

    assert result.stable_keep.tolist() == [True]
    assert result.keep_mask.tolist() == [True]


def test_stable_gate_keeps_when_two_groups_pass_threshold() -> None:
    df = _two_group_df([6000] * 10, [6000] * 10)

    result = _decide(df, _options(qc_ratio=False, mnar=False))

    assert result.stable_keep.tolist() == [True]
    assert result.keep_mask.tolist() == [True]


def test_mnar_high_low_gate_keeps_presence_absence_marker() -> None:
    df = _two_group_df([6000] * 10, [0] * 10)

    result = _decide(df, _options(background=False, qc_ratio=False, mnar=True))

    assert result.mnar_keep.tolist() == [True]
    assert result.keep_mask.tolist() == [True]


def test_qc_zero_forced_delete_gate() -> None:
    df = _two_group_df([6000] * 10, [6000] * 10, qc_values=[0])

    result = _decide(df, _options(mnar=False))

    assert result.qc_zero.tolist() == [True]
    assert result.keep_mask.tolist() == [False]
    assert result.stats["qc_zero_deleted"] == 1


def test_qc_low_forced_delete_gate() -> None:
    df = _two_group_df([6000] * 10, [6000] * 10, qc_values=[6000, 0])

    result = _decide(df, _options(mnar=False), _thresholds(qc_ratio=0.75))

    assert result.qc_low.tolist() == [True]
    assert result.keep_mask.tolist() == [False]
    assert result.stats["qc_low_deleted"] == 1


def test_intensity_fold_change_gate() -> None:
    df = _two_group_df([1000, 1000], [4000, 4000])

    result = _decide(
        df,
        _options(background=False, qc_ratio=False, intensity_fc=True, mnar=False),
        _thresholds(intensity_fc=4.0),
    )

    assert result.intensity_fc_keep.tolist() == [True]
    assert result.keep_mask.tolist() == [True]


def test_protected_row_override() -> None:
    df = _two_group_df([0], [0], qc_values=[0])

    result = _decide(df, _options(), protected_rows={1})

    assert result.protected_mask.tolist() == [True]
    assert result.keep_mask.tolist() == [True]
    assert result.stats["protected_kept"] == 1


def test_unique_contribution_stats() -> None:
    df = pd.DataFrame(
        {
            "feature": ["Sample_Type", "stable", "mnar", "fc"],
            **{f"A{i}": ["a", 6000, 6000, 1000] for i in range(1, 11)},
            **{f"B{i}": ["b", 6000, 0, 4000] for i in range(1, 11)},
        }
    )

    result = _decide(
        df,
        _options(background=True, qc_ratio=False, intensity_fc=True, mnar=True),
        _thresholds(background=0.8, intensity_fc=4.0),
    )

    assert result.stats["unique_stable_kept"] == 1
    assert result.stats["unique_mnar_kept"] == 1
    assert result.stats["unique_intensity_fc_kept"] == 1
    np.testing.assert_array_equal(result.keep_mask, np.array([True, True, True]))


def _two_group_df_counts(
    a_detected: int,
    a_total: int,
    b_detected: int,
    b_total: int,
    qc_values: list[int] | None = None,
) -> pd.DataFrame:
    a_values = [6000] * a_detected + [0] * (a_total - a_detected)
    b_values = [6000] * b_detected + [0] * (b_total - b_detected)
    return _two_group_df(a_values, b_values, qc_values=qc_values)


def _three_group_df_counts(
    a_detected: int,
    a_total: int,
    b_detected: int,
    b_total: int,
    c_detected: int,
    c_total: int,
) -> pd.DataFrame:
    data: dict[str, list[object]] = {"feature": ["Sample_Type", "f1"]}
    for i in range(1, a_total + 1):
        data[f"A{i}"] = ["a", 6000 if i <= a_detected else 0]
    for i in range(1, b_total + 1):
        data[f"B{i}"] = ["b", 6000 if i <= b_detected else 0]
    for i in range(1, c_total + 1):
        data[f"C{i}"] = ["c", 6000 if i <= c_detected else 0]
    return pd.DataFrame(data)


def test_ratio_rescue_keeps_dead_zone_feature() -> None:
    df = _two_group_df_counts(a_detected=4, a_total=10, b_detected=2, b_total=10)

    result = _decide(
        df,
        _options(qc_ratio=False, mnar=True),
        _thresholds(background=0.5, high_det=0.5, low_det=0.1, ratio_rescue=2.0),
    )

    assert result.ratio_rescue_keep.tolist() == [True]
    assert result.stable_keep.tolist() == [False]
    assert result.mnar_keep.tolist() == [False]
    assert result.keep_mask.tolist() == [True]


def test_ratio_rescue_keeps_32_16_with_default_mnar_floor() -> None:
    df = _two_group_df_counts(a_detected=8, a_total=25, b_detected=4, b_total=25)

    result = _decide(
        df,
        _options(qc_ratio=False, mnar=True),
        _thresholds(background=0.33, high_det=0.8, low_det=0.2, ratio_rescue=2.0),
    )

    assert result.stable_keep.tolist() == [False]
    assert result.mnar_keep.tolist() == [False]
    assert result.ratio_rescue_keep.tolist() == [True]
    assert result.keep_mask.tolist() == [True]


def test_ratio_rescue_rejects_when_any_group_detection_below_ten_percent() -> None:
    df = _two_group_df_counts(a_detected=3, a_total=10, b_detected=1, b_total=11)

    result = _decide(
        df,
        _options(qc_ratio=False, mnar=False),
        _thresholds(background=0.33, low_det=0.2, ratio_rescue=2.0),
    )

    assert result.ratio_rescue_keep.tolist() == [False]
    assert result.keep_mask.tolist() == [False]


def test_ratio_rescue_rejects_when_min_below_ten_percent_floor() -> None:
    df = _two_group_df_counts(a_detected=6, a_total=20, b_detected=1, b_total=20)

    result = _decide(
        df,
        _options(qc_ratio=False, mnar=False),
        _thresholds(background=0.5, low_det=0.1, ratio_rescue=2.0),
    )

    assert result.ratio_rescue_keep.tolist() == [False]


def test_ratio_rescue_rejects_when_ratio_below_threshold() -> None:
    df = _two_group_df_counts(a_detected=5, a_total=20, b_detected=3, b_total=20)

    result = _decide(
        df,
        _options(qc_ratio=False, mnar=False),
        _thresholds(background=0.5, high_det=0.5, low_det=0.1, ratio_rescue=2.0),
    )

    assert result.ratio_rescue_keep.tolist() == [False]
    assert result.keep_mask.tolist() == [False]


def test_ratio_rescue_handles_zero_min_safely() -> None:
    df = _two_group_df_counts(a_detected=5, a_total=10, b_detected=0, b_total=10)

    result = _decide(
        df,
        _options(qc_ratio=False, mnar=False),
        _thresholds(background=0.8, low_det=0.1, ratio_rescue=2.0),
    )

    assert result.ratio_rescue_keep.tolist() == [False]


def test_ratio_rescue_disabled_yields_legacy_behavior() -> None:
    df = _two_group_df_counts(a_detected=4, a_total=10, b_detected=2, b_total=10)

    enabled = _decide(
        df,
        _options(qc_ratio=False, mnar=True, ratio_rescue=True),
        _thresholds(background=0.5, high_det=0.5, low_det=0.1, ratio_rescue=2.0),
    )
    disabled = _decide(
        df,
        _options(qc_ratio=False, mnar=True, ratio_rescue=False),
        _thresholds(background=0.5, high_det=0.5, low_det=0.1, ratio_rescue=2.0),
    )

    assert enabled.keep_mask.tolist() == [True]
    assert disabled.keep_mask.tolist() == [False]
    assert disabled.ratio_rescue_keep.tolist() == [False]


def test_ratio_rescue_overrides_qc_force_delete() -> None:
    df = _two_group_df_counts(
        a_detected=4,
        a_total=10,
        b_detected=2,
        b_total=10,
        qc_values=[0, 0, 0],
    )

    result = _decide(
        df,
        _options(qc_ratio=True, mnar=True),
        _thresholds(
            background=0.5,
            high_det=0.5,
            low_det=0.1,
            qc_ratio=0.5,
            ratio_rescue=2.0,
        ),
    )

    assert result.qc_zero.tolist() == [True]
    assert result.ratio_rescue_keep.tolist() == [True]
    assert result.qc_force_delete.tolist() == [False]
    assert result.keep_mask.tolist() == [True]


def test_ratio_rescue_three_groups_uses_global_max_min() -> None:
    df = _three_group_df_counts(
        a_detected=4, a_total=10,
        b_detected=2, b_total=10,
        c_detected=3, c_total=10,
    )

    result = _decide(
        df,
        _options(qc_ratio=False, mnar=True),
        _thresholds(background=0.5, high_det=0.5, low_det=0.1, ratio_rescue=2.0),
    )

    assert result.stable_keep.tolist() == [False]
    assert result.mnar_keep.tolist() == [False]
    assert result.ratio_rescue_keep.tolist() == [True]
    assert result.keep_mask.tolist() == [True]


def test_unique_ratio_rescue_stats_excludes_overlap() -> None:
    df = pd.DataFrame(
        {
            "feature": ["Sample_Type", "stable_only", "rescue_only", "stable_and_rescue"],
            **{f"A{i}": ["a", 6000, 6000 if i <= 4 else 0, 6000 if i <= 8 else 0] for i in range(1, 11)},
            **{f"B{i}": ["b", 6000, 6000 if i <= 2 else 0, 6000 if i <= 4 else 0] for i in range(1, 11)},
        }
    )

    result = _decide(
        df,
        _options(background=True, qc_ratio=False, mnar=False, ratio_rescue=True),
        _thresholds(background=0.3, low_det=0.1, ratio_rescue=2.0),
    )

    assert result.stable_keep.tolist() == [True, False, True]
    assert result.ratio_rescue_keep.tolist() == [False, True, True]
    assert result.stats["unique_stable_kept"] == 1
    assert result.stats["unique_ratio_rescue_kept"] == 1
    assert result.stats["ratio_rescue_kept"] == 2
