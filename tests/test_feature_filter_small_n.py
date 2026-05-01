"""Tests for small-N observed detection-ratio behavior in FeatureFilter."""

from __future__ import annotations
import pandas as pd
from ms_core.preprocessing.ms_quality_filter import FeatureFilter


# ── Helper to build minimal FeatureFilter-ready DataFrame ──────────


def _make_two_group_df(
    n_a: int,
    present_a: int,
    n_b: int,
    present_b: int,
    signal: int = 10000,
) -> pd.DataFrame:
    """Two biological groups A and B. present_X = number of samples with signal."""
    rows: dict[str, list] = {"feature": ["Sample_Type", "f1"]}
    for i in range(present_a):
        rows[f"a_{i + 1}"] = ["a", signal]
    for i in range(present_a, n_a):
        rows[f"a_{i + 1}"] = ["a", 0]
    for i in range(present_b):
        rows[f"b_{i + 1}"] = ["b", signal]
    for i in range(present_b, n_b):
        rows[f"b_{i + 1}"] = ["b", 0]
    return pd.DataFrame(rows)


# ── Stable gate tests ───────────────────────────────────────────────


def test_stable_gate_small_n_uses_observed_ratio() -> None:
    """4/5 = 80% passes an 80% threshold because Step4 uses observed ratios."""
    df = _make_two_group_df(n_a=5, present_a=4, n_b=5, present_b=4)
    ff = FeatureFilter()
    result = ff.process(
        df,
        background_threshold=0.8,
        enable_background_threshold=True,
        enable_qc_ratio_threshold=False,
        enable_mnar_gate=False,
    )
    assert result.success
    assert result.statistics["kept_count"] == 1


def test_stable_gate_large_n_accepts_raw_ratio() -> None:
    """16/20 = 80% should pass 80% threshold with the same observed-ratio rule."""
    df = _make_two_group_df(n_a=20, present_a=16, n_b=20, present_b=16)
    ff = FeatureFilter()
    result = ff.process(
        df,
        background_threshold=0.8,
        enable_background_threshold=True,
        enable_qc_ratio_threshold=False,
        enable_mnar_gate=False,
    )
    assert result.success
    assert result.statistics["kept_count"] == 1


# ── Statistics fields tests ─────────────────────────────────────────


def test_statistics_include_qc_count_and_group_counts() -> None:
    """process() statistics must include qc_count and group_counts."""
    rows: dict[str, list] = {"feature": ["Sample_Type", "f1"]}
    for i in range(3):
        rows[f"a_{i + 1}"] = ["a", 10000]
    for i in range(5):
        rows[f"b_{i + 1}"] = ["b", 10000]
    for i in range(2):
        rows[f"QC_{i + 1}"] = ["qc", 10000]
    df = pd.DataFrame(rows)
    ff = FeatureFilter()
    result = ff.process(df, enable_qc_ratio_threshold=False, enable_mnar_gate=False)
    assert result.success
    assert result.statistics["qc_count"] == 2
    assert result.statistics["group_counts"]["a"] == 3
    assert result.statistics["group_counts"]["b"] == 5


def test_process_facade_preserves_output_builder_contract() -> None:
    rows: dict[str, list] = {
        "feature": ["Sample_Type", "stable", "mnar", "deleted"],
    }
    for i in range(10):
        rows[f"a_{i + 1}"] = ["a", 10000, 10000, 10000]
        rows[f"b_{i + 1}"] = ["b", 10000, 0, 10000]
    rows["QC_1"] = ["qc", 10000, 10000, 0]

    result = FeatureFilter().process(
        pd.DataFrame(rows),
        background_threshold=0.8,
        high_det_thresh=0.8,
        low_det_thresh=0.2,
        qc_ratio_threshold=0.5,
        enable_background_threshold=True,
        enable_qc_ratio_threshold=True,
        enable_mnar_gate=True,
    )

    assert result.success
    output = result.data
    assert output is not None
    assert output["feature"].tolist() == ["Sample_Type", "stable", "mnar"]
    assert "a_ratio" in output.columns
    assert "b_ratio" in output.columns
    assert "QC_ratio" in output.columns
    assert output["is_Presence_Absence_Marker"].tolist() == [
        "is_Presence_Absence_Marker",
        False,
        True,
    ]
    assert pd.isna(output.loc[2, "b_1"])
    assert result.metadata["deleted_features"][0]["feature"] == "deleted"
    assert result.statistics["zeros_converted_to_nan"] == 10
