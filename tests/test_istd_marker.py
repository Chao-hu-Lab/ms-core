"""Unit tests for ISTD Marker (Step 2)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from ms_core.preprocessing.istd_marker import ISTDMarker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(feature_ids: list[str], sample_values: dict[str, list]) -> pd.DataFrame:
    """Build a DataFrame with Sample_Type header row."""
    data = {"Mz/RT": ["Sample_Type"] + feature_ids}
    for col, vals in sample_values.items():
        data[col] = [col.split("_")[0]] + vals  # e.g. "case_S1" -> type "case"
    return pd.DataFrame(data)


def _write_xic_workbook(path: Path, *, mz: float = 261.1273, rt: float = 5.0) -> None:
    targets = pd.DataFrame(
        [
            {
                "Label": "d3-target",
                "Role": "ISTD",
                "ISTD Pair": None,
                "m/z": mz,
                "RT min": rt - 0.1,
                "RT max": rt + 0.1,
                "ppm tol": 20,
            }
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "Target": "d3-target",
                "Role": "ISTD",
                "Detected": 1,
                "Total": 1,
                "Detection %": "100%",
                "Mean RT": rt,
            }
        ]
    )
    with pd.ExcelWriter(path) as writer:
        targets.to_excel(writer, sheet_name="Targets", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)


@pytest.fixture
def xic_workbook(project_temp_dir) -> Iterator[Path]:
    with project_temp_dir("xic-") as temp_dir:
        path = temp_dir / "xic.xlsx"
        _write_xic_workbook(path)
        yield path


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

class TestSortByMz:
    def test_sorts_ascending(self):
        df = _make_df(
            ["300.0000/1.0", "100.0000/2.0", "200.0000/3.0"],
            {"case_S1": [10, 20, 30]},
        )
        marker = ISTDMarker()
        result = marker.process(df, istd_features=set())
        assert result.success
        mz_values = [
            float(str(result.data.iat[i, 0]).split("/")[0])
            for i in range(1, len(result.data))
        ]
        assert mz_values == sorted(mz_values)

    def test_preserves_sample_type_row(self):
        df = _make_df(["100.0000/1.0"], {"case_S1": [10]})
        marker = ISTDMarker()
        result = marker.process(df, istd_features=set())
        assert result.success
        assert str(result.data.iat[0, 0]).lower() == "sample_type"


# ---------------------------------------------------------------------------
# ISTD detection via m/z list
# ---------------------------------------------------------------------------

class TestISTDDetection:
    def test_marks_matching_xic_target(self, xic_workbook):
        """Features matching XIC ISTD targets should be identified."""
        df = _make_df(
            ["261.1273/5.0", "500.0000/3.0"],
            {"case_S1": [1000, 2000], "case_S2": [1100, 2100]},
        )
        marker = ISTDMarker()
        result = marker.process(df, xic_results_file=xic_workbook)
        assert result.success
        assert "261.1273/5.0" in result.metadata.get("istd_features", [])

    def test_no_match_outside_xic_tolerance(self, xic_workbook):
        """Features far from XIC ISTD targets should fail instead of marking rows."""
        df = _make_df(
            ["999.9999/1.0"],
            {"case_S1": [1000]},
        )
        marker = ISTDMarker()
        result = marker.process(df, xic_results_file=xic_workbook)
        assert not result.success
        assert "No ISTD features matched XIC targets" in result.message


# ---------------------------------------------------------------------------
# Duplicate detection (relative to ISTD)
# ---------------------------------------------------------------------------

class TestDuplicateDetection:
    def test_does_not_detect_duplicate_of_istd_in_step2(self):
        """Step 2 only marks the ISTD row; duplicate grouping belongs to Step 3."""
        istd_mz = 261.1273
        close_mz = istd_mz + istd_mz * 5e-6  # 5 ppm away
        df = _make_df(
            [f"{istd_mz:.4f}/5.00", f"{close_mz:.4f}/5.10"],
            {"case_S1": [1000, 900], "case_S2": [1100, 800]},
        )
        marker = ISTDMarker()
        result = marker.process(df, istd_features={f"{istd_mz:.4f}/5.00"})
        assert result.success
        assert result.statistics.get("duplicates_marked", 0) == 0
        assert result.metadata.get("duplicate_indices") == []

    def test_marks_only_explicit_istd_row(self):
        df = _make_df(
            ["261.1273/5.00", "999.0000/20.00"],
            {"case_S1": [1000, 500]},
        )
        marker = ISTDMarker()
        result = marker.process(df, istd_features={"261.1273/5.00"})
        assert result.success
        assert result.statistics.get("duplicates_marked", 0) == 0
        assert result.metadata.get("istd_features") == ["261.1273/5.00"]


# ---------------------------------------------------------------------------
# Red font / protected rows metadata
# ---------------------------------------------------------------------------

class TestMetadataOutput:
    def test_red_font_rows_populated_for_istd(self, xic_workbook):
        df = _make_df(
            ["261.1273/5.00", "500.0000/3.00"],
            {"case_S1": [1000, 2000]},
        )
        marker = ISTDMarker()
        result = marker.process(df, xic_results_file=xic_workbook)
        assert result.success
        assert len(result.metadata.get("red_font_rows", [])) >= 1

    def test_keeps_istd_rows(self):
        """Step 2 marks ISTD rows but does not remove them."""
        df = _make_df(
            ["261.1273/5.00", "500.0000/3.00"],
            {"case_S1": [1000, 2000]},
        )
        marker = ISTDMarker()
        result = marker.process(df, istd_features={"261.1273/5.00"})
        assert result.success
        assert result.statistics.get("rows_removed", 0) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_df_fails_validation(self):
        df = pd.DataFrame()
        marker = ISTDMarker()
        result = marker.process(df)
        assert not result.success

    def test_single_data_row(self):
        df = _make_df(["100.0000/1.0"], {"case_S1": [10]})
        marker = ISTDMarker()
        result = marker.process(df, istd_features=set())
        assert result.success

    def test_no_slash_in_feature_id(self):
        """Mz/RT value without '/' should not crash _extract_mz_rt_arrays."""
        df = pd.DataFrame({
            "Mz/RT": ["Sample_Type", "no_slash_here", "also_no_slash"],
            "S1": ["case", 10, 20],
        })
        marker = ISTDMarker()
        result = marker.process(df)
        # Validation should reject (< 50% valid format)
        assert not result.success

    def test_cancellation(self):
        """Cancellation mid-process should return failure."""
        df = _make_df(
            [f"{100 + i:.4f}/1.0" for i in range(20)],
            {"case_S1": list(range(20))},
        )
        marker = ISTDMarker()
        # Cancel during progress callback (after process starts and calls reset)
        def cancel_on_progress(percent, msg):
            if percent >= 20:
                marker.cancel()
        marker.set_progress_callback(cancel_on_progress)
        result = marker.process(df)
        assert not result.success
        assert "cancelled" in result.message.lower()
