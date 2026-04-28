from __future__ import annotations

from pathlib import Path

import pytest
import pandas as pd

from ms_core.preprocessing.xic_targets import (
    XICISTDTargetParseError,
    load_xic_istd_targets,
)


def _write_xic_workbook(
    path: Path,
    *,
    targets: pd.DataFrame | None = None,
    summary: pd.DataFrame | None = None,
) -> None:
    targets_df = targets if targets is not None else pd.DataFrame(
        [
            {
                "Label": "5-hmdC",
                "Role": "Analyte",
                "ISTD Pair": "d3-5-hmdC",
                "m/z": 258.1085,
                "RT min": 8.0,
                "RT max": 10.0,
                "ppm tol": 20,
            },
            {
                "Label": "d3-5-hmdC",
                "Role": "ISTD",
                "ISTD Pair": None,
                "m/z": 261.127276,
                "RT min": 8.0,
                "RT max": 10.0,
                "ppm tol": 20,
            },
            {
                "Label": "d3-5-medC",
                "Role": "ISTD",
                "ISTD Pair": None,
                "m/z": 245.132362,
                "RT min": 11.0,
                "RT max": 13.0,
                "ppm tol": 15,
            },
        ]
    )
    summary_df = summary if summary is not None else pd.DataFrame(
        [
            {
                "Target": "d3-5-hmdC",
                "Role": "ISTD",
                "Detected": 85,
                "Total": 85,
                "Detection %": "100%",
                "Mean RT": "8.9698",
            },
            {
                "Target": "d3-5-medC",
                "Role": "ISTD",
                "Detected": 84,
                "Total": 85,
                "Detection %": "99%",
                "Mean RT": "—",
            },
        ]
    )
    with pd.ExcelWriter(path) as writer:
        targets_df.to_excel(writer, sheet_name="Targets", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)


def test_load_xic_istd_targets_prefers_summary_mean_rt_and_filters_istd_rows(project_temp_dir):
    with project_temp_dir() as temp_dir:
        path = Path(temp_dir) / "xic.xlsx"
        _write_xic_workbook(path)

        targets, metadata = load_xic_istd_targets(path)

    assert [target.label for target in targets] == ["d3-5-hmdC", "d3-5-medC"]
    assert targets[0].mz == pytest.approx(261.127276)
    assert targets[0].rt == pytest.approx(8.9698)
    assert targets[0].ppm_tolerance == pytest.approx(20)
    assert targets[1].rt == pytest.approx(12.0)
    assert metadata["xic_target_count"] == 2
    assert metadata["xic_targets_using_summary_rt"] == ["d3-5-hmdC"]
    assert metadata["xic_targets_using_midpoint_rt"] == ["d3-5-medC"]


def test_load_xic_istd_targets_reports_missing_targets_sheet(project_temp_dir):
    with project_temp_dir() as temp_dir:
        path = Path(temp_dir) / "xic.xlsx"
        with pd.ExcelWriter(path) as writer:
            pd.DataFrame({"Target": ["d3-5-hmdC"]}).to_excel(
                writer,
                sheet_name="Summary",
                index=False,
            )

        with pytest.raises(XICISTDTargetParseError, match="Targets"):
            load_xic_istd_targets(path)


def test_load_xic_istd_targets_requires_ppm_tolerance(project_temp_dir):
    with project_temp_dir() as temp_dir:
        path = Path(temp_dir) / "xic.xlsx"
        targets = pd.DataFrame(
            [
                {
                    "Label": "d3-5-hmdC",
                    "Role": "ISTD",
                    "ISTD Pair": None,
                    "m/z": 261.127276,
                    "RT min": 8.0,
                    "RT max": 10.0,
                    "ppm tol": None,
                }
            ]
        )
        _write_xic_workbook(path, targets=targets)

        with pytest.raises(XICISTDTargetParseError, match="No usable ISTD targets"):
            load_xic_istd_targets(path)


def test_load_xic_istd_targets_allows_missing_rt_window_when_summary_has_mean_rt(project_temp_dir):
    with project_temp_dir() as temp_dir:
        path = Path(temp_dir) / "xic.xlsx"
        targets = pd.DataFrame(
            [
                {
                    "Label": "d3-5-hmdC",
                    "Role": "ISTD",
                    "ISTD Pair": None,
                    "m/z": 261.127276,
                    "ppm tol": 20,
                }
            ]
        )
        summary = pd.DataFrame(
            [
                {
                    "Target": "d3-5-hmdC",
                    "Role": "ISTD",
                    "Detected": 85,
                    "Total": 85,
                    "Detection %": "100%",
                    "Mean RT": "8.9698",
                }
            ]
        )
        _write_xic_workbook(path, targets=targets, summary=summary)

        targets_out, metadata = load_xic_istd_targets(path)

    assert len(targets_out) == 1
    assert targets_out[0].rt == pytest.approx(8.9698)
    assert targets_out[0].rt_min is None
    assert targets_out[0].rt_max is None
    assert metadata["xic_targets_using_summary_rt"] == ["d3-5-hmdC"]


def test_load_xic_istd_targets_closes_excel_workbook(monkeypatch, project_temp_dir):
    closed = {"value": False}

    class FakeWorkbook:
        sheet_names = ["Targets", "Summary"]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            closed["value"] = True

    def fake_excel_file(path):
        return FakeWorkbook()

    def fake_read_excel(workbook, *, sheet_name):
        if sheet_name == "Targets":
            return pd.DataFrame(
                [
                    {
                        "Label": "d3-5-hmdC",
                        "Role": "ISTD",
                        "m/z": 261.127276,
                        "RT min": 8.0,
                        "RT max": 10.0,
                        "ppm tol": 20,
                    }
                ]
            )
        return pd.DataFrame(
            [
                {
                    "Target": "d3-5-hmdC",
                    "Role": "ISTD",
                    "Mean RT": "8.9698",
                }
            ]
        )

    with project_temp_dir() as temp_dir:
        path = Path(temp_dir) / "xic.xlsx"
        path.write_text("placeholder", encoding="utf-8")
        monkeypatch.setattr("ms_core.preprocessing.xic_targets.pd.ExcelFile", fake_excel_file)
        monkeypatch.setattr("ms_core.preprocessing.xic_targets.pd.read_excel", fake_read_excel)

        targets_out, _ = load_xic_istd_targets(path)

    assert len(targets_out) == 1
    assert closed["value"] is True
