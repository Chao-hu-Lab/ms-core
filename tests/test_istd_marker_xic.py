from __future__ import annotations

from pathlib import Path

import pandas as pd

from ms_core.preprocessing.istd_marker import ISTDMarker


def _make_df(feature_ids: list[str], sample_values: dict[str, list]) -> pd.DataFrame:
    data = {"Mz/RT": ["Sample_Type"] + feature_ids}
    for column, values in sample_values.items():
        data[column] = [column.split("_")[0]] + values
    return pd.DataFrame(data)


def _write_xic_workbook(path: Path) -> None:
    targets = pd.DataFrame(
        [
            {
                "Label": "d3-5-hmdC",
                "Role": "ISTD",
                "ISTD Pair": None,
                "m/z": 261.127276,
                "RT min": 8.0,
                "RT max": 10.0,
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
    with pd.ExcelWriter(path) as writer:
        targets.to_excel(writer, sheet_name="Targets", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)


def test_istd_marker_uses_xic_targets_to_choose_best_matching_row(project_temp_dir):
    with project_temp_dir() as temp_dir:
        xic_path = Path(temp_dir) / "xic.xlsx"
        _write_xic_workbook(xic_path)
        df = _make_df(
            ["261.1273/8.970", "261.1280/8.980", "500.0000/3.000"],
            {
                "case_S1": [100, 300, 1000],
                "case_S2": [0, 400, 1100],
            },
        )

        result = ISTDMarker().process(df, xic_results_file=xic_path)

    assert result.success
    assert "261.1273/8.970" in result.metadata["istd_features"]
    assert "261.1280/8.980" not in result.metadata["istd_features"]
    assert result.statistics["duplicates_marked"] == 0
    assert result.metadata["duplicate_indices"] == []
    assert result.metadata["red_font_rows"]
    assert result.metadata["protected_rows"] == result.metadata["red_font_rows"]
    assert result.metadata["xic_target_count"] == 1


def test_istd_marker_fails_when_xic_targets_match_zero_feature_rows(project_temp_dir):
    with project_temp_dir() as temp_dir:
        xic_path = Path(temp_dir) / "xic.xlsx"
        _write_xic_workbook(xic_path)
        df = _make_df(
            ["500.0000/3.000"],
            {"case_S1": [1000], "case_S2": [1100]},
        )

        result = ISTDMarker().process(df, xic_results_file=xic_path)

    assert not result.success
    assert "No ISTD features matched XIC targets" in result.message


def test_istd_marker_uses_xic_rt_window_for_matching(project_temp_dir):
    with project_temp_dir() as temp_dir:
        xic_path = Path(temp_dir) / "xic.xlsx"
        _write_xic_workbook(xic_path)
        df = _make_df(
            ["261.1273/10.200"],
            {"case_S1": [1000], "case_S2": [1100]},
        )

        result = ISTDMarker().process(df, xic_results_file=xic_path)

    assert not result.success
    assert "No ISTD features matched XIC targets" in result.message


def test_istd_marker_requires_xic_or_explicit_istd_features() -> None:
    df = _make_df(
        ["261.1273/8.970"],
        {"case_S1": [1000], "case_S2": [1100]},
    )

    result = ISTDMarker().process(df)

    assert not result.success
    assert "XIC Extractor results workbook" in result.message


def test_istd_marker_rejects_legacy_manual_mz_argument() -> None:
    df = _make_df(
        ["261.1273/8.970"],
        {"case_S1": [1000], "case_S2": [1100]},
    )

    try:
        ISTDMarker().process(df, istd_mz_list=[261.1273])
    except TypeError as exc:
        assert "istd_mz_list" in str(exc)
    else:
        raise AssertionError("legacy istd_mz_list argument should not be accepted")
