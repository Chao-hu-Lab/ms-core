"""XIC Extractor target parsing for Step 2 ISTD marking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class XICISTDTarget:
    """Normalized ISTD target definition extracted from an XIC workbook."""

    label: str
    mz: float
    rt: float
    rt_min: float | None
    rt_max: float | None
    ppm_tolerance: float
    source: str = "xic_extractor"
    detected: int | None = None
    total: int | None = None
    detection_percent: str | None = None
    rt_source: str = "midpoint"


class XICISTDTargetParseError(ValueError):
    """Raised when an XIC Extractor workbook cannot provide usable ISTD targets."""


def _require_columns(df: pd.DataFrame, required: set[str], sheet_name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise XICISTDTargetParseError(
            f"XIC workbook sheet '{sheet_name}' missing required columns: {', '.join(missing)}"
        )


def _numeric_or_none(value: object) -> float | None:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return float(parsed)


def _int_or_none(value: object) -> int | None:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return int(parsed)


def _is_istd_role(value: object) -> bool:
    return str(value).strip().lower() == "istd"


def _load_summary_by_target(workbook: pd.ExcelFile) -> dict[str, dict[str, object]]:
    if "Summary" not in workbook.sheet_names:
        return {}

    summary_df = pd.read_excel(workbook, sheet_name="Summary")
    required = {"Target", "Role"}
    if not required.issubset(set(summary_df.columns)):
        return {}

    rows: dict[str, dict[str, object]] = {}
    for _, row in summary_df.iterrows():
        if not _is_istd_role(row.get("Role")):
            continue
        label = str(row.get("Target", "")).strip()
        if label:
            rows[label] = row.to_dict()
    return rows


def load_xic_istd_targets(path: str | Path) -> tuple[list[XICISTDTarget], dict[str, Any]]:
    """Load normalized ISTD targets from an XIC Extractor workbook."""
    workbook_path = Path(path)
    if not workbook_path.exists():
        raise XICISTDTargetParseError(f"XIC workbook not found: {workbook_path}")

    targets: list[XICISTDTarget] = []
    skipped: list[dict[str, str]] = []
    using_summary_rt: list[str] = []
    using_midpoint_rt: list[str] = []

    with pd.ExcelFile(workbook_path) as workbook:
        if "Targets" not in workbook.sheet_names:
            raise XICISTDTargetParseError("XIC workbook missing required sheet: Targets")

        targets_df = pd.read_excel(workbook, sheet_name="Targets")
        _require_columns(
            targets_df,
            {"Label", "Role", "m/z", "ppm tol"},
            "Targets",
        )
        summary_by_target = _load_summary_by_target(workbook)

        for _, row in targets_df.iterrows():
            if not _is_istd_role(row.get("Role")):
                continue

            label = str(row.get("Label", "")).strip()
            if not label:
                skipped.append({"label": "", "reason": "missing label"})
                continue

            mz = _numeric_or_none(row.get("m/z"))
            ppm_tolerance = _numeric_or_none(row.get("ppm tol"))
            rt_min = _numeric_or_none(row.get("RT min"))
            rt_max = _numeric_or_none(row.get("RT max"))
            if mz is None:
                skipped.append({"label": label, "reason": "missing or non-numeric m/z"})
                continue
            if ppm_tolerance is None:
                skipped.append({"label": label, "reason": "missing or non-numeric ppm tol"})
                continue

            summary = summary_by_target.get(label, {})
            mean_rt = _numeric_or_none(summary.get("Mean RT"))
            if mean_rt is not None:
                rt = mean_rt
                rt_source = "summary_mean_rt"
                using_summary_rt.append(label)
            elif rt_min is not None and rt_max is not None:
                rt = (rt_min + rt_max) / 2
                rt_source = "target_midpoint"
                using_midpoint_rt.append(label)
            else:
                skipped.append({"label": label, "reason": "missing usable RT"})
                continue

            targets.append(
                XICISTDTarget(
                    label=label,
                    mz=mz,
                    rt=rt,
                    rt_min=rt_min,
                    rt_max=rt_max,
                    ppm_tolerance=ppm_tolerance,
                    detected=_int_or_none(summary.get("Detected")),
                    total=_int_or_none(summary.get("Total")),
                    detection_percent=(
                        str(summary["Detection %"]).strip()
                        if "Detection %" in summary and pd.notna(summary["Detection %"])
                        else None
                    ),
                    rt_source=rt_source,
                )
            )

    metadata: dict[str, Any] = {
        "xic_source_path": str(workbook_path),
        "xic_target_count": len(targets),
        "xic_skipped_targets": skipped,
        "xic_targets_using_summary_rt": using_summary_rt,
        "xic_targets_using_midpoint_rt": using_midpoint_rt,
    }
    if not targets:
        raise XICISTDTargetParseError("No usable ISTD targets found in XIC workbook")
    return targets, metadata
