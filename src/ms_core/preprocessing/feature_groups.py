"""Sample group detection for Step4 feature filtering."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ms_core.preprocessing.settings import FeatureFilterConfig
from ms_core.utils.validators import detect_fixed_columns


class FeatureGroupDetector:
    """Detect analysis groups and QC columns from the Sample_Type row."""

    def __init__(self, config: FeatureFilterConfig | None = None) -> None:
        self.config = config or FeatureFilterConfig()

    def detect(self, df: pd.DataFrame) -> dict[str, Any]:
        """Return Step4 group metadata from the first row of *df*."""
        info: dict[str, Any] = {
            "groups": {},
            "qc_cols": [],
            "excluded_cols": [],
            "unknown_types": set(),
            "has_qc": False,
        }
        excluded_types = {str(sample_type).lower() for sample_type in self.config.excluded_types}

        fixed_cols, start_idx = detect_fixed_columns(df)
        if not fixed_cols:
            start_idx = 1

        for col_idx in range(start_idx, len(df.columns)):
            sample_type = str(df.iat[0, col_idx]).lower().strip()
            if sample_type in {"", "nan", "na", "none"}:
                continue

            if sample_type == "qc":
                info["qc_cols"].append(col_idx)
                info["has_qc"] = True
            elif sample_type in excluded_types:
                info["excluded_cols"].append(col_idx)
            else:
                info["groups"].setdefault(sample_type, []).append(col_idx)

        return info
