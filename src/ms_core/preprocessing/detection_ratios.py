"""Detection ratio calculation for Step4 feature filtering."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ms_core.preprocessing.settings import FeatureFilterConfig


class DetectionRatioCalculator:
    """Append per-group and QC detection ratio columns."""

    def __init__(self, config: FeatureFilterConfig | None = None) -> None:
        self.config = config or FeatureFilterConfig()

    def calculate(
        self,
        df: pd.DataFrame,
        group_info: dict[str, Any],
    ) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any]]:
        """Return DataFrame with ratio columns, ratio column names, and numeric block."""
        ratio_cols: dict[str, str] = {}
        signal_threshold = self.config.signal_threshold

        all_cols_set: set[int] = set()
        for cols in group_info["groups"].values():
            all_cols_set.update(cols)
        all_cols_set.update(group_info.get("qc_cols", []))
        all_cols = sorted(all_cols_set)
        col_pos = {col_idx: pos for pos, col_idx in enumerate(all_cols)}
        if all_cols:
            block_all = df.iloc[1:, all_cols].apply(pd.to_numeric, errors="coerce")
            block_all_values = block_all.to_numpy()
        else:
            block_all_values = np.zeros((len(df) - 1, 0))

        for group_name, col_indices in group_info["groups"].items():
            ratio_col = f"{group_name}_ratio"
            ratio_cols[group_name] = ratio_col

            if not col_indices:
                df[ratio_col] = ["na"] + [0] * (len(df) - 1)
                continue

            pos = [col_pos[c] for c in col_indices]
            block = block_all_values[:, pos]
            signal_count = (block >= signal_threshold).sum(axis=1)
            total_count = len(pos)
            ratios = signal_count / total_count if total_count > 0 else np.zeros(len(signal_count))
            df[ratio_col] = ["na"] + ratios.tolist()

        if group_info["has_qc"]:
            qc_ratio_col = "QC_ratio"
            ratio_cols["QC"] = qc_ratio_col

            qc_cols = group_info["qc_cols"]
            if qc_cols:
                pos = [col_pos[c] for c in qc_cols]
                block = block_all_values[:, pos]
                signal_count = (block >= signal_threshold).sum(axis=1)
                total_count = len(pos)
                qc_ratios = (
                    signal_count / total_count if total_count > 0 else np.zeros(len(signal_count))
                )
                df[qc_ratio_col] = ["na"] + qc_ratios.tolist()
            else:
                df[qc_ratio_col] = ["na"] + [0] * (len(df) - 1)

        numeric_block = {
            "values": block_all_values,
            "all_cols": all_cols,
            "col_pos": col_pos,
        }
        return df, ratio_cols, numeric_block
