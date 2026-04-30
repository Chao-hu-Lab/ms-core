"""Output shaping for Step4 feature filtering."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ms_core.preprocessing.feature_filter_decisions import FeatureFilterDecisionResult


class FeatureFilterOutputBuilder:
    """Build filtered Step4 output frames from decision masks."""

    def build(
        self,
        df: pd.DataFrame,
        group_info: dict[str, Any],
        decision: FeatureFilterDecisionResult,
        protected_rows: set[int],
    ) -> tuple[pd.DataFrame, list[pd.Series], dict[str, Any]]:
        """Return filtered DataFrame, deleted feature rows, and output stats."""
        stats = dict(decision.stats)
        deleted_features: list[pd.Series] = []
        rows_to_keep = [0]

        for i, keep in enumerate(decision.keep_mask, start=1):
            if keep:
                rows_to_keep.append(i)
            else:
                deleted_features.append(df.iloc[i].copy())

        row_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(rows_to_keep)}
        stats["red_font_rows"] = sorted(
            row_mapping[idx] for idx in protected_rows if idx in row_mapping
        )

        result_df = df.iloc[rows_to_keep].reset_index(drop=True).copy()

        mnar_col: list[object] = ["is_Presence_Absence_Marker"]
        for orig_row_idx in rows_to_keep[1:]:
            mnar_col.append(
                bool(
                    decision.mnar_keep[orig_row_idx - 1]
                    or decision.ratio_rescue_keep[orig_row_idx - 1]
                )
            )
        result_df.insert(len(result_df.columns), "is_Presence_Absence_Marker", mnar_col)

        zeros_converted = self._convert_sample_zeros_to_nan(
            result_df,
            group_info,
        )
        if zeros_converted is not None:
            stats["zeros_converted_to_nan"] = zeros_converted

        return result_df, deleted_features, stats

    @staticmethod
    def _convert_sample_zeros_to_nan(
        result_df: pd.DataFrame,
        group_info: dict[str, Any],
    ) -> int | None:
        all_data_cols: list[int] = []
        for cols in group_info["groups"].values():
            all_data_cols.extend(cols)
        all_data_cols.extend(group_info.get("qc_cols", []))
        if not all_data_cols:
            return None

        zeros_converted_total = 0
        for col_idx in all_data_cols:
            col_name = result_df.columns[col_idx]
            series = pd.to_numeric(result_df[col_name].iloc[1:], errors="coerce")
            zeros_converted = int((series == 0).sum())
            series = series.replace(0, np.nan)
            result_df[col_name] = [result_df.iat[0, col_idx]] + series.tolist()
            zeros_converted_total += zeros_converted

        return zeros_converted_total
