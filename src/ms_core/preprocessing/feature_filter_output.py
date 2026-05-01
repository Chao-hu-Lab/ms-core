"""Output shaping for Step4 feature filtering."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ms_core.preprocessing.feature_filter_decisions import FeatureFilterDecisionResult


class FeatureFilterOutputBuilder:
    """Build filtered Step4 output frames from decision masks."""

    _KEEP_REASON_MASKS = (
        ("stable", "stable_keep"),
        ("mnar", "mnar_keep"),
        ("intensity_fc", "intensity_fc_keep"),
        ("ratio_rescue", "ratio_rescue_keep"),
        ("protected", "protected_mask"),
        ("unfiltered", "unfiltered_keep"),
    )
    _TAG_REASON_MASKS = (
        ("structural_absence", "tag_reason_structural"),
        ("low_overall_detection", "tag_reason_low_overall"),
    )

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
                deleted_features.append(
                    self._with_deleted_diagnostics(
                        df.iloc[i].copy(),
                        decision,
                        feature_pos=i - 1,
                    )
                )

        row_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(rows_to_keep)}
        stats["red_font_rows"] = sorted(
            row_mapping[idx] for idx in protected_rows if idx in row_mapping
        )

        result_df = df.iloc[rows_to_keep].reset_index(drop=True).copy()

        mnar_col: list[object] = ["is_Presence_Absence_Marker"]
        keep_reasons_col: list[object] = ["Feature_Filter_Keep_Reasons"]
        tag_reasons_col: list[object] = ["Imputation_Tag_Reasons"]
        for orig_row_idx in rows_to_keep[1:]:
            feature_pos = orig_row_idx - 1
            mnar_col.append(bool(decision.imputation_tag[feature_pos]))
            keep_reasons_col.append(self._compose_keep_reasons(decision, feature_pos))
            tag_reasons_col.append(self._compose_tag_reasons(decision, feature_pos))
        marker_idx = len(result_df.columns)
        result_df.insert(marker_idx, "is_Presence_Absence_Marker", mnar_col)
        result_df.insert(
            marker_idx + 1,
            "Feature_Filter_Keep_Reasons",
            keep_reasons_col,
        )
        result_df.insert(
            marker_idx + 2,
            "Imputation_Tag_Reasons",
            tag_reasons_col,
        )

        zeros_converted = self._convert_sample_zeros_to_nan(
            result_df,
            group_info,
        )
        if zeros_converted is not None:
            stats["zeros_converted_to_nan"] = zeros_converted

        return result_df, deleted_features, stats

    @classmethod
    def _compose_keep_reasons(
        cls,
        decision: FeatureFilterDecisionResult,
        feature_pos: int,
    ) -> str:
        tokens = [
            token
            for token, attr_name in cls._KEEP_REASON_MASKS
            if bool(getattr(decision, attr_name)[feature_pos])
        ]
        return "|".join(tokens)

    @classmethod
    def _compose_tag_reasons(
        cls,
        decision: FeatureFilterDecisionResult,
        feature_pos: int,
    ) -> str:
        tokens = [
            token
            for token, attr_name in cls._TAG_REASON_MASKS
            if bool(getattr(decision, attr_name)[feature_pos])
        ]
        return "|".join(tokens)

    @classmethod
    def _with_deleted_diagnostics(
        cls,
        row: pd.Series,
        decision: FeatureFilterDecisionResult,
        feature_pos: int,
    ) -> pd.Series:
        row["Feature_Filter_Delete_Reasons"] = cls._compose_delete_reasons(
            decision,
            feature_pos,
        )
        return row

    @staticmethod
    def _compose_delete_reasons(
        decision: FeatureFilterDecisionResult,
        feature_pos: int,
    ) -> str:
        tokens: list[str] = []
        if bool(decision.qc_zero[feature_pos]):
            tokens.append("qc_zero")
        elif bool(decision.qc_low[feature_pos]):
            tokens.append("qc_low")

        positive_keep_reason = bool(
            decision.stable_keep[feature_pos]
            or decision.mnar_keep[feature_pos]
            or decision.intensity_fc_keep[feature_pos]
            or decision.ratio_rescue_keep[feature_pos]
            or decision.protected_mask[feature_pos]
        )
        if not positive_keep_reason:
            tokens.append("no_keep_rule")

        return "|".join(tokens)

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
