"""Decision gates for Step4 feature filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureFilterThresholds:
    background: float
    high_det: float
    low_det: float
    qc_ratio: float
    intensity_fc: float
    ratio_rescue: float = 2.0


@dataclass(frozen=True)
class FeatureFilterOptions:
    enable_background: bool
    enable_qc_ratio: bool
    enable_intensity_fc: bool
    enable_mnar: bool
    allow_single_group_stable: bool
    enable_ratio_rescue: bool = True


@dataclass(frozen=True)
class FeatureFilterDecisionResult:
    keep_mask: np.ndarray
    stable_keep: np.ndarray
    mnar_keep: np.ndarray
    intensity_fc_keep: np.ndarray
    ratio_rescue_keep: np.ndarray
    protected_mask: np.ndarray
    qc_zero: np.ndarray
    qc_low: np.ndarray
    qc_force_delete: np.ndarray
    stats: dict[str, Any]


class FeatureFilterDecisionTable:
    """Apply Step4 feature keep/delete gates without shaping output rows."""

    _RATIO_RESCUE_MIN_DETECTION: float = 0.10

    def decide(
        self,
        df: pd.DataFrame,
        group_info: dict[str, Any],
        ratio_cols: dict[str, str],
        thresholds: FeatureFilterThresholds,
        options: FeatureFilterOptions,
        protected_rows: set[int],
        numeric_block: dict[str, Any],
    ) -> FeatureFilterDecisionResult:
        stats: dict[str, Any] = {
            "kept_count": 0,
            "deleted_count": 0,
            "stable_kept": 0,
            "mnar_kept": 0,
            "intensity_fc_kept": 0,
            "ratio_rescue_kept": 0,
            "qc_zero_deleted": 0,
            "qc_low_deleted": 0,
            "protected_kept": 0,
            "unique_stable_kept": 0,
            "unique_mnar_kept": 0,
            "unique_intensity_fc_kept": 0,
            "unique_ratio_rescue_kept": 0,
        }

        group_names = list(group_info["groups"].keys())
        has_qc = group_info["has_qc"]
        qc_ratio_col = ratio_cols.get("QC")
        n_features = len(df) - 1

        ratio_matrix = []
        for group_name in group_names:
            ratio_col = ratio_cols[group_name]
            ratio_series = pd.to_numeric(df[ratio_col].iloc[1:], errors="coerce").fillna(0)
            ratio_matrix.append(ratio_series.to_numpy())

        if ratio_matrix:
            ratio_matrix = np.vstack(ratio_matrix).T
        else:
            ratio_matrix = np.zeros((n_features, 0))

        if has_qc and qc_ratio_col:
            qc_ratio = (
                pd.to_numeric(df[qc_ratio_col].iloc[1:], errors="coerce").fillna(0).to_numpy()
            )
        else:
            qc_ratio = np.ones(n_features)

        protected_mask = np.zeros(n_features, dtype=bool)
        for idx in protected_rows:
            if 0 < idx < len(df):
                protected_mask[idx - 1] = True

        if options.enable_qc_ratio:
            qc_zero = qc_ratio == 0
            qc_low = (
                (qc_ratio < thresholds.qc_ratio) & (qc_ratio > 0)
                if has_qc and thresholds.qc_ratio > 0
                else np.zeros(n_features, dtype=bool)
            )
        else:
            qc_zero = np.zeros(n_features, dtype=bool)
            qc_low = np.zeros(n_features, dtype=bool)

        if ratio_matrix.shape[1] > 0:
            mnar_keep = (
                (ratio_matrix >= thresholds.high_det).any(axis=1)
                & (ratio_matrix <= thresholds.low_det).any(axis=1)
                if (options.enable_mnar and ratio_matrix.shape[1] >= 2)
                else np.zeros(n_features, dtype=bool)
            )
            if options.enable_background:
                n_groups = ratio_matrix.shape[1]
                required_groups = (
                    1 if (options.allow_single_group_stable and n_groups == 1) else 2
                )
                stable_keep = (ratio_matrix >= thresholds.background).sum(axis=1) >= required_groups
            else:
                stable_keep = np.zeros(n_features, dtype=bool)
        else:
            mnar_keep = np.zeros(n_features, dtype=bool)
            stable_keep = np.zeros(n_features, dtype=bool)

        if options.enable_intensity_fc and ratio_matrix.shape[1] >= 2:
            block_values = numeric_block["values"]
            col_pos = numeric_block["col_pos"]
            intensity_means = []
            for group_name in group_names:
                col_indices = group_info["groups"][group_name]
                pos = [col_pos[c] for c in col_indices]
                group_block = block_values[:, pos]
                intensity_means.append(np.nanmean(group_block, axis=1))
            intensity_matrix = np.column_stack(intensity_means)

            safe_matrix = np.where(intensity_matrix > 0, intensity_matrix, np.nan)
            max_mean = np.nanmax(safe_matrix, axis=1)
            min_mean = np.nanmin(safe_matrix, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                fold_change = np.where(min_mean > 0, max_mean / min_mean, np.inf)
            fold_change = np.where(np.isnan(fold_change), 0.0, fold_change)
            intensity_fc_keep = fold_change >= thresholds.intensity_fc
        else:
            intensity_fc_keep = np.zeros(n_features, dtype=bool)

        if options.enable_ratio_rescue and ratio_matrix.shape[1] >= 2:
            detection_max = ratio_matrix.max(axis=1)
            detection_min = ratio_matrix.min(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                det_ratio = np.where(
                    detection_min > 0, detection_max / detection_min, 0.0
                )
            ratio_rescue_keep = (det_ratio >= thresholds.ratio_rescue) & (
                detection_min >= self._RATIO_RESCUE_MIN_DETECTION
            )
        else:
            ratio_rescue_keep = np.zeros(n_features, dtype=bool)

        positive_rules = []
        if options.enable_background:
            positive_rules.append(stable_keep)
        positive_rules.append(mnar_keep)
        if options.enable_intensity_fc:
            positive_rules.append(intensity_fc_keep)
        if options.enable_ratio_rescue:
            positive_rules.append(ratio_rescue_keep)

        if positive_rules:
            keep_mask = protected_mask | np.logical_or.reduce(positive_rules)
        else:
            keep_mask = np.ones(n_features, dtype=bool)

        qc_force_delete = (
            (qc_zero | qc_low) & ~protected_mask & ~mnar_keep & ~ratio_rescue_keep
        )
        keep_mask = np.where(qc_force_delete, False, keep_mask)

        non_protected = ~protected_mask
        effective = non_protected & ~qc_force_delete

        stats["protected_kept"] = int(protected_mask.sum())
        stats["stable_kept"] = int((stable_keep & non_protected).sum())
        stats["mnar_kept"] = int((mnar_keep & non_protected).sum())
        stats["intensity_fc_kept"] = int((intensity_fc_keep & non_protected).sum())
        stats["ratio_rescue_kept"] = int((ratio_rescue_keep & non_protected).sum())
        stats["unique_stable_kept"] = int(
            (
                stable_keep
                & ~mnar_keep
                & ~intensity_fc_keep
                & ~ratio_rescue_keep
                & effective
            ).sum()
        )
        stats["unique_mnar_kept"] = int(
            (
                mnar_keep
                & ~stable_keep
                & ~intensity_fc_keep
                & ~ratio_rescue_keep
                & effective
            ).sum()
        )
        stats["unique_intensity_fc_kept"] = int(
            (
                intensity_fc_keep
                & ~stable_keep
                & ~mnar_keep
                & ~ratio_rescue_keep
                & effective
            ).sum()
        )
        stats["unique_ratio_rescue_kept"] = int(
            (
                ratio_rescue_keep
                & ~stable_keep
                & ~mnar_keep
                & ~intensity_fc_keep
                & effective
            ).sum()
        )
        stats["qc_zero_deleted"] = int(
            (qc_zero & non_protected & ~mnar_keep & ~ratio_rescue_keep).sum()
        )
        stats["qc_low_deleted"] = int(
            (qc_low & non_protected & ~mnar_keep & ~ratio_rescue_keep).sum()
        )
        stats["kept_count"] = int(keep_mask.sum())
        stats["deleted_count"] = int((~keep_mask).sum())

        return FeatureFilterDecisionResult(
            keep_mask=keep_mask,
            stable_keep=stable_keep,
            mnar_keep=mnar_keep,
            intensity_fc_keep=intensity_fc_keep,
            ratio_rescue_keep=ratio_rescue_keep,
            protected_mask=protected_mask,
            qc_zero=qc_zero,
            qc_low=qc_low,
            qc_force_delete=qc_force_delete,
            stats=stats,
        )
