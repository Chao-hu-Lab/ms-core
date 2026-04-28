"""
Feature Filter Module - Step 4 of the preprocessing pipeline.

This module handles feature filtering:
- Dynamic sample type detection
- Ratio calculation for each group
- Multi-criteria feature filtering

Based on: Feature_barrier_V3.bas
"""

import warnings
from typing import Optional, Dict, Any, List, Set, Tuple
import pandas as pd
import numpy as np

from ms_core.preprocessing.base import BaseProcessor, ProcessingResult
from ms_core.preprocessing.detection_ratios import DetectionRatioCalculator
from ms_core.preprocessing.feature_filter_decisions import (
    FeatureFilterDecisionTable,
    FeatureFilterOptions,
    FeatureFilterThresholds,
)
from ms_core.preprocessing.feature_filter_output import FeatureFilterOutputBuilder
from ms_core.preprocessing.feature_groups import FeatureGroupDetector
from ms_core.preprocessing.settings import FeatureFilterConfig


class FeatureFilter(BaseProcessor):
    """
    Filters features based on detection ratio and intensity criteria.

    This processor:
    1. Automatically detects sample types from row 2
    2. Calculates signal ratio for each group
    3. Filters features based on multiple criteria:
       - Stable: >=2 groups with ratio >= background threshold
       - Different: Any two groups with ratio difference >= diff threshold
       - Intensity: Any two groups with mean intensity fold-change >= threshold
    4. Removes features with QC_ratio = 0 or below threshold
    """

    _SMALL_N_THRESHOLD: int = 10

    def __init__(self, config: Optional[FeatureFilterConfig] = None):
        """
        Initialize the Feature Filter.

        Args:
            config: Configuration options for feature filtering
        """
        super().__init__("Feature Filter")
        self.config = config or FeatureFilterConfig()

    def validate_input(self, df: pd.DataFrame) -> tuple:
        """
        Validate input data for feature filtering.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (is_valid, error_message)
        """
        if df is None or df.empty:
            return False, "Input data is empty"

        if len(df) < 2:
            return False, "Data must have at least 2 rows (Sample_Type + data)"

        if len(df.columns) < 3:
            return False, "Data must have at least 3 columns (feature + samples)"

        return True, ""

    def process(
        self,
        df: pd.DataFrame,
        background_threshold: Optional[float] = None,
        high_det_thresh: Optional[float] = None,
        low_det_thresh: Optional[float] = None,
        qc_ratio_threshold: Optional[float] = None,
        intensity_fc_threshold: Optional[float] = None,
        enable_background_threshold: bool = True,
        enable_qc_ratio_threshold: bool = True,
        enable_intensity_fc_threshold: bool = False,
        enable_mnar_gate: bool = True,
        allow_single_group_stable: bool = False,
        protected_rows: Optional[Set[int]] = None,
        **kwargs,
    ) -> ProcessingResult:
        """
        Process data for feature filtering and missing value imputation.

        Args:
            df: Input DataFrame
            background_threshold: Threshold for stable features (0-1)
            high_det_thresh: MNAR high detection rate threshold (0-1, default 0.8)
            low_det_thresh: MNAR low detection rate threshold (0-1, default 0.2)
            qc_ratio_threshold: Minimum QC_ratio to keep a feature (0-1)
            intensity_fc_threshold: Minimum fold-change of group mean intensities (>=1)
            enable_background_threshold: Whether to apply stable feature rule
            enable_qc_ratio_threshold: Whether to apply QC-based deletion rules
            enable_intensity_fc_threshold: Whether to apply intensity fold-change rule
            enable_mnar_gate: Whether to apply the MNAR 80/20 presence/absence rule
            allow_single_group_stable: When True and only 1 analysis group exists,
                degrade the stable gate to require only that single group to meet
                the background threshold (instead of the usual >= 2 groups).
                Only takes effect when enable_background_threshold is also True.
            protected_rows: Set of row indices (red font) to protect from removal
            **kwargs: Additional parameters

        Returns:
            ProcessingResult with filtered data
        """
        self.reset()

        # Deprecation guard for removed parameters
        _REMOVED = {
            "skew_threshold",
            "enable_skew_threshold",
            "diff_threshold",
            "enable_diff_threshold",
        }
        for removed_key in _REMOVED & kwargs.keys():
            warnings.warn(
                f"Parameter '{removed_key}' was removed in the gate logic refactor. "
                "It will be silently ignored. Use high_det_thresh/low_det_thresh instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Use config defaults if not specified
        bg_thresh = (
            background_threshold
            if background_threshold is not None
            else self.config.default_background_threshold
        )
        high_thresh = (
            high_det_thresh if high_det_thresh is not None else self.config.default_high_det_thresh
        )
        low_thresh = (
            low_det_thresh if low_det_thresh is not None else self.config.default_low_det_thresh
        )
        qc_ratio_thresh = (
            qc_ratio_threshold
            if qc_ratio_threshold is not None
            else self.config.default_qc_ratio_threshold
        )
        intensity_fc_thresh = (
            intensity_fc_threshold
            if intensity_fc_threshold is not None
            else self.config.default_intensity_fc_threshold
        )

        # Validate input
        is_valid, error_msg = self.validate_input(df)
        if not is_valid:
            return ProcessingResult(
                success=False,
                errors=[error_msg],
                message=f"Validation failed: {error_msg}",
            )

        self.update_progress(5, "Starting feature filtering...")

        try:
            # Create a copy
            result_df = df.copy()
            deleted_features = []

            # Step 1: Detect sample types
            self.update_progress(10, "Detecting sample types...")
            group_info = self._detect_sample_types(result_df)

            if self._cancelled:
                return ProcessingResult(success=False, message="Processing cancelled")

            # Step 2: Calculate ratios for each group
            self.update_progress(25, "Calculating group ratios...")
            result_df, ratio_cols, numeric_block = self._calculate_ratios(result_df, group_info)

            if self._cancelled:
                return ProcessingResult(success=False, message="Processing cancelled")

            # Step 3: Filter features
            self.update_progress(50, "Filtering features...")
            result_df, deleted_features, filter_stats = self._filter_features(
                result_df,
                group_info,
                ratio_cols,
                bg_thresh,
                high_thresh,
                low_thresh,
                qc_ratio_thresh,
                intensity_fc_thresh,
                enable_background_threshold,
                enable_qc_ratio_threshold,
                enable_intensity_fc_threshold,
                enable_mnar_gate,
                allow_single_group_stable,
                protected_rows or set(),
                numeric_block,
            )

            # Step 4: zero-to-NaN conversion is handled by FeatureFilterOutputBuilder.
            self.update_progress(85, "Converting zeros to NaN...")
            self.update_progress(100, "Feature filtering complete")

            # Compile statistics
            stats = {
                **filter_stats,
                "final_features": len(result_df) - 1,
                "groups_detected": len(group_info["groups"]),
                "has_qc": group_info["has_qc"],
                "qc_count": len(group_info.get("qc_cols", [])),
                "group_counts": {gname: len(cols) for gname, cols in group_info["groups"].items()},
            }

            return ProcessingResult(
                success=True,
                data=result_df,
                message=f"Feature filtering completed. Kept {filter_stats.get('kept_count', 0)}, "
                f"removed {filter_stats.get('deleted_count', 0)} features.",
                statistics=stats,
                metadata={
                    "group_info": group_info,
                    "ratio_columns": ratio_cols,
                    "thresholds": {
                        "background": bg_thresh,
                        "high_det": high_thresh,
                        "low_det": low_thresh,
                        "qc_ratio": qc_ratio_thresh,
                        "intensity_fc": intensity_fc_thresh,
                    },
                    "enabled_thresholds": {
                        "background": bool(enable_background_threshold),
                        "qc_ratio": bool(enable_qc_ratio_threshold),
                        "intensity_fc": bool(enable_intensity_fc_threshold),
                        "mnar_gate": bool(enable_mnar_gate),
                        "single_group_stable": bool(allow_single_group_stable),
                    },
                    "deleted_features": deleted_features,
                    "red_font_rows": filter_stats.get("red_font_rows", []),
                    "protected_rows": filter_stats.get("red_font_rows", []),
                },
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                errors=[str(e)],
                message=f"Error during feature filtering: {str(e)}",
            )

    def _detect_sample_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect sample types from row index 1 (Sample_Type row).

        Returns dict with group information.
        """
        return FeatureGroupDetector(self.config).detect(df)

    def count_analysis_groups(self, df: pd.DataFrame) -> int:
        """Count the number of non-QC analysis groups in df.

        Uses the same group-detection logic as the internal pipeline.
        Safe to call from external code without depending on private API.
        """
        group_info = self._detect_sample_types(df)
        return len(group_info["groups"])

    def _calculate_ratios(
        self,
        df: pd.DataFrame,
        group_info: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, Any]]:
        """
        Calculate signal ratio for each group.

        Returns:
            - DataFrame with ratio columns appended
            - dict mapping group name to ratio column name
            - numeric_block dict with 'values' (numpy array), 'all_cols' (sorted col indices),
              'col_pos' (col_idx -> position mapping) for reuse in _filter_features
        """
        return DetectionRatioCalculator(self.config).calculate(df, group_info)

    def _filter_features(
        self,
        df: pd.DataFrame,
        group_info: Dict[str, Any],
        ratio_cols: Dict[str, str],
        bg_threshold: float,
        high_det_thresh: float,
        low_det_thresh: float,
        qc_ratio_threshold: float,
        intensity_fc_threshold: float,
        enable_background_threshold: bool,
        enable_qc_ratio_threshold: bool,
        enable_intensity_fc_threshold: bool,
        enable_mnar_gate: bool,
        allow_single_group_stable: bool,
        protected_rows: Set[int],
        numeric_block: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, List[pd.Series], Dict[str, Any]]:
        """
        Filter features based on ratio and intensity criteria.

        Returns filtered DataFrame, deleted rows, and statistics.
        """
        thresholds = FeatureFilterThresholds(
            background=bg_threshold,
            high_det=high_det_thresh,
            low_det=low_det_thresh,
            qc_ratio=qc_ratio_threshold,
            intensity_fc=intensity_fc_threshold,
        )
        options = FeatureFilterOptions(
            enable_background=enable_background_threshold,
            enable_qc_ratio=enable_qc_ratio_threshold,
            enable_intensity_fc=enable_intensity_fc_threshold,
            enable_mnar=enable_mnar_gate,
            allow_single_group_stable=allow_single_group_stable,
        )
        decision = FeatureFilterDecisionTable().decide(
            df,
            group_info,
            ratio_cols,
            thresholds,
            options,
            protected_rows,
            numeric_block,
        )
        return FeatureFilterOutputBuilder().build(df, group_info, decision, protected_rows)

    @staticmethod
    def _wilson_lower_vec(p: np.ndarray, n: int, z: float = 1.96) -> np.ndarray:
        """Return the 95% Wilson CI lower bound for each proportion in *p*.

        Args:
            p: Array of observed proportions in [0, 1].
            n: Sample size (integer).  When 0, returns an all-zero array.
            z: Z-score for the desired confidence level (default 1.96 → 95%).

        Returns:
            Array of lower-bound proportions, clipped to [0, 1].
        """
        return FeatureFilterDecisionTable.wilson_lower_vec(p, n, z)

    def _get_max_ratio_diff(self, ratios: List[float]) -> float:
        """Calculate maximum difference between any two ratios."""
        if not ratios:
            return 0.0
        return float(max(ratios) - min(ratios))

    def get_group_summary(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Get a summary of detected groups and their statistics.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with group summary information
        """
        group_info = self._detect_sample_types(df)

        summary = {
            "groups": {},
            "qc_count": len(group_info["qc_cols"]),
            "has_qc": group_info["has_qc"],
            "excluded_count": len(group_info["excluded_cols"]),
            "unknown_types": list(group_info["unknown_types"]),
        }

        for group_name, col_indices in group_info["groups"].items():
            summary["groups"][group_name] = {
                "sample_count": len(col_indices),
                "columns": [df.columns[i] for i in col_indices],
            }

        return summary
