"""
ISTD Marker Module - Step 2 of the preprocessing pipeline.

This module handles Internal Standard (ISTD) row marking:
- Sort data by m/z values
- Mark ISTD features with protected-row metadata

Based on: FindSTDs_mzRT_Jia_Simplified.bas
"""

from typing import Optional, Dict, Any, Set, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

from ms_core.preprocessing.base import BaseProcessor, ProcessingResult
from ms_core.preprocessing.settings import ISTDConfig
from ms_core.preprocessing.xic_targets import (
    XICISTDTarget,
    load_xic_istd_targets,
)
from ms_core.utils.file_handler import parse_mz_rt_string


class ISTDMarker(BaseProcessor):
    """
    Marks Internal Standards (ISTD).

    This processor:
    1. Renames the first sheet to 'RawIntensity'
    2. Sorts data by m/z values (ascending)
    3. Infers ISTD feature rows from XIC Extractor targets
    4. Marks ISTD features with red font/protected-row metadata
    """

    def __init__(self, config: Optional[ISTDConfig] = None):
        """
        Initialize the ISTD Marker.

        Args:
            config: Configuration options for ISTD marking
        """
        super().__init__("ISTD Marker")
        self.config = config or ISTDConfig()

    def validate_input(self, df: pd.DataFrame) -> tuple:
        """
        Validate input data for ISTD marking.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (is_valid, error_message)
        """
        if df is None or df.empty:
            return False, "Input data is empty"

        if len(df) < 2:
            return False, "Data must have at least 2 rows (Sample_Type + data)"

        # Check if first column contains m/z/RT format
        first_col = df.columns[0]
        sample_values = df[first_col].iloc[1:11]
        valid_format_count = sum(1 for v in sample_values if self._is_valid_mz_rt(v))

        if valid_format_count < len(sample_values) * 0.5:
            return False, "First column should contain m/z/RT format (e.g., '123.456/1.23')"

        return True, ""

    def process(
        self,
        df: pd.DataFrame,
        istd_features: Optional[Set[str]] = None,
        xic_results_file: Optional[Path] = None,
    ) -> ProcessingResult:
        """
        Process data for ISTD marking.

        Args:
            df: Input DataFrame (data starts from row index 1)
            istd_features: Explicit feature IDs to mark as ISTD
            xic_results_file: XIC Extractor result workbook used to infer ISTD rows

        Returns:
            ProcessingResult with processed data
        """
        self.reset()

        # Validate input
        is_valid, error_msg = self.validate_input(df)
        if not is_valid:
            return ProcessingResult(
                success=False,
                errors=[error_msg],
                message=f"Validation failed: {error_msg}",
            )

        self.update_progress(10, "Starting ISTD marking...")

        try:
            # Create a copy to avoid modifying original
            result_df = df.copy()

            # Step 1: Standardize header structure
            self.update_progress(20, "Standardizing headers...")
            result_df = self._standardize_headers(result_df)

            if self._cancelled:
                return ProcessingResult(success=False, message="Processing cancelled")

            # Step 2: Sort by m/z values
            self.update_progress(40, "Sorting by m/z values...")
            result_df, sort_stats = self._sort_by_mz(result_df)

            if self._cancelled:
                return ProcessingResult(success=False, message="Processing cancelled")

            # Step 3: Determine ISTD features
            self.update_progress(60, "Finding ISTD feature rows...")
            has_explicit_istd_features = istd_features is not None
            istd_features_set: Set[str] = set(istd_features or set())
            metadata: Dict[str, Any] = {}

            if xic_results_file:
                targets, xic_meta = load_xic_istd_targets(xic_results_file)
                inferred, infer_meta = self._infer_istd_from_xic_targets(
                    result_df,
                    targets,
                )
                istd_features_set = inferred
                metadata.update(xic_meta)
                metadata.update(infer_meta)
                if not istd_features_set:
                    message = (
                        "No ISTD features matched XIC targets. "
                        "Check that the XIC results file belongs to this feature matrix."
                    )
                    return ProcessingResult(
                        success=False,
                        errors=[message],
                        message=message,
                        metadata=metadata,
                    )
            elif not has_explicit_istd_features:
                message = (
                    "Step2 now requires an XIC Extractor results workbook "
                    "or explicit ISTD features."
                )
                return ProcessingResult(
                    success=False,
                    errors=[message],
                    message=message,
                    metadata=metadata,
                )

            if self._cancelled:
                return ProcessingResult(success=False, message="Processing cancelled")

            self.update_progress(80, "Marking ISTD features...")

            self.update_progress(100, "ISTD marking complete")

            istd_rows = {
                idx
                for idx in range(1, len(result_df))
                if str(result_df.iat[idx, 0]) in istd_features_set
            }
            stats = {
                **sort_stats,
                "istd_marked": len(istd_rows),
                "duplicates_marked": 0,
                "rows_removed": 0,
            }
            red_rows = sorted(istd_rows)

            return ProcessingResult(
                success=True,
                data=result_df,
                message=f"ISTD marking completed. Marked {len(istd_rows)} ISTD features.",
                statistics=stats,
                metadata={
                    "duplicate_indices": [],
                    "istd_features": list(istd_features_set),
                    "istd_rows": sorted(istd_rows),
                    "protected_rows": sorted(istd_rows),
                    "red_font_rows": red_rows,
                    **metadata,
                },
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                errors=[str(e)],
                message=f"Error during ISTD marking: {str(e)}",
            )

    def _standardize_headers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize the header structure."""
        # Drop tolerance column if it exists (not needed after Step 2)
        columns = list(df.columns)
        tolerance_cols = [c for c in columns if "tolerance" in str(c).lower()]
        if tolerance_cols:
            df = df.drop(columns=tolerance_cols)
            columns = list(df.columns)

        df.columns = columns

        # Ensure Sample_Type row values
        if len(df) > 0:
            df.iat[0, 0] = self.config.sample_type_col
            # Only set tolerance column to 'na' if it exists as second column
            if len(df.columns) > 1 and "tolerance" in str(df.columns[1]).lower():
                df.iat[0, 1] = "na"

        return df

    def _sort_by_mz(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Sort data rows by m/z values."""
        stats = {"original_order_changed": False}

        # Data rows start from index 1 (after Sample_Type row)
        if len(df) <= 1:
            return df, stats

        # Extract header rows
        header_rows = df.iloc[:1].copy()
        data_rows = df.iloc[1:].copy()

        # Extract m/z values for sorting (vectorized)
        mz_arr, _ = self._extract_mz_rt_arrays(data_rows)
        mz_arr = np.where(np.isnan(mz_arr), np.inf, mz_arr)
        data_rows["_sort_mz"] = mz_arr

        # Sort by m/z
        data_rows_sorted = data_rows.sort_values('_sort_mz', ascending=True)
        data_rows_sorted = data_rows_sorted.drop('_sort_mz', axis=1)

        # Check if order changed
        if not data_rows_sorted.index.equals(data_rows.index):
            stats["original_order_changed"] = True

        # Reconstruct DataFrame
        data_rows_sorted = data_rows_sorted.reset_index(drop=True)
        result_df = pd.concat([header_rows.reset_index(drop=True), data_rows_sorted], ignore_index=True)

        stats["rows_sorted"] = len(data_rows)

        return result_df, stats

    @staticmethod
    def _is_valid_mz_rt(value) -> bool:
        """Check if value is in valid m/z/RT format."""
        mz, rt = parse_mz_rt_string(str(value))
        return mz is not None and rt is not None

    def _infer_istd_from_xic_targets(
        self,
        df: pd.DataFrame,
        targets: list[XICISTDTarget],
    ) -> tuple[Set[str], Dict[str, Any]]:
        """Infer the best feature row for each XIC-derived ISTD target."""
        if len(df) <= 1 or not targets:
            return set(), {
                "xic_istd_candidates": {},
                "xic_istd_chosen_rows": {},
            }

        mz_arr, rt_arr = self._extract_mz_rt_arrays(df)

        candidates: dict[str, list[tuple[int, float, float]]] = {}
        for row_idx in range(1, len(df)):
            mz = mz_arr[row_idx]
            rt = rt_arr[row_idx]
            if np.isnan(mz) or np.isnan(rt):
                continue

            for target in targets:
                ppm_diff = (
                    abs((mz - target.mz) / target.mz * 1_000_000)
                    if target.mz
                    else np.inf
                )
                rt_diff = abs(rt - target.rt)
                if ppm_diff > target.ppm_tolerance:
                    continue
                if target.rt_min is not None and target.rt_max is not None:
                    rt_lower = min(target.rt_min, target.rt_max)
                    rt_upper = max(target.rt_min, target.rt_max)
                    if not rt_lower <= rt <= rt_upper:
                        continue

                candidates.setdefault(target.label, []).append((row_idx, rt_diff, ppm_diff))

        istd_features: Set[str] = set()
        chosen_rows: dict[str, int] = {}

        for target in targets:
            rows = candidates.get(target.label, [])
            if not rows:
                continue
            rows_sorted = sorted(rows, key=lambda x: (x[1], x[2], x[0]))
            best_row = rows_sorted[0][0]
            chosen_rows[target.label] = best_row
            istd_features.add(str(df.iat[best_row, 0]))

        return istd_features, {
            "xic_istd_candidates": {
                label: [candidate[0] for candidate in rows]
                for label, rows in candidates.items()
            },
            "xic_istd_chosen_rows": chosen_rows,
        }

    def _extract_mz_rt_arrays(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized extraction of m/z and RT arrays from Mz/RT column."""
        col = df.iloc[:, 0].astype(str)
        parts = col.str.split("/", n=1, expand=True)
        mz = pd.to_numeric(parts[0], errors="coerce").to_numpy()
        if parts.shape[1] >= 2:
            rt = pd.to_numeric(parts[1], errors="coerce").to_numpy()
        else:
            rt = np.full(len(df), np.nan)
        return mz, rt
