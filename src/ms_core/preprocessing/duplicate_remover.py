"""
Duplicate Remover Module - Step 3 of the preprocessing pipeline.

This module handles intelligent duplicate signal removal:
- Automatic column detection (RT, m/z, intensity)
- Tolerance-based deduplication
- Preserve highest intensity signals
- Red font protection

Based on: ms-data-processor (https://github.com/bosschen0429/ms-data-processor)
"""

from typing import Optional, Dict, Any, List, Set, Tuple
import pandas as pd
import numpy as np

from ms_core.preprocessing.base import BaseProcessor, ProcessingResult
from ms_core.preprocessing.degeneracy_annotation import DegeneracyAnnotator
from ms_core.preprocessing.duplicate_intensity_merge import DuplicateIntensityMerger, MergeMode
from ms_core.preprocessing.settings import DuplicateRemovalConfig
from ms_core.utils.file_handler import parse_mz_rt_string
from ms_core.utils.validators import detect_fixed_columns


class DuplicateRemover(BaseProcessor):
    """
    Removes duplicate signals from mass spectrometry data.

    This processor:
    1. Automatically detects RT, m/z, and intensity columns
    2. Groups signals by m/z within tolerance (RT binning)
    3. Preserves highest intensity signals in each group
    4. Protects red-font marked rows from removal
    5. Optionally limits output to top N signals
    """

    def __init__(self, config: Optional[DuplicateRemovalConfig] = None):
        """
        Initialize the Duplicate Remover.

        Args:
            config: Configuration options for duplicate removal
        """
        super().__init__("Duplicate Remover")
        self.config = config or DuplicateRemovalConfig()

    def validate_input(self, df: pd.DataFrame) -> tuple:
        """
        Validate input data for duplicate removal.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (is_valid, error_message)
        """
        if df is None or df.empty:
            return False, "Input data is empty"

        if len(df) < 2:
            return False, "Data must have at least 2 rows"

        return True, ""

    def process(
        self,
        df: pd.DataFrame,
        mz_tolerance_ppm: Optional[float] = None,
        rt_tolerance: Optional[float] = None,
        merge_mode: Optional[str] = None,
        top_n: Optional[int] = None,
        protected_rows: Optional[Set[int]] = None,
        enable_degeneracy_annotation: Optional[bool] = None,
        degeneracy_ppm_tolerance: Optional[float] = None,
        degeneracy_rt_tolerance: Optional[float] = None,
        degeneracy_correlation_threshold: Optional[float] = None,
        degeneracy_min_correlation_points: Optional[int] = None,
        degeneracy_adduct_table_file: Optional[str] = None,
        **kwargs,
    ) -> ProcessingResult:
        """
        Process data to remove duplicate signals.

        Args:
            df: Input DataFrame
            mz_tolerance_ppm: m/z tolerance in ppm (default from config)
            rt_tolerance: RT tolerance in minutes (default from config)
            merge_mode: Duplicate-row merge policy (`per_sample_max` or `fill_gaps`)
            top_n: Limit output to top N signals by intensity
            protected_rows: Set of row indices to protect from removal
            **kwargs: Additional parameters

        Returns:
            ProcessingResult with deduplicated data
        """
        self.reset()

        # Use config defaults if not specified
        mz_tol = mz_tolerance_ppm if mz_tolerance_ppm is not None else self.config.mz_tolerance_ppm
        rt_tol = rt_tolerance if rt_tolerance is not None else self.config.rt_tolerance
        normalized_merge_mode = self._normalize_merge_mode(
            merge_mode if merge_mode is not None else self.config.merge_mode
        )

        degeneracy_enabled = (
            enable_degeneracy_annotation
            if enable_degeneracy_annotation is not None
            else self.config.enable_degeneracy_annotation
        )
        degeneracy_ppm = (
            degeneracy_ppm_tolerance
            if degeneracy_ppm_tolerance is not None
            else self.config.degeneracy_ppm_tolerance
        )
        degeneracy_rt = (
            degeneracy_rt_tolerance
            if degeneracy_rt_tolerance is not None
            else self.config.degeneracy_rt_tolerance
        )
        degeneracy_adduct_table_path = (
            degeneracy_adduct_table_file
            if degeneracy_adduct_table_file is not None
            else self.config.degeneracy_adduct_table_file
        )
        degeneracy_corr_threshold = (
            degeneracy_correlation_threshold
            if degeneracy_correlation_threshold is not None
            else self.config.degeneracy_correlation_threshold
        )
        degeneracy_min_corr_points_value = (
            degeneracy_min_correlation_points
            if degeneracy_min_correlation_points is not None
            else self.config.degeneracy_min_correlation_points
        )

        # Validate input
        is_valid, error_msg = self.validate_input(df)
        if not is_valid:
            return ProcessingResult(
                success=False,
                errors=[error_msg],
                message=f"Validation failed: {error_msg}",
            )

        self.update_progress(10, "Starting duplicate removal...")

        try:
            # Create a copy
            result_df = df.copy()
            result_df["_orig_index"] = range(len(result_df))

            # Step 1: Detect columns
            self.update_progress(20, "Detecting columns...")
            col_info = self._detect_columns(result_df)

            if self._cancelled:
                return ProcessingResult(success=False, message="Processing cancelled")

            # Step 2: Parse m/z and RT values
            self.update_progress(30, "Parsing m/z and RT values...")
            result_df, parse_stats = self._parse_mz_rt(result_df, col_info)

            if self._cancelled:
                return ProcessingResult(success=False, message="Processing cancelled")

            # Step 3: Calculate total intensity
            self.update_progress(40, "Calculating intensities...")
            result_df = self._calculate_intensities(result_df, col_info)

            if self._cancelled:
                return ProcessingResult(success=False, message="Processing cancelled")

            # Step 4: Find unique signals
            self.update_progress(50, "Identifying duplicate signals...")
            unique_indices, dup_stats, merge_groups = self._find_unique_signals(
                result_df,
                mz_tol,
                rt_tol,
                protected_rows or set(),
            )

            if self._cancelled:
                return ProcessingResult(success=False, message="Processing cancelled")

            # Step 5: Merge intensity data from duplicate rows into representatives
            self.update_progress(65, "Merging duplicate intensity data...")
            result_df, merge_stats = self._merge_duplicate_groups(
                result_df,
                merge_groups,
                col_info["intensity_cols"],
                normalized_merge_mode,
            )
            dup_stats.update(merge_stats)

            # Recalculate occurrence/intensity after merge
            result_df = self._calculate_intensities(result_df, col_info)

            if self._cancelled:
                return ProcessingResult(success=False, message="Processing cancelled")

            # Step 6: Filter to unique signals
            self.update_progress(80, "Filtering duplicates...")

            # Get header rows
            header_rows = result_df.iloc[:1].copy()
            data_rows = result_df.iloc[1:].copy()

            # Filter data rows
            data_positions = [idx - 1 for idx in unique_indices if idx >= 1]
            unique_data = data_rows.iloc[data_positions].copy()

            # Sort by total intensity (descending)
            if "_total_intensity" in unique_data.columns:
                unique_data = unique_data.sort_values("_total_intensity", ascending=False)

            # Apply top N limit
            if top_n and top_n > 0 and len(unique_data) > top_n:
                unique_data = unique_data.head(top_n)
                dup_stats["limited_to_top_n"] = top_n

            # Track protected rows after filtering/sorting
            protected_rows = protected_rows or set()
            new_protected_rows = set()
            if 0 in protected_rows:
                new_protected_rows.add(0)
            if protected_rows and "_orig_index" in unique_data.columns:
                for new_idx, orig_idx in enumerate(unique_data["_orig_index"].tolist()):
                    if orig_idx in protected_rows:
                        new_protected_rows.add(new_idx + 1)

            degeneracy_stats: Dict[str, Any] = {
                "degeneracy_annotation_enabled": bool(degeneracy_enabled),
                "degeneracy_matches": 0,
                "degeneracy_groups": 0,
                "degeneracy_base_count": 0,
                "degeneracy_adduct_count": 0,
                "degeneracy_corr_rejected": 0,
            }
            adduct_table_source = "disabled"
            if degeneracy_enabled:
                self.update_progress(90, "Annotating degeneracy relationships...")
                unique_data, degeneracy_stats, adduct_table_source = self._annotate_degeneracy(
                    unique_data,
                    col_info=col_info,
                    sample_type_row=result_df.iloc[0].copy(),
                    ppm_tolerance=degeneracy_ppm,
                    rt_tolerance=degeneracy_rt,
                    correlation_threshold=degeneracy_corr_threshold,
                    min_correlation_points=degeneracy_min_corr_points_value,
                    adduct_table_file=degeneracy_adduct_table_path,
                )

            # Clean up temporary columns
            temp_cols = ["_mz", "_rt", "_total_intensity", "_occurrence", "_orig_index"]
            for col in temp_cols:
                if col in unique_data.columns:
                    unique_data = unique_data.drop(col, axis=1)
                if col in header_rows.columns:
                    header_rows = header_rows.drop(col, axis=1)

            # Reconstruct DataFrame
            result_df = pd.concat(
                [header_rows.reset_index(drop=True), unique_data.reset_index(drop=True)],
                ignore_index=True,
            )

            self.update_progress(100, "Duplicate removal complete")

            # Compile statistics
            stats = {
                **parse_stats,
                **dup_stats,
                **degeneracy_stats,
                "final_features": len(result_df) - 1,
            }

            message = f"Duplicate removal completed. Removed {dup_stats.get('duplicates_removed', 0)} duplicates."
            recovered = dup_stats.get("data_points_recovered", 0)
            upgraded = dup_stats.get("data_points_upgraded", 0)
            if recovered > 0:
                message += f" Merged {dup_stats.get('groups_merged', 0)} groups, recovered {recovered} data points."
            elif dup_stats.get("groups_merged", 0) > 0:
                message += f" Merged {dup_stats.get('groups_merged', 0)} groups."
            if upgraded > 0:
                message += f" Upgraded {upgraded} overlapping data points via {normalized_merge_mode}."
            if degeneracy_enabled:
                message += f" Annotated {degeneracy_stats.get('degeneracy_adduct_count', 0)} degeneracy features."

            return ProcessingResult(
                success=True,
                data=result_df,
                message=message,
                statistics=stats,
                metadata={
                    "mz_tolerance_ppm": mz_tol,
                    "rt_tolerance": rt_tol,
                    "merge_mode": normalized_merge_mode,
                    "column_info": col_info,
                    "red_font_rows": sorted(new_protected_rows),
                    "protected_rows": sorted(new_protected_rows),
                    "degeneracy_annotation_enabled": bool(degeneracy_enabled),
                    "degeneracy_ppm_tolerance": degeneracy_ppm,
                    "degeneracy_rt_tolerance": degeneracy_rt,
                    "degeneracy_correlation_threshold": degeneracy_corr_threshold,
                    "degeneracy_min_correlation_points": degeneracy_min_corr_points_value,
                    "degeneracy_adduct_table_source": adduct_table_source,
                },
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                errors=[str(e)],
                message=f"Error during duplicate removal: {str(e)}",
            )

    def _detect_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect RT, m/z, and intensity columns.

        Returns dict with column information.
        """
        col_info = {
            "feature_col": df.columns[0],
            "combined_mz_rt": False,
            "rt_col": None,
            "mz_col": None,
            "intensity_cols": [],
        }

        # Check if first column is combined m/z/RT format
        first_col = df.columns[0]
        sample_values = df[first_col].iloc[1:6] if len(df) > 1 else []

        valid_combined = 0
        for val in sample_values:
            mz, rt = parse_mz_rt_string(str(val))
            if mz is not None and rt is not None:
                valid_combined += 1

        if (
            sample_values is not None
            and len(sample_values) > 0
            and valid_combined >= len(sample_values) * 0.6
        ):
            col_info["combined_mz_rt"] = True
        else:
            # Look for separate RT and m/z columns
            for col in df.columns:
                col_lower = str(col).lower()
                if any(kw in col_lower for kw in self.config.rt_keywords):
                    col_info["rt_col"] = col
                elif any(kw in col_lower for kw in self.config.mz_keywords):
                    col_info["mz_col"] = col

        # Identify intensity columns (typically data columns)
        # Skip fixed columns (Mz/RT and optional Tolerance)
        fixed_cols, start_idx = detect_fixed_columns(df)
        if not fixed_cols:
            start_idx = 1

        for col in df.columns[start_idx:]:
            if str(col).startswith("_"):
                continue
            col_lower = str(col).lower()
            # Check if column name suggests intensity data
            is_intensity = any(kw in col_lower for kw in self.config.intensity_keywords)
            # Or if column contains numeric data
            if is_intensity or self._is_numeric_column(
                df[col].iloc[1:] if len(df) > 1 else df[col]
            ):
                col_info["intensity_cols"].append(col)

        return col_info

    @staticmethod
    def _normalize_merge_mode(merge_mode: str) -> MergeMode:
        """Compatibility wrapper for duplicate-row merge policy validation."""
        return DuplicateIntensityMerger.normalize_merge_mode(merge_mode)

    @staticmethod
    def _coerce_positive_number(value: Any) -> float | None:
        """Compatibility wrapper for positive numeric cell coercion."""
        return DuplicateIntensityMerger.coerce_positive_number(value)

    def _is_numeric_column(self, series: pd.Series) -> bool:
        """Check if a series contains numeric values.

        LC-MS intensity columns are highly sparse (each sample typically detects
        only 15-25% of features), so even a single numeric value qualifies.
        Columns that are already numeric dtype are accepted immediately.
        """
        if pd.api.types.is_numeric_dtype(series):
            return True
        try:
            converted = pd.to_numeric(series, errors="coerce")
            return bool(converted.notna().any())
        except Exception:
            return False

    def _parse_mz_rt(
        self,
        df: pd.DataFrame,
        col_info: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Parse m/z and RT values into separate columns."""
        stats = {"rows_parsed": 0, "parse_errors": 0}

        if col_info["combined_mz_rt"]:
            feature_col = col_info["feature_col"]
            col = df[feature_col].astype(str)
            parts = col.str.split("/", n=1, expand=True)
            mz_series = pd.to_numeric(parts[0], errors="coerce")
            rt_series = pd.to_numeric(parts[1], errors="coerce")
            # Ensure sample type row (index 0) is NaN
            if len(mz_series) > 0:
                mz_series.iloc[0] = np.nan
                rt_series.iloc[0] = np.nan

            stats["rows_parsed"] = int(mz_series.notna().sum())
            stats["parse_errors"] = int(len(df) - stats["rows_parsed"] - 1)
            df["_mz"] = mz_series.to_numpy()
            df["_rt"] = rt_series.to_numpy()
        else:
            # Use separate columns
            mz_series = (
                pd.to_numeric(df[col_info["mz_col"]], errors="coerce")
                if col_info["mz_col"]
                else pd.Series([np.nan] * len(df))
            )
            rt_series = (
                pd.to_numeric(df[col_info["rt_col"]], errors="coerce")
                if col_info["rt_col"]
                else pd.Series([np.nan] * len(df))
            )
            if len(mz_series) > 0:
                mz_series.iloc[0] = np.nan
                rt_series.iloc[0] = np.nan
            stats["rows_parsed"] = int(mz_series.notna().sum())
            stats["parse_errors"] = int(len(df) - stats["rows_parsed"] - 1)
            df["_mz"] = mz_series.to_numpy()
            df["_rt"] = rt_series.to_numpy()

        return df, stats

    def _calculate_intensities(
        self,
        df: pd.DataFrame,
        col_info: Dict[str, Any],
    ) -> pd.DataFrame:
        """Calculate total intensity and occurrence count for each feature."""
        intensity_cols = col_info["intensity_cols"]
        if intensity_cols:
            intensity_df = df[intensity_cols].apply(pd.to_numeric, errors="coerce")
            mask = intensity_df > 0
            total_intensity = intensity_df.where(mask).sum(axis=1, skipna=True)
            occurrence = mask.sum(axis=1)
        else:
            total_intensity = pd.Series([0] * len(df), index=df.index)
            occurrence = pd.Series([0] * len(df), index=df.index)

        # Ensure sample type row is zero
        if len(df) > 0:
            total_intensity.iloc[0] = 0
            occurrence.iloc[0] = 0

        df["_total_intensity"] = total_intensity.to_numpy()
        df["_occurrence"] = occurrence.to_numpy()

        return df

    def _find_unique_signals(
        self,
        df: pd.DataFrame,
        mz_tolerance_ppm: float,
        rt_tolerance: float,
        protected_rows: Set[int],
    ) -> Tuple[Set[int], Dict[str, Any], List[List[int]]]:
        """
        Find unique signals using RT-window grouping and m/z tolerance.

        Protected rows only influence representative selection. Once a
        representative is chosen, downstream merge behavior is still governed
        by ``merge_mode``.

        Returns:
            keep_indices: set of row indices to keep (representative per group)
            stats: deduplication statistics
            merge_groups: list of [best_idx, donor_idx, ...] lists for intensity merging
        """
        stats = {
            "original_features": len(df) - 1,
            "duplicates_removed": 0,
            "protected_kept": 0,
        }

        if len(df) <= 1:
            return set(range(len(df))), stats, []

        keep_indices = {0}  # Always keep Sample_Type row
        merge_groups: List[List[int]] = []

        # Build valid rows list
        valid_rows = []
        for idx in range(1, len(df)):
            mz = df.at[idx, "_mz"]
            rt = df.at[idx, "_rt"]
            if mz is not None and rt is not None and mz > 0:
                valid_rows.append(
                    {
                        "idx": idx,
                        "mz": mz,
                        "rt": rt,
                        "intensity": df.at[idx, "_total_intensity"],
                        "occurrence": df.at[idx, "_occurrence"],
                        "protected": idx in protected_rows,
                    }
                )

        # Sort by RT first to allow a forward RT window scan.
        valid_rows.sort(key=lambda x: (x["rt"], x["mz"]))
        rt_window = rt_tolerance if rt_tolerance > 0 else float("inf")
        processed_ids: Set[int] = set()

        for i, current in enumerate(valid_rows):
            if current["idx"] in processed_ids:
                continue

            group = [current]

            # Compare with following rows only while still inside RT window.
            j = i + 1
            while j < len(valid_rows):
                other = valid_rows[j]
                if (other["rt"] - current["rt"]) > rt_window:
                    break
                if other["idx"] in processed_ids:
                    j += 1
                    continue
                ppm_diff = abs((current["mz"] - other["mz"]) / current["mz"] * 1_000_000)
                if ppm_diff <= mz_tolerance_ppm:
                    group.append(other)
                j += 1

            # Select representative
            protected_in_group = [r for r in group if r["protected"]]
            if protected_in_group:
                best = max(protected_in_group, key=lambda x: (x["occurrence"], x["intensity"]))
                keep_indices.add(best["idx"])
                stats["protected_kept"] += 1
            else:
                best = max(group, key=lambda x: (x["occurrence"], x["intensity"]))
                keep_indices.add(best["idx"])

            # Record merge group: [best_idx, donor1_idx, donor2_idx, ...]
            if len(group) > 1:
                donor_indices = [r["idx"] for r in group if r["idx"] != best["idx"]]
                merge_groups.append([best["idx"]] + donor_indices)

            stats["duplicates_removed"] += len(group) - 1
            for r in group:
                processed_ids.add(r["idx"])

        return keep_indices, stats, merge_groups

    def _merge_duplicate_groups(
        self,
        df: pd.DataFrame,
        merge_groups: List[List[int]],
        intensity_cols: List[str],
        merge_mode: MergeMode,
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Compatibility wrapper for delegated duplicate intensity merge policy."""
        return DuplicateIntensityMerger().merge(df, merge_groups, intensity_cols, merge_mode)

    def get_duplicate_groups(
        self,
        df: pd.DataFrame,
        mz_tolerance_ppm: float,
        rt_tolerance: float,
    ) -> List[List[int]]:
        """
        Get groups of duplicate signals for inspection.

        Returns list of groups, where each group is a list of row indices.
        """
        # Parse m/z and RT
        col_info = self._detect_columns(df)
        df_with_mz_rt, _ = self._parse_mz_rt(df.copy(), col_info)

        groups = []
        processed = set()

        for idx in range(1, len(df_with_mz_rt)):
            if idx in processed:
                continue

            mz = df_with_mz_rt.at[idx, "_mz"]
            rt = df_with_mz_rt.at[idx, "_rt"]

            if mz is None or rt is None:
                continue

            # Find all duplicates
            group = [idx]
            processed.add(idx)

            for other_idx in range(idx + 1, len(df_with_mz_rt)):
                if other_idx in processed:
                    continue

                other_mz = df_with_mz_rt.at[other_idx, "_mz"]
                other_rt = df_with_mz_rt.at[other_idx, "_rt"]

                if other_mz is None or other_rt is None:
                    continue

                # Check tolerance
                if mz != 0:
                    ppm_diff = abs((mz - other_mz) / mz * 1_000_000)
                else:
                    ppm_diff = float("inf")

                rt_diff = abs(rt - other_rt)

                if ppm_diff <= mz_tolerance_ppm and rt_diff <= rt_tolerance:
                    group.append(other_idx)
                    processed.add(other_idx)

            if len(group) > 1:
                groups.append(group)

        return groups

    def _annotate_degeneracy(
        self,
        df: pd.DataFrame,
        *,
        col_info: Dict[str, Any],
        sample_type_row: pd.Series,
        ppm_tolerance: float,
        rt_tolerance: float,
        correlation_threshold: float,
        min_correlation_points: int,
        adduct_table_file: Optional[str],
    ) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
        """Compatibility wrapper for delegated degeneracy annotation."""
        return DegeneracyAnnotator().annotate(
            df,
            col_info=col_info,
            sample_type_row=sample_type_row,
            ppm_tolerance=ppm_tolerance,
            rt_tolerance=rt_tolerance,
            correlation_threshold=correlation_threshold,
            min_correlation_points=min_correlation_points,
            adduct_table_file=adduct_table_file,
        )

    def _select_degeneracy_correlation_columns(
        self,
        sample_type_row: pd.Series,
        col_info: Dict[str, Any],
    ) -> List[str]:
        """Compatibility wrapper for delegated degeneracy correlation-column selection."""
        return DegeneracyAnnotator._select_correlation_columns(sample_type_row, col_info)

    def _compute_feature_correlation(
        self,
        df: pd.DataFrame,
        base_idx: int,
        pair_idx: int,
        correlation_cols: List[str],
        min_correlation_points: int,
    ) -> Optional[float]:
        """Compatibility wrapper for delegated degeneracy Pearson correlation."""
        return DegeneracyAnnotator._compute_feature_correlation(
            df,
            base_idx,
            pair_idx,
            correlation_cols,
            min_correlation_points,
        )

    def _find_best_adduct_match(
        self,
        mz_diff: float,
        reference_mz: float,
        adduct_table: pd.DataFrame,
        ppm_tolerance: float,
    ) -> Optional[Dict[str, Any]]:
        """Compatibility wrapper for delegated adduct-table matching."""
        return DegeneracyAnnotator._find_best_adduct_match(
            mz_diff,
            reference_mz,
            adduct_table,
            ppm_tolerance,
        )

    def _load_adduct_table(self, custom_file: Optional[str]) -> Tuple[pd.DataFrame, str]:
        """Compatibility wrapper for delegated adduct-table loading."""
        return DegeneracyAnnotator()._load_adduct_table(custom_file)

    def _create_default_adduct_table(self) -> pd.DataFrame:
        """Compatibility wrapper for delegated built-in adduct table."""
        return DegeneracyAnnotator._create_default_adduct_table()
