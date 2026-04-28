"""Combined FH/MZmine TSV preprocessing for Step1."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Union

import pandas as pd

from ms_core.preprocessing.base import ProcessingResult


class StatisticsProcessor(Protocol):
    """Callable contract for processing one side of a combined TSV in statistics mode."""

    def __call__(
        self,
        *,
        df: pd.DataFrame,
        method_file: Optional[Union[str, Path]] = None,
        mz_decimals: int = 4,
        rt_decimals: int = 2,
        sample_type_mapping: Optional[Dict[str, str]] = None,
    ) -> ProcessingResult:
        """Process one FH or MZmine side and return a Step1-style result."""


class CombinedTsvPreprocessor:
    """Preprocess combined FH/MZmine TSV inputs and apply the false-positive fix."""

    # MZmine exports chromatographic peak area integrated over time in minutes,
    # while FH / XIC Extractor use seconds.  Multiply MZmine area by this factor
    # so all downstream values share a common unit (counts · s).
    MZMINE_AREA_UNIT_FACTOR: float = 60.0

    def __init__(self, statistics_processor: Optional[StatisticsProcessor] = None) -> None:
        self._statistics_processor = statistics_processor

    @staticmethod
    def detect_split(df: pd.DataFrame) -> Optional[int]:
        """Return the column index of the MZmine ID column marking the FH/MZmine split."""
        for idx, col in enumerate(df.columns):
            compact = re.sub(r"[^a-z0-9]+", "", str(col).strip().lower())
            if compact == "mzmineid":
                return idx
        return None

    def process_combined(
        self,
        df: pd.DataFrame,
        method_file: Optional[Union[str, Path]] = None,
        mz_decimals: int = 4,
        rt_decimals: int = 2,
        sample_type_mapping: Optional[Dict[str, str]] = None,
    ) -> ProcessingResult:
        """Process a combined FH+MZmine TSV directly into beforeVBA format."""
        if self._statistics_processor is None:
            return ProcessingResult(
                success=False,
                errors=["Statistics processor is required for combined TSV preprocessing"],
                message="Combined mode requires a statistics-mode processor",
            )

        split_idx = self.detect_split(df)
        if split_idx is None:
            return ProcessingResult(
                success=False,
                errors=["No 'MZmine ID' column found — cannot identify FH/MZmine boundary"],
                message="Combined mode requires a 'MZmine ID' split column in the input",
            )

        fh_df = df.iloc[:, :split_idx].copy().reset_index(drop=True)
        mz_df = df.iloc[:, split_idx:].copy().reset_index(drop=True)

        valid_mz_cols = [
            c
            for c in mz_df.columns
            if not (str(c).startswith("Unnamed:") and mz_df[c].isna().all())
        ]
        mz_df = mz_df[valid_mz_cols]

        fh_result = self._statistics_processor(
            df=fh_df,
            method_file=method_file,
            mz_decimals=mz_decimals,
            rt_decimals=rt_decimals,
            sample_type_mapping=sample_type_mapping,
        )
        if not fh_result.success:
            return ProcessingResult(
                success=False,
                errors=fh_result.errors,
                message=f"FH side processing failed: {fh_result.message}",
            )

        mz_result = self._statistics_processor(
            df=mz_df,
            method_file=method_file,
            mz_decimals=mz_decimals,
            rt_decimals=rt_decimals,
            sample_type_mapping=sample_type_mapping,
        )
        if not mz_result.success:
            return ProcessingResult(
                success=False,
                errors=mz_result.errors,
                message=f"MZmine side processing failed: {mz_result.message}",
            )

        fh_data = fh_result.data.reset_index(drop=True)
        mz_data = mz_result.data.reset_index(drop=True)

        is_mzmine_id = [
            re.sub(r"[^a-z0-9]+", "", str(c).strip().lower()) == "mzmineid"
            for c in mz_data.columns
        ]
        id_cols = [c for c, flag in zip(mz_data.columns, is_mzmine_id) if flag]
        other_cols = [c for c, flag in zip(mz_data.columns, is_mzmine_id) if not flag]
        mz_data = mz_data[id_cols + other_cols]

        combined = pd.concat([fh_data, mz_data], axis=1)

        return ProcessingResult(
            success=True,
            data=combined,
            message="Combined mode completed.",
            statistics={
                "mode": "combined",
                "fh_cols": len(fh_data.columns),
                "mz_cols": len(mz_data.columns),
                "total_cols": len(combined.columns),
                "total_rows": len(combined),
            },
            metadata={
                "mode": "combined",
                "fh_metadata": fh_result.metadata,
                "mz_metadata": mz_result.metadata,
                "sample_info": fh_result.metadata.get("sample_info")
                if fh_result.metadata
                else None,
            },
        )

    @staticmethod
    def post_vba_cleanup(df: pd.DataFrame) -> pd.DataFrame:
        """Replace FH Mz/RT with MZmine values and drop the MZmine side."""
        mzmine_id_idx = CombinedTsvPreprocessor.detect_split(df)
        if mzmine_id_idx is None:
            raise ValueError(
                "MZmine ID column not found — is this a beforeVBA/afterVBA format file?"
            )

        mzmine_mz_idx = mzmine_id_idx + 1
        mzmine_rt_idx = mzmine_id_idx + 2

        if mzmine_rt_idx >= len(df.columns):
            raise ValueError(
                "Expected MZmine m/z and RT columns immediately after MZmine ID column"
            )

        result = df.copy()
        result.iloc[:, 0] = result.iloc[:, mzmine_mz_idx]
        result.iloc[:, 1] = result.iloc[:, mzmine_rt_idx]
        return result.iloc[:, :mzmine_id_idx]

    def false_positive_fix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply MZmine false-positive filtering to a beforeVBA format DataFrame."""
        mzmine_id_idx = self.detect_split(df)
        if mzmine_id_idx is None:
            raise ValueError("MZmine ID column not found — expected beforeVBA format input")

        mzmine_mz_idx = mzmine_id_idx + 1
        mzmine_rt_idx = mzmine_id_idx + 2
        mzmine_area_start = mzmine_id_idx + 3

        result = df.copy()

        keep = ~(
            result.iloc[:, mzmine_id_idx].map(self._is_blank_missing)
            | result.iloc[:, mzmine_mz_idx].map(self._is_blank_missing)
            | result.iloc[:, mzmine_rt_idx].map(self._is_blank_missing)
        )
        result = result.loc[keep].reset_index(drop=True)

        mzmine_area_name_to_pos: Dict[str, int] = {}
        for i in range(mzmine_area_start, len(df.columns)):
            name = str(df.columns[i])
            if name not in mzmine_area_name_to_pos:
                mzmine_area_name_to_pos[name] = i

        for fh_pos in range(2, mzmine_id_idx):
            fh_col_name = str(result.columns[fh_pos])
            area_pos = mzmine_area_name_to_pos.get(fh_col_name)
            if area_pos is None:
                continue
            fh_vals = result.iloc[:, fh_pos]
            area_vals = pd.to_numeric(result.iloc[:, area_pos], errors="coerce")
            fh_present = ~fh_vals.map(self._is_measurement_missing)
            area_present = ~(area_vals.isna() | area_vals.eq(0))
            merged = pd.to_numeric(fh_vals, errors="coerce")
            merged = merged.mask(merged.eq(0))
            replace_mask = fh_present & area_present
            merged.loc[replace_mask] = (
                area_vals.loc[replace_mask] * self.MZMINE_AREA_UNIT_FACTOR
            )
            merged = merged.mask(merged.eq(0))
            result = self._assign_column_by_position(result, fh_pos, merged.astype("float64"))

        result.iloc[:, 0] = result.iloc[:, mzmine_mz_idx].to_numpy()
        result.iloc[:, 1] = result.iloc[:, mzmine_rt_idx].to_numpy()

        final = result.iloc[:, :mzmine_id_idx].copy()
        for fh_pos in range(2, len(final.columns)):
            sample_vals = pd.to_numeric(final.iloc[:, fh_pos], errors="coerce")
            final = self._assign_column_by_position(
                final,
                fh_pos,
                sample_vals.mask(sample_vals.eq(0)).astype("float64"),
            )
        return final

    def process_combined_and_fix(
        self,
        df: pd.DataFrame,
        method_file: Optional[Union[str, Path]] = None,
        mz_decimals: int = 4,
        rt_decimals: int = 2,
        sample_type_mapping: Optional[Dict[str, str]] = None,
    ) -> ProcessingResult:
        """Run combined TSV processing and false-positive filtering end to end."""
        combined_result = self.process_combined(
            df=df,
            method_file=method_file,
            mz_decimals=mz_decimals,
            rt_decimals=rt_decimals,
            sample_type_mapping=sample_type_mapping,
        )
        if not combined_result.success:
            return combined_result

        try:
            final_df = self.false_positive_fix(combined_result.data)
        except Exception as exc:
            return ProcessingResult(
                success=False,
                errors=[str(exc)],
                message=f"False-positive fix failed: {exc}",
            )

        input_rows = len(combined_result.data) - 1
        return ProcessingResult(
            success=True,
            data=final_df,
            message=f"Combined pipeline completed. {input_rows - len(final_df)} features removed.",
            statistics={
                "mode": "combined_fix",
                "input_features": input_rows,
                "output_features": len(final_df),
                "removed_features": input_rows - len(final_df),
                "output_cols": len(final_df.columns),
            },
            metadata={
                **(combined_result.metadata or {}),
                "mode": "combined_fix",
            },
        )

    @staticmethod
    def _is_blank_missing(value: Any) -> bool:
        if pd.isna(value):
            return True
        return str(value).strip().upper() in ("", "NA", "NAN")

    @classmethod
    def _is_measurement_missing(cls, value: Any) -> bool:
        if cls._is_blank_missing(value):
            return True
        try:
            return float(str(value).strip()) == 0.0
        except ValueError:
            return False

    @staticmethod
    def _assign_column_by_position(
        df: pd.DataFrame,
        position: int,
        values: pd.Series,
    ) -> pd.DataFrame:
        """Assign one positional column while preserving duplicate column labels."""
        columns = list(df.columns)
        frame_values = df.to_numpy(dtype=object, copy=True)
        frame_values[:, position] = values.to_numpy()
        return pd.DataFrame(frame_values, columns=columns)
