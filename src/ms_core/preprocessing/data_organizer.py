"""
Data Organizer Module - Step 1 of the preprocessing pipeline.

This module handles initial data organization and standardization:
- Merge Mz and RT columns into Mz/RT format (mz/RT)
- Simplify column headers (extract sample names from paths)
- Add Sample_Type row with auto-detection
- Parse method file (Word) for sample type mapping and injection sequence
- Create SampleInfo worksheet with injection order
- Reorder columns based on Injection_Order

Input format expected:
    Mz | RT | Intensity of path1 | Intensity of path2 | ...

Output format (RawIntensity):
    Mz/RT          | Sample1 | Sample2 | ...  (ordered by Injection_Order)
    Sample_Type    | exposure| control | ...
    252.1098/18.45 | 12345   | 67890   | ...

Output format (SampleInfo):
    Sample_Name | Sample_Type | Injection_Order | Injection_Volume | Method_Sample_Name
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import pandas as pd

from ms_core.preprocessing.base import BaseProcessor, ProcessingResult
from ms_core.preprocessing.combined_tsv import CombinedTsvPreprocessor
from ms_core.preprocessing.data_organizer_layout import (
    expand_pre_merged_mz_rt,
    extract_sample_type_row_from_input,
    is_pre_merged_mz_rt_header,
    move_leading_metadata_to_end,
    normalize_sample_type_value,
)
from ms_core.preprocessing.data_organizer_matrix import (
    auto_detect_sample_types as auto_detect_sample_types_from_columns,
    detect_fixed_columns_for_statistics,
    detect_sample_type,
    finalize_structure,
    find_matching_sample_column_position,
    insert_sample_type_row,
    merge_mz_rt,
    reorder_columns_by_injection,
    reorder_columns_statistics_mode,
    simplify_headers,
    validate_statistics_input,
)
from ms_core.preprocessing.data_organizer_method import (
    extract_sample_id,
    parse_method_file,
)
from ms_core.preprocessing.data_organizer_step1 import (
    assemble_step1_output,
    prepare_step1_input,
    simplify_input_sample_type_overrides,
)
from ms_core.preprocessing.method_sequence import (
    InjectionInfo,
    extract_docx_tables_fallback,
    extract_injection_rows_from_table,
    parse_injection_sequence,
    parse_injection_volume_from_cells,
)
from ms_core.preprocessing.sample_identity import (
    extract_primary_sample_token,
    extract_raw_sample_name,
    is_likely_sample_name,
    normalize_sample_key,
    simplify_method_sample_name,
)
from ms_core.preprocessing.sample_info_builder import SampleInfoBuilder, is_non_sample_column
from ms_core.preprocessing.settings import DataOrganizerConfig


class DataOrganizer(BaseProcessor):
    """
    Organizes and standardizes raw mass spectrometry data.

    This processor prepares the data for subsequent processing steps by:
    1. Merging Mz and RT columns into Mz/RT format
    2. Simplifying column headers to extract sample names
    3. Inserting Sample_Type row with auto-detection
    4. Optionally parsing method files for accurate sample mapping
    """

    # Patterns for sample type detection
    # Output mapping: tumor->Exposure, benign->Control, normal->Normal, qc->QC
    SAMPLE_TYPE_PATTERNS = {
        "QC": [r"qc", r"pool", r"quality"],
        "Exposure": [r"tumor", r"cancer", r"tumour"],  # Tumor -> Exposure
        "Normal": [r"normal", r"healthy"],
        "Control": [r"benign", r"benignfat", r"control"],  # Benign/control -> Control
        "blank": [r"blank", r"blk"],
        "standard": [r"std", r"standard", r"sdolek"],
    }
    def __init__(self, config: Optional[DataOrganizerConfig] = None):
        """
        Initialize the Data Organizer.

        Args:
            config: Configuration options for data organization
        """
        super().__init__("Data Organizer")
        self.config = config or DataOrganizerConfig()
        self._sample_info_builder = SampleInfoBuilder()
        self._combined_tsv_preprocessor = CombinedTsvPreprocessor(self._process_statistics_mode)

    @staticmethod
    def _is_pre_merged_mz_rt_header(col_name: str) -> bool:
        """Return True when a column header indicates an already-combined Mz/RT column.

        Accepts variants such as "Mz/RT", "mz/rt", "MZ/RT", "m/z/rt".
        """
        return is_pre_merged_mz_rt_header(col_name)

    def _expand_pre_merged_mz_rt(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        """Expand an already-combined Mz/RT column into separate numeric Mz and RT columns.

        If the first column is a combined "Mz/RT" column (e.g., "274.0920/18.32"),
        each value is parsed into a float Mz and a float RT, and two new columns
        ("Mz" and "RT") replace the single combined column.  All remaining columns
        are preserved as-is.

        Returns:
            Tuple of (resulting_df, was_expanded).  *was_expanded* is False when the
            first column was not a pre-merged Mz/RT column or no values could be parsed.
        """
        return expand_pre_merged_mz_rt(df)

    def _is_non_sample_column(self, column_name: str) -> bool:
        """Return True when a column is metadata and should not be treated as a sample."""
        return is_non_sample_column(column_name)

    def _normalize_sample_type_value(self, value: Any) -> Optional[str]:
        """Normalize sample type labels to toolkit's canonical values."""
        return normalize_sample_type_value(value)

    def _extract_sample_type_row_from_input(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, Any]]:
        """
        Extract user-provided Sample type row from raw input when present.

        Expected marker is in the first row / first column (e.g., "Sample Type").
        """
        extraction = extract_sample_type_row_from_input(df)
        return extraction.data, extraction.sample_types, extraction.stats

    def _extract_primary_sample_token(self, text: str) -> Optional[str]:
        """Extract a canonical sample token from free text."""
        return extract_primary_sample_token(text)

    def _normalize_sample_key(self, sample_name: str) -> str:
        """Normalize sample names from files/headers for robust matching."""
        return normalize_sample_key(sample_name)

    def _is_likely_sample_name(self, text: str) -> bool:
        """Identify whether a method-file entry looks like a real sample."""
        return is_likely_sample_name(text)

    def _move_leading_metadata_to_end(self, df: pd.DataFrame) -> pd.DataFrame:
        """Move any leading non-mz/rt metadata columns (e.g. MZmine ID) to the end.

        MZmine's default export puts the ID column first.  Relocating it lets
        validate_input and _merge_mz_rt find m/z and RT in the expected positions
        without requiring manual column reordering by the user.
        """
        return move_leading_metadata_to_end(df)

    def validate_input(self, df: pd.DataFrame) -> tuple:
        """
        Validate input data for organization.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (is_valid, error_message)
        """
        if df is None or df.empty:
            return False, "Input data is empty"

        if len(df.columns) < 2:
            return False, "Input data must have at least 2 columns"

        first_col = str(df.columns[0]).lower().strip()

        # Pre-merged Mz/RT: first column is already "Mz/RT" (combined).
        # Only one sample column is required in addition to the Mz/RT column.
        if self._is_pre_merged_mz_rt_header(first_col):
            return True, ""

        # Standard MZmine export: separate Mz and RT columns required.
        if len(df.columns) < 3:
            return (
                False,
                "Input data must have at least 3 columns (Mz, RT, and at least one sample)",
            )

        second_col = str(df.columns[1]).lower()

        if "mz" not in first_col and "m/z" not in first_col and "mass" not in first_col:
            return False, f"First column '{df.columns[0]}' doesn't appear to be m/z values"

        if "rt" not in second_col and "time" not in second_col and "retention" not in second_col:
            return False, f"Second column '{df.columns[1]}' doesn't appear to be RT values"

        return True, ""

    def process(
        self,
        df: pd.DataFrame,
        method_file: Optional[Union[str, Path]] = None,
        mz_decimals: int = 4,
        rt_decimals: int = 2,
        sample_type_mapping: Optional[Dict[str, str]] = None,
        mode: str = "normalization",
        **kwargs,
    ) -> ProcessingResult:
        """
        Process raw data for organization.

        Args:
            df: Input DataFrame with Mz, RT, and intensity columns
            method_file: Optional path to method file (Word) for sample mapping
            mz_decimals: Decimal places for m/z values (default: 4)
            rt_decimals: Decimal places for RT values (default: 2)
            sample_type_mapping: Custom sample name to type mapping
            **kwargs: Additional parameters

        Returns:
            ProcessingResult with organized data (RawIntensity and SampleInfo)
        """
        self.reset()

        mode_name = str(mode or "normalization").strip().lower()
        if mode_name not in {"normalization", "statistics", "combined", "combined_fix"}:
            return ProcessingResult(
                success=False,
                errors=[f"Unsupported mode: {mode}"],
                message=f"Unsupported mode: {mode}",
            )

        # Auto-detect combined TSV: MZmine ID found in the middle of the columns
        # (split_idx > 2 rules out standalone MZmine files where it sits at col 0)
        if mode_name in {"normalization", "statistics"}:
            _split = self._detect_combined_split(df)
            if _split is not None and 2 < _split < len(df.columns) - 1:
                mode_name = "combined_fix"

        if mode_name == "statistics":
            return self._process_statistics_mode(
                df=df,
                method_file=method_file,
                mz_decimals=mz_decimals,
                rt_decimals=rt_decimals,
                sample_type_mapping=sample_type_mapping,
            )

        if mode_name == "combined_fix":
            return self.process_combined_and_fix(
                df=df,
                method_file=method_file,
                mz_decimals=mz_decimals,
                rt_decimals=rt_decimals,
                sample_type_mapping=sample_type_mapping,
            )

        if mode_name == "combined":
            return self.process_combined(
                df=df,
                method_file=method_file,
                mz_decimals=mz_decimals,
                rt_decimals=rt_decimals,
                sample_type_mapping=sample_type_mapping,
            )

        df = self._move_leading_metadata_to_end(df)

        # Validate input
        is_valid, error_msg = self.validate_input(df)
        if not is_valid:
            return ProcessingResult(
                success=False,
                errors=[error_msg],
                message=f"Validation failed: {error_msg}",
            )

        self.update_progress(5, "Starting data organization...")

        try:
            # Step 1: Parse method file if provided
            self.update_progress(10, "Parsing method file...")
            prepared = prepare_step1_input(
                df,
                method_file=method_file,
                sample_type_mapping=sample_type_mapping,
                mode=mode_name,
                parse_method_file=self._parse_method_file,
                parse_injection_sequence=self._parse_injection_sequence,
            )
            result_df = prepared.data
            stats = prepared.statistics
            sample_mapping = prepared.sample_mapping
            injection_info_list = prepared.injection_info_list

            if self._cancelled:
                return ProcessingResult(success=False, message="Processing cancelled")

            # Step 2: Merge Mz and RT into Mz/RT
            self.update_progress(30, "Merging Mz/RT columns...")
            result_df, merge_stats = self._merge_mz_rt(
                result_df, mz_decimals, rt_decimals, include_tolerance_col=False
            )
            stats.update(merge_stats)

            if self._cancelled:
                return ProcessingResult(success=False, message="Processing cancelled")

            # Step 3: Simplify column headers
            self.update_progress(50, "Simplifying column headers...")
            result_df, header_mapping = self._simplify_headers(result_df)
            stats["columns_simplified"] = len(header_mapping)
            input_sample_types_simplified = simplify_input_sample_type_overrides(
                prepared.input_sample_types,
                header_mapping,
                self._extract_sample_name,
            )

            if self._cancelled:
                return ProcessingResult(success=False, message="Processing cancelled")

            # Step 4: Insert Sample_Type row
            self.update_progress(60, "Detecting sample types...")
            result_df, type_stats = self._insert_sample_type_row(
                result_df,
                header_mapping,
                sample_mapping,
                sample_type_overrides=input_sample_types_simplified,
            )
            stats.update(type_stats)

            if self._cancelled:
                return ProcessingResult(success=False, message="Processing cancelled")

            assembled = assemble_step1_output(
                result_df,
                injection_info_list=injection_info_list,
                build_sample_info=self._build_sample_info,
                reorder_columns_by_injection=self._reorder_columns_by_injection,
                finalize_structure=self._finalize_structure,
                update_progress=self.update_progress,
                cancellation_requested=lambda: self._cancelled,
            )
            if assembled.cancelled:
                return ProcessingResult(success=False, message="Processing cancelled")
            result_df = assembled.data
            sample_info_df = assembled.sample_info
            stats.update(assembled.statistics)

            self.update_progress(100, "Data organization complete")

            return ProcessingResult(
                success=True,
                data=result_df,
                message=f"Data organization completed. {stats['original_rows']} features processed.",
                statistics=stats,
                metadata={
                    "mode": mode_name,
                    "header_mapping": header_mapping,
                    "sample_mapping": sample_mapping,
                    "sample_info": sample_info_df,
                },
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                errors=[str(e)],
                message=f"Error during data organization: {str(e)}",
            )

    def _process_statistics_mode(
        self,
        df: pd.DataFrame,
        method_file: Optional[Union[str, Path]] = None,
        mz_decimals: int = 4,
        rt_decimals: int = 2,
        sample_type_mapping: Optional[Dict[str, str]] = None,
    ) -> ProcessingResult:
        """
        Statistics mode:
        - Keep separate Mz and RT output columns (no merged Mz/RT in final output)
        - Otherwise follow normalization workflow
        """
        df = self._move_leading_metadata_to_end(df)

        is_valid, error_msg = self.validate_input(df)
        if not is_valid:
            return ProcessingResult(
                success=False,
                errors=[error_msg],
                message=f"Validation failed: {error_msg}",
            )

        self.update_progress(5, "Starting statistics mode...")

        try:
            # Step 1: Parse method file if provided
            self.update_progress(10, "Parsing method file...")
            prepared = prepare_step1_input(
                df,
                method_file=method_file,
                sample_type_mapping=sample_type_mapping,
                mode="statistics",
                parse_method_file=self._parse_method_file,
                parse_injection_sequence=self._parse_injection_sequence,
            )
            result_df = prepared.data
            stats = prepared.statistics
            sample_mapping = prepared.sample_mapping
            injection_info_list = prepared.injection_info_list

            if self._cancelled:
                return ProcessingResult(success=False, message="Processing cancelled")

            # Step 2: Keep normalization internals, but restore separate Mz/RT later
            self.update_progress(30, "Preparing internal Mz/RT for downstream steps...")
            result_df, merge_stats = self._merge_mz_rt(
                result_df,
                mz_decimals=mz_decimals,
                rt_decimals=rt_decimals,
                include_tolerance_col=False,
            )
            stats["merge_skipped_in_output"] = True
            stats["invalid_values"] = merge_stats.get("invalid_values", 0)
            stats["mz_rt_merged"] = 0

            if self._cancelled:
                return ProcessingResult(success=False, message="Processing cancelled")

            # Step 3: Simplify column headers
            self.update_progress(50, "Simplifying column headers...")
            result_df, header_mapping = self._simplify_headers(result_df)
            stats["columns_simplified"] = len(header_mapping)
            input_sample_types_simplified = simplify_input_sample_type_overrides(
                prepared.input_sample_types,
                header_mapping,
                self._extract_sample_name,
            )

            if self._cancelled:
                return ProcessingResult(success=False, message="Processing cancelled")

            # Step 4: Insert Sample_Type row
            self.update_progress(60, "Detecting sample types...")
            result_df, type_stats = self._insert_sample_type_row(
                result_df,
                header_mapping,
                sample_mapping,
                sample_type_overrides=input_sample_types_simplified,
            )
            stats.update(type_stats)

            if self._cancelled:
                return ProcessingResult(success=False, message="Processing cancelled")

            assembled = assemble_step1_output(
                result_df,
                injection_info_list=injection_info_list,
                build_sample_info=self._build_sample_info,
                reorder_columns_by_injection=self._reorder_columns_by_injection,
                finalize_structure=self._finalize_structure,
                statistics_restore=prepared.statistics_restore,
                update_progress=self.update_progress,
                cancellation_requested=lambda: self._cancelled,
            )
            if assembled.cancelled:
                return ProcessingResult(success=False, message="Processing cancelled")
            result_df = assembled.data
            sample_info_df = assembled.sample_info
            stats.update(assembled.statistics)

            self.update_progress(100, "Statistics mode complete")

            return ProcessingResult(
                success=True,
                data=result_df,
                message="Statistics mode completed.",
                statistics=stats,
                metadata={
                    "mode": "statistics",
                    "header_mapping": header_mapping,
                    "sample_mapping": sample_mapping,
                    "sample_info": sample_info_df,
                },
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                errors=[str(e)],
                message=f"Error during statistics mode: {str(e)}",
            )

    def _detect_combined_split(self, df: pd.DataFrame) -> Optional[int]:
        """Return the column index of the MZmine ID column that marks the FH/MZmine split."""
        return CombinedTsvPreprocessor.detect_split(df)

    def process_combined(
        self,
        df: pd.DataFrame,
        method_file: Optional[Union[str, Path]] = None,
        mz_decimals: int = 4,
        rt_decimals: int = 2,
        sample_type_mapping: Optional[Dict[str, str]] = None,
    ) -> ProcessingResult:
        """Process a combined FH+MZmine TSV directly into beforeVBA format."""
        return self._combined_tsv_preprocessor.process_combined(
            df=df,
            method_file=method_file,
            mz_decimals=mz_decimals,
            rt_decimals=rt_decimals,
            sample_type_mapping=sample_type_mapping,
        )

    @staticmethod
    def post_vba_cleanup(df: pd.DataFrame) -> pd.DataFrame:
        """Replace FH Mz/RT with MZmine values and drop the MZmine side."""
        return CombinedTsvPreprocessor.post_vba_cleanup(df)

    MZMINE_AREA_UNIT_FACTOR: float = CombinedTsvPreprocessor.MZMINE_AREA_UNIT_FACTOR

    def false_positive_fix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply MZmine false-positive filtering to a beforeVBA format DataFrame."""
        return self._combined_tsv_preprocessor.false_positive_fix(df)

    def process_combined_and_fix(
        self,
        df: pd.DataFrame,
        method_file: Optional[Union[str, Path]] = None,
        mz_decimals: int = 4,
        rt_decimals: int = 2,
        sample_type_mapping: Optional[Dict[str, str]] = None,
    ) -> ProcessingResult:
        """Full end-to-end MZmine pipeline: combined TSV to final output."""
        return self._combined_tsv_preprocessor.process_combined_and_fix(
            df=df,
            method_file=method_file,
            mz_decimals=mz_decimals,
            rt_decimals=rt_decimals,
            sample_type_mapping=sample_type_mapping,
        )

    def _validate_statistics_input(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate input data for statistics mode."""
        return validate_statistics_input(df)

    def _detect_fixed_columns_for_statistics(self, df: pd.DataFrame) -> Tuple[List[str], int]:
        """Detect fixed columns for statistics-mode sorting."""
        return detect_fixed_columns_for_statistics(df)

    def _reorder_columns_statistics_mode(
        self,
        df: pd.DataFrame,
        injection_info_list: List[InjectionInfo],
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Reorder sample columns for statistics mode without creating SampleInfo."""
        return reorder_columns_statistics_mode(df, injection_info_list)

    def _find_matching_sample_column_position(
        self,
        df: pd.DataFrame,
        candidate_positions: List[int],
        file_name: str,
    ) -> Optional[int]:
        """Find a matching sample column index for a method-file sample name."""
        return find_matching_sample_column_position(df, candidate_positions, file_name)

    def _merge_mz_rt(
        self,
        df: pd.DataFrame,
        mz_decimals: int = 4,
        rt_decimals: int = 2,
        include_tolerance_col: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Merge Mz and RT columns into a single Mz/RT column.

        Format: "mz/RT" (e.g., "252.1098/18.45")
        """
        return merge_mz_rt(
            df,
            mz_decimals=mz_decimals,
            rt_decimals=rt_decimals,
            include_tolerance_col=include_tolerance_col,
        )

    def _simplify_headers(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Simplify column headers by extracting sample names from paths.

        Examples:
            "Intensity of C:\\...\\program2_program1_TumorBC2257_DNA.tsv" -> "TumorBC2257_DNA"
            "Intensity of C:\\...\\Breast_Cancer_Tissue_pooled_QC1.tsv" -> "pooled_QC1"
        """
        return simplify_headers(df)

    def _extract_sample_name(self, header: str) -> str:
        """
        Extract sample name from a column header (typically a file path).

        Args:
            header: Original column header

        Returns:
            Simplified sample name
        """
        return extract_raw_sample_name(header)

    def _insert_sample_type_row(
        self,
        df: pd.DataFrame,
        header_mapping: Dict[str, str],
        sample_mapping: Dict[str, str],
        sample_type_overrides: Optional[Dict[str, str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Insert Sample_Type row at the top of the data.

        Auto-detects sample types based on column names, unless user-provided
        sample types are available from the input file.
        """
        _ = header_mapping
        return insert_sample_type_row(
            df,
            sample_mapping=sample_mapping,
            sample_type_patterns=self.SAMPLE_TYPE_PATTERNS,
            sample_type_overrides=sample_type_overrides,
        )

    def _detect_sample_type(
        self,
        column_name: str,
        sample_mapping: Dict[str, str],
    ) -> str:
        """
        Detect sample type from column name.

        Priority:
        1. Direct pattern matching from column name (e.g., TumorBC2257 -> tumor)
        2. Pre-defined mapping from method file (fallback)
        3. Default to "sample"

        Args:
            column_name: Name of the column (sample name)
            sample_mapping: Pre-defined mapping from method file

        Returns:
            Detected sample type
        """
        return detect_sample_type(column_name, sample_mapping, self.SAMPLE_TYPE_PATTERNS)

    def _finalize_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Finalize the DataFrame structure.

        Ensures proper data types and formatting.
        """
        return finalize_structure(df)

    def _extract_docx_tables_fallback(self, file_path: Union[str, Path]) -> List[List[List[str]]]:
        """Compatibility wrapper for method sequence DOCX fallback parsing."""
        return extract_docx_tables_fallback(file_path)

    def _parse_injection_volume_from_cells(self, cells: List[str]) -> float:
        """Compatibility wrapper for method sequence injection-volume parsing."""
        return parse_injection_volume_from_cells(cells)

    def _extract_injection_rows_from_table(
        self,
        table_rows: List[List[str]],
        preserve_source_order: bool = False,
    ) -> List[InjectionInfo]:
        """Compatibility wrapper for method sequence row extraction."""
        return extract_injection_rows_from_table(
            table_rows,
            preserve_source_order=preserve_source_order,
        )

    def _parse_injection_sequence(self, file_path: Union[str, Path]) -> List[InjectionInfo]:
        """Compatibility wrapper for method sequence parsing."""
        return parse_injection_sequence(file_path)

    def _simplify_word_sample_name(self, file_name: str) -> str:
        """
        Simplify sample name from Word document to match column headers.

        Examples:
            "Breast Cancer Tissue_ pooled_QC_1" -> "pooled_QC_1"
            "Tumor tissue BC2257_DNA" -> "TumorBC2257_DNA"
            "Normal tissue BC2257_DNA" -> "NormalBC2257_DNA"
            "Benign tissue Fat BC2250_DNA" -> "BenignBC2250_DNA"

        Args:
            file_name: Original sample name from Word document

        Returns:
            Simplified sample name that matches column header
        """
        return simplify_method_sample_name(file_name)

    def _build_sample_info(
        self,
        df: pd.DataFrame,
        injection_info_list: List[InjectionInfo],
    ) -> pd.DataFrame:
        """Compatibility wrapper for SampleInfo construction."""
        return self._sample_info_builder.build(df, injection_info_list)

    def _reorder_columns_by_injection(
        self,
        df: pd.DataFrame,
        sample_info_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Reorder DataFrame columns based on Injection_Order from SampleInfo.

        Args:
            df: DataFrame with sample columns
            sample_info_df: SampleInfo DataFrame with Injection_Order

        Returns:
            DataFrame with columns reordered by Injection_Order
        """
        return reorder_columns_by_injection(df, sample_info_df)

    def _parse_method_file(self, file_path: Union[str, Path]) -> Dict[str, str]:
        """
        Parse method file (Word document) to extract sample type mapping.

        Args:
            file_path: Path to the Word document

        Returns:
            Dictionary mapping sample names to sample types
        """
        return parse_method_file(file_path)

    def _extract_sample_id(self, text: str) -> Optional[str]:
        """
        Extract sample ID from text.

        Examples:
            "Tumor tissue BC2257_DNA" -> "BC2257_DNA"
            "Normal tissue BC2257_DNA" -> "BC2257_DNA"
        """
        return extract_sample_id(text)

    def auto_detect_sample_types(
        self,
        column_names: List[str],
        patterns: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        Auto-detect sample types from column names.

        Args:
            column_names: List of column names to analyze
            patterns: Custom patterns for detection

        Returns:
            Dictionary mapping column names to detected sample types
        """
        return auto_detect_sample_types_from_columns(
            column_names,
            self.SAMPLE_TYPE_PATTERNS,
            patterns,
        )


def load_raw_data(
    file_path: Union[str, Path],
    **kwargs,
) -> pd.DataFrame:
    """
    Load raw mass spectrometry data from file.

    Supports TSV, CSV, and Excel formats.

    Args:
        file_path: Path to the data file
        **kwargs: Additional arguments passed to pandas read functions

    Returns:
        DataFrame with raw data
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() == ".tsv":
        return pd.read_csv(file_path, sep="\t", **kwargs)
    elif file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path, **kwargs)
    elif file_path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(file_path, **kwargs)
    else:
        # Try TSV as default
        return pd.read_csv(file_path, sep="\t", **kwargs)
