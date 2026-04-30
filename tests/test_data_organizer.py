"""Tests for DataOrganizer module."""

import pandas as pd
import pytest
from ms_core.preprocessing.data_organizer import DataOrganizer, InjectionInfo


class TestDataOrganizer:
    """Test cases for DataOrganizer."""

    @pytest.fixture
    def organizer(self):
        """Create a DataOrganizer instance."""
        return DataOrganizer()

    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        data = {
            "Mz": [100.123, 200.456, 300.789],
            "RT": [1.5, 2.5, 3.5],
            "Sample1": [1000, 2000, 3000],
            "Sample2": [1100, 2100, 3100],
            "QC1": [1050, 2050, 3050],
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def normalized_data(self):
        """Create raw input for statistics mode."""
        data = {
            "Mz": [100.1234, 200.5678],
            "RT": [1.5, 2.5],
            r"Intensity of C:\data\program2_program1_NormalBC2257_DNA.tsv": [1000, 1100],
            r"Intensity of C:\data\program2_program1_TumorBC2257_DNA.tsv": [2000, 2100],
            r"Intensity of C:\data\program2_program1_pooled_QC_1.tsv": [1500, 1550],
        }
        return pd.DataFrame(data)

    def test_validate_input_empty(self, organizer):
        """Test validation with empty DataFrame."""
        df = pd.DataFrame()
        is_valid, message = organizer.validate_input(df)
        assert not is_valid
        assert "empty" in message.lower()

    def test_validate_input_valid(self, organizer, sample_data):
        """Test validation with valid DataFrame."""
        is_valid, message = organizer.validate_input(sample_data)
        assert is_valid
        assert message == ""

    def test_auto_detect_sample_types(self, organizer):
        """Test auto detection of sample types."""
        columns = ["QC_1", "QC_2", "Blank_1", "Sample_A", "Sample_B"]
        mapping = organizer.auto_detect_sample_types(columns)

        assert mapping["QC_1"] == "QC"
        assert mapping["QC_2"] == "QC"
        assert mapping["Blank_1"] == "blank"
        assert mapping["Sample_A"] == "sample"
        assert mapping["Sample_B"] == "sample"

    def test_process_basic(self, organizer, sample_data):
        """Test basic processing."""
        result = organizer.process(sample_data)

        assert result.success
        assert result.data is not None
        assert len(result.data) > 0

    def test_process_with_sample_type_mapping(self, organizer, sample_data):
        """Test processing with custom sample type mapping."""
        mapping = {"Sample": "case", "QC": "qc"}
        result = organizer.process(sample_data, sample_type_mapping=mapping)

        assert result.success

    def test_process_uses_input_sample_type_row_when_present(self, organizer):
        """When input already has a Sample Type row, Step1 should preserve it."""
        df = pd.DataFrame(
            {
                "Mz": ["Sample Type", 100.123, 200.456],
                "RT": [None, 1.5, 2.5],
                "Sample1": ["Exposure", 1000, 2000],
                "Sample2": ["Control", 1100, 2100],
                "QC1": ["QC", 1050, 2050],
            }
        )

        result = organizer.process(df)

        assert result.success
        assert result.data.iloc[0, 0] == "Sample_Type"
        assert result.data.iloc[1, 0] == "100.1230/1.50"
        assert result.data.iloc[0]["Sample1"] == "Exposure"
        assert result.data.iloc[0]["Sample2"] == "Control"
        assert result.data.iloc[0]["QC_1"] == "QC"

        sample_info = result.metadata.get("sample_info")
        assert sample_info is not None
        sample_type_by_name = dict(
            zip(sample_info["Sample_Name"], sample_info["Sample_Type"], strict=True)
        )
        assert sample_type_by_name["Sample1"] == "Exposure"
        assert sample_type_by_name["Sample2"] == "Control"
        assert sample_type_by_name["QC_1"] == "QC"

    def test_process_statistics_mode_keeps_mz_rt_and_sorts(self, organizer, normalized_data, monkeypatch):
        """Statistics mode keeps Mz/RT separate while preserving normalization behavior."""
        injection_sequence = [
            InjectionInfo(1, "Tumor tissue BC2257_DNA", "TumorBC2257_DNA", 1.0),
            InjectionInfo(2, "Normal tissue BC2257_DNA", "NormalBC2257_DNA", 1.0),
            InjectionInfo(3, "Breast Cancer Tissue_ pooled_QC_1", "pooled_QC_1", 1.0),
        ]

        monkeypatch.setattr(organizer, "_parse_method_file", lambda _path: {})
        monkeypatch.setattr(organizer, "_parse_injection_sequence", lambda _path: injection_sequence)

        result = organizer.process(
            normalized_data,
            mode="statistics",
            method_file="dummy.docx",
        )

        assert result.success
        assert result.metadata.get("sample_info") is not None
        assert "Mz/RT" not in list(result.data.columns)
        assert list(result.data.columns)[:2] == ["Mz", "RT"]
        assert result.data.iloc[0, 0] == "Sample_Type"
        assert result.data.iloc[0, 1] == "na"
        assert list(result.data.columns)[2:] == [
            "TumorBC2257_DNA",
            "NormalBC2257_DNA",
            "pooled_QC_1",
        ]

    def test_process_statistics_mode_rejects_unknown_mode(self, organizer, sample_data):
        """Unknown mode should fail with a clear message."""
        result = organizer.process(sample_data, mode="unknown_mode")
        assert not result.success
        assert "Unsupported mode" in result.message

    def test_step1_sampleinfo_batch_and_dna_columns_are_empty(self, organizer, sample_data):
        """Step1 should create Batch/DNA columns but keep them empty for manual post-processing."""
        result = organizer.process(sample_data)
        assert result.success
        sample_info = result.metadata.get("sample_info")
        assert sample_info is not None
        assert "Batch" in sample_info.columns
        assert "DNA_mg/20uL" in sample_info.columns
        assert sample_info["Batch"].isna().all()
        assert sample_info["DNA_mg/20uL"].isna().all()

    def test_build_sample_info_maps_ec_u_and_ignores_metadata_columns(self, organizer):
        """SampleInfo should map EC-style names and skip metadata-only trailing columns."""
        df = pd.DataFrame(
            [
                ["Sample_Type", "sample", "sample", "sample"],
                ["100.1000/1.00", 1.0, 2.0, 10],
            ],
            columns=["Mz/RT", "EC013_2", "EC0301", "row ID"],
        )
        injection_info = [
            InjectionInfo(1, "EC013", "EC013", 5.0),
            InjectionInfo(2, "EC301", "EC301", 5.0),
        ]

        sample_info = organizer._build_sample_info(df, injection_info)

        assert list(sample_info["Sample_Name"]) == ["EC013_2", "EC0301"]
        assert list(sample_info["Method_Sample_Name"]) == ["EC013", "EC301"]
        assert list(sample_info["Injection_Order"]) == [1, 2]
        assert "row ID" not in sample_info["Sample_Name"].tolist()

    def test_reorder_columns_keeps_metadata_columns_at_end(self, organizer):
        """Reordering by injection order should move sample columns only."""
        df = pd.DataFrame(
            [
                ["Sample_Type", "Exposure", "Control", 111],
                ["100.1000/1.00", 10.0, 20.0, 222],
            ],
            columns=["Mz/RT", "Sample_A", "Sample_B", "row ID"],
        )
        sample_info = pd.DataFrame(
            {
                "Sample_Name": ["Sample_B", "Sample_A"],
                "Sample_Type": ["Control", "Exposure"],
                "Injection_Order": [1, 2],
                "_col_name": ["Sample_B", "Sample_A"],
            }
        )

        reordered = organizer._reorder_columns_by_injection(df, sample_info)

        assert list(reordered.columns) == ["Mz/RT", "Sample_B", "Sample_A", "row ID"]

    def test_extract_injection_rows_from_dual_column_table(self, organizer):
        """DOCX dual-column sequence rows should be parsed into injection info."""
        table_rows = [
            ["序號", "樣本", "體積", "序號", "樣本", "體積"],
            ["1", "QC_sample_1", "7", "33", "EC046", "85"],
            ["2", "U00001ZBEE", "85", "34", "EC047", "85"],
        ]

        parsed = organizer._extract_injection_rows_from_table(table_rows)

        assert [x.injection_order for x in parsed] == [1, 2, 33, 34]
        assert [x.file_name for x in parsed] == ["QC_sample_1", "U00001ZBEE", "EC046", "EC047"]

    def test_extract_primary_sample_token_handles_pooled_qc(self, organizer):
        """Pooled QC naming should be recognized from breast-cancer method rows."""
        token = organizer._extract_primary_sample_token("Breast Cancer Tissue_ pooled_QC_1")
        assert token is not None
        assert "pooled_qc_1" in token.lower()

    def test_extract_injection_rows_preserve_source_order(self, organizer):
        """When preserving source order, rows should be renumbered by appearance."""
        table_rows = [
            ["序號", "樣本", "體積"],
            ["45", "Normal tissue BC2258_DNA", "20"],
            ["46", "Tumor tissue BC2312_DNA", "20"],
            ["46", "Normal tissue BC2259_DNA", "20"],
            ["47", "Tumor tissue BC2313_DNA", "20"],
        ]

        parsed = organizer._extract_injection_rows_from_table(
            table_rows,
            preserve_source_order=True,
        )

        assert [x.file_name for x in parsed] == [
            "Normal tissue BC2258_DNA",
            "Tumor tissue BC2312_DNA",
            "Normal tissue BC2259_DNA",
            "Tumor tissue BC2313_DNA",
        ]
        assert [x.injection_order for x in parsed] == [1, 2, 3, 4]

    def test_extract_sample_name_removes_dna_program_prefix(self, organizer):
        """RawIntensity sample headers should not keep technical DNA_program prefixes."""
        header = r"Intensity of C:\data\DNA_program1_TumorBC2257_DNA.tsv"
        assert organizer._extract_sample_name(header) == "TumorBC2257_DNA"

    def test_extract_sample_name_removes_program_then_dna_prefix(self, organizer):
        """Compound prefixes like program2_DNA_program1_ should be removed."""
        header = r"Intensity of C:\data\program2_DNA_program1_TumorBC2257_DNA.tsv"
        assert organizer._extract_sample_name(header) == "TumorBC2257_DNA"

    # ------------------------------------------------------------------
    # Pre-merged Mz/RT input (file already has combined "Mz/RT" column)
    # ------------------------------------------------------------------

    @pytest.fixture
    def pre_merged_data(self):
        """Input that already has a combined Mz/RT column (e.g. from VBA processing)."""
        data = {
            "Mz/RT": [
                "Sample_Type",
                "359.045013/2.72",
                "274.092012345/18.3267",
                "100.1234/5.00",
            ],
            "QC_1": ["QC", 2024319.647, 1234567.0, 999999.0],
            "TumorBC01": ["Exposure", None, 8660497.854, 111111.0],
        }
        return pd.DataFrame(data)

    def test_validate_input_accepts_pre_merged_mz_rt(self, organizer, pre_merged_data):
        """validate_input should accept a file whose first column is already Mz/RT."""
        is_valid, message = organizer.validate_input(pre_merged_data)
        assert is_valid, message

    def test_expand_pre_merged_mz_rt_splits_values(self, organizer, pre_merged_data):
        """_expand_pre_merged_mz_rt should split parseable rows into Mz and RT floats."""
        # Remove Sample_Type row first (as the pipeline does)
        data_df = pre_merged_data.iloc[1:].reset_index(drop=True)
        expanded, was_expanded = organizer._expand_pre_merged_mz_rt(data_df)
        assert was_expanded
        assert list(expanded.columns[:2]) == ["Mz", "RT"]
        assert abs(expanded.iloc[0, 0] - 359.045013) < 1e-6
        assert abs(expanded.iloc[0, 1] - 2.72) < 1e-6

    def test_process_normalizes_pre_merged_mz_rt_precision(self, organizer, pre_merged_data):
        """Step 1 must reformat Mz/RT to 4-decimal/2-decimal standard precision."""
        result = organizer.process(pre_merged_data)
        assert result.success, result.message
        data_rows = [
            v for v in result.data.iloc[:, 0].tolist()
            if isinstance(v, str) and "/" in v and v != "Sample_Type"
        ]
        assert len(data_rows) > 0
        # 359.045013/2.72 → 359.0450/2.72
        assert data_rows[0] == "359.0450/2.72"
        # 274.092012345/18.3267 → 274.0920/18.33
        assert data_rows[1] == "274.0920/18.33"

    def test_expand_pre_merged_ignores_non_mz_rt_column(self, organizer):
        """_expand_pre_merged_mz_rt must leave standard Mz/RT columns unchanged."""
        df = pd.DataFrame({"Mz": [100.0, 200.0], "RT": [1.5, 2.5], "S1": [1, 2]})
        expanded, was_expanded = organizer._expand_pre_merged_mz_rt(df)
        assert not was_expanded
        assert list(expanded.columns) == ["Mz", "RT", "S1"]
