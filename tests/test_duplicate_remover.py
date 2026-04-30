"""Tests for DuplicateRemover module."""

import pandas as pd
import pytest
from ms_core.preprocessing.duplicate_remover import DuplicateRemover


class TestDuplicateRemover:
    """Test cases for DuplicateRemover."""

    @pytest.fixture
    def remover(self):
        """Create a DuplicateRemover instance."""
        return DuplicateRemover()

    @pytest.fixture
    def sample_data_with_duplicates(self):
        """Create sample data with duplicate signals."""
        data = {
            "Mz/RT": [
                "Sample_Type",
                "100.1234/1.50",  # Original
                "100.1235/1.51",  # Duplicate (within tolerance)
                "200.5678/2.50",  # Unique
                "300.9999/3.50",  # Unique
            ],
            "Tolerance": ["na", "na", "na", "na", "na"],
            "Sample1": ["case", 5000, 4000, 6000, 7000],
            "Sample2": ["case", 5500, 4500, 6500, 7500],
        }
        return pd.DataFrame(data)

    def test_validate_input_empty(self, remover):
        """Test validation with empty DataFrame."""
        df = pd.DataFrame()
        is_valid, message = remover.validate_input(df)
        assert not is_valid

    def test_validate_input_valid(self, remover, sample_data_with_duplicates):
        """Test validation with valid DataFrame."""
        is_valid, message = remover.validate_input(sample_data_with_duplicates)
        assert is_valid

    def test_detect_columns_combined_format(self, remover, sample_data_with_duplicates):
        """Test column detection with combined m/z/RT format."""
        col_info = remover._detect_columns(sample_data_with_duplicates)

        assert col_info["combined_mz_rt"] is True
        assert col_info["feature_col"] == "Mz/RT"

    def test_find_duplicate_groups(self, remover, sample_data_with_duplicates):
        """Test finding duplicate groups."""
        groups = remover.get_duplicate_groups(
            sample_data_with_duplicates,
            mz_tolerance_ppm=20,
            rt_tolerance=0.1,
        )

        # Should find one duplicate group (first two data rows)
        assert len(groups) >= 1

    def test_process_removes_duplicates(self, remover, sample_data_with_duplicates):
        """Test that processing removes duplicates."""
        result = remover.process(
            sample_data_with_duplicates,
            mz_tolerance_ppm=20,
            rt_tolerance=0.1,
        )

        assert result.success
        # Original has 4 data rows, should be reduced
        original_data_rows = len(sample_data_with_duplicates) - 1
        result_data_rows = len(result.data) - 1
        assert result_data_rows <= original_data_rows

    def test_process_preserves_highest_intensity(self, remover, sample_data_with_duplicates):
        """Test that processing preserves highest intensity signal."""
        result = remover.process(
            sample_data_with_duplicates,
            mz_tolerance_ppm=20,
            rt_tolerance=0.1,
        )

        assert result.success
        result_values = result.data.iloc[1:]["Mz/RT"].tolist()
        assert "100.1234/1.50" in result_values, (
            f"Higher-intensity feature should be kept. Got: {result_values}"
        )
        assert "100.1235/1.51" not in result_values, (
            f"Lower-intensity duplicate should be removed. Got: {result_values}"
        )

    def test_keeps_higher_occurrence_over_higher_intensity(self, remover):
        """高出現率 feature 應被保留，即使強度較低。"""
        data = {
            "Mz/RT": [
                "Sample_Type",
                "258.109431/9.042764",  # Feature A: occurrence=7, total=202000 → 應保留
                "258.109432/9.043000",  # Feature B: occurrence=4, total=203000 → 應刪除
            ],
            "Tolerance": ["na", "na", "na"],
            "Case1": ["case", 30000, 50000],
            "Case2": ["case", 28000, 48000],
            "Case3": ["case", 32000, 52000],
            "Control1": ["control", 25000, 0],
            "Control2": ["control", 27000, 0],
            "Control3": ["control", 29000, 0],
            "QC1": ["qc", 31000, 51000],
        }
        df = pd.DataFrame(data)
        result = remover.process(df, mz_tolerance_ppm=20, rt_tolerance=0.1)

        assert result.success
        assert len(result.data) - 1 == 1, (
            f"Expected 1 feature after dedup, got {len(result.data) - 1}"
        )
        kept = result.data.iloc[1]["Mz/RT"]
        assert kept == "258.109431/9.042764", (
            f"Higher-occurrence feature A should be kept, got: {kept}"
        )

    def test_distant_rt_feature_not_merged_with_close_duplicates(self, remover):
        """使用 rt_tolerance=0.1 時，RT 差距大的 feature 不應與近距離 duplicates 合併。"""
        data = {
            "Mz/RT": [
                "Sample_Type",
                "258.109431/8.100000",  # C: RT 遠端 (9.0-8.1=0.9 min), occurrence=5
                "258.109431/9.042764",  # A: occurrence=5，應在 A/B 中勝出
                "258.109432/9.043000",  # B: occurrence=3，應被 A 刪除
            ],
            "Case1": ["case", 80000, 30000, 10000],
            "Case2": ["case", 82000, 28000, 0],
            "Case3": ["case", 78000, 32000, 0],
            "QC1": ["qc", 79000, 26000, 9000],
            "QC2": ["qc", 81000, 27000, 8000],
        }
        df = pd.DataFrame(data)
        result = remover.process(df, mz_tolerance_ppm=20, rt_tolerance=0.1)

        assert result.success
        result_mz_rt = set(result.data.iloc[1:]["Mz/RT"].tolist())
        assert len(result.data) - 1 == 2, (
            f"Expected 2 features (C + A), got {len(result.data) - 1}: {result_mz_rt}"
        )
        assert "258.109431/8.100000" in result_mz_rt, (
            "Feature C (distant RT) should not be merged with A/B"
        )
        assert "258.109431/9.042764" in result_mz_rt, (
            "Higher-occurrence Feature A should survive A/B dedup"
        )

    def test_istd_duplicates_are_deduplicated_keeping_highest_occurrence(self, remover):
        """ISTD (protected) rows 之間也要去重，保留檢出率最高的代表列。"""
        data = {
            "Mz/RT": [
                "Sample_Type",
                "261.1273/5.10",  # ISTD-A: occurrence=5 → 應保留
                "261.1274/5.11",  # ISTD-B: occurrence=2 → 應刪除（同為 protected，但檢出率低）
                "400.0000/15.00",  # 普通 feature，不受影響
            ],
            "Sample1": ["case", 60000, 20000, 5000],
            "Sample2": ["case", 58000, 22000, 4800],
            "Sample3": ["control", 62000, 0, 5100],
            "QC1": ["qc", 59000, 0, 4900],
            "QC2": ["qc", 61000, 21000, 5050],
        }
        df = pd.DataFrame(data)
        # row index 1 = ISTD-A, row index 2 = ISTD-B，兩者皆標為 protected
        protected = {1, 2}
        result = remover.process(
            df, mz_tolerance_ppm=20, rt_tolerance=0.1, protected_rows=protected
        )

        assert result.success
        result_mz_rt = set(result.data.iloc[1:]["Mz/RT"].tolist())
        assert len(result.data) - 1 == 2, (
            f"Expected 2 features (best ISTD + normal), got {len(result.data) - 1}: {result_mz_rt}"
        )
        assert "261.1273/5.10" in result_mz_rt, "Higher-occurrence ISTD-A should be kept"
        assert "261.1274/5.11" not in result_mz_rt, "Lower-occurrence ISTD-B should be removed"
        assert "400.0000/15.00" in result_mz_rt, "Unrelated feature should be untouched"

    def test_protected_rows_keep_representative_but_allow_per_sample_max_upgrade(self, remover):
        """Protected rows lock representative selection, not original sample intensities."""
        data = {
            "Mz/RT": [
                "Sample_Type",
                "261.1273/5.10",  # Protected representative
                "261.1274/5.11",  # Donor has stronger overlap on Sample1
            ],
            "Sample1": ["case", 1000, 4500],
            "Sample2": ["case", 800, 700],
            "Sample3": ["case", 600, 0],
        }
        df = pd.DataFrame(data)
        result = remover.process(
            df,
            mz_tolerance_ppm=20,
            rt_tolerance=0.1,
            protected_rows={1},
        )

        assert result.success
        assert len(result.data) - 1 == 1

        merged_row = result.data.iloc[1]
        assert merged_row["Mz/RT"] == "261.1273/5.10", "Protected row should remain the representative"
        assert float(merged_row["Sample1"]) == 4500, "per_sample_max may upgrade protected representative values"
        assert float(merged_row["Sample2"]) == 800
        assert float(merged_row["Sample3"]) == 600
        assert result.statistics["data_points_upgraded"] == 1

    def test_invalid_merge_mode_raises_value_error(self, remover):
        """Invalid merge mode should fail fast for direct API callers."""
        data = {
            "Mz/RT": ["Sample_Type", "250.080914/8.571", "250.080914/8.571"],
            "Sample1": ["case", 1000, 2000],
        }
        df = pd.DataFrame(data)

        with pytest.raises(ValueError, match="Unsupported merge_mode"):
            remover.process(df, mz_tolerance_ppm=20, rt_tolerance=0.1, merge_mode="sum")

    def test_merge_recovers_complementary_data_points(self, remover):
        """Duplicate rows with complementary sample coverage should be merged, not discarded."""
        data = {
            "Mz/RT": [
                "Sample_Type",
                "250.080914/8.571",  # Row A: has Sample1-3
                "250.080914/8.571",  # Row B: has Sample3-5 (overlaps on Sample3, adds Sample4-5)
            ],
            "Sample1": ["case", 1000, 0],
            "Sample2": ["case", 2000, 0],
            "Sample3": ["case", 3000, 3000],
            "Sample4": ["case", 0, 4000],
            "Sample5": ["case", 0, 5000],
        }
        df = pd.DataFrame(data)
        result = remover.process(df, mz_tolerance_ppm=20, rt_tolerance=0.1)

        assert result.success
        assert len(result.data) - 1 == 1, "Should merge into 1 feature"

        merged_row = result.data.iloc[1]
        assert float(merged_row["Sample1"]) == 1000, "Sample1 from Row A preserved"
        assert float(merged_row["Sample2"]) == 2000, "Sample2 from Row A preserved"
        assert float(merged_row["Sample3"]) == 3000, "Sample3 keeps best row value"
        assert float(merged_row["Sample4"]) == 4000, "Sample4 recovered from Row B"
        assert float(merged_row["Sample5"]) == 5000, "Sample5 recovered from Row B"

        assert result.statistics["data_points_recovered"] == 2
        assert result.statistics["groups_merged"] >= 1

    def test_merge_uses_per_sample_max_by_default(self, remover):
        """Default merge policy should keep the highest overlapping sample value."""
        data = {
            "Mz/RT": [
                "Sample_Type",
                "250.080914/8.571",  # Representative row chosen by higher occurrence
                "250.080914/8.571",  # Donor row has stronger overlapping Sample2 value
            ],
            "Sample1": ["case", 1000, 0],
            "Sample2": ["case", 2000, 4500],
            "Sample3": ["case", 3000, 2800],
        }
        df = pd.DataFrame(data)
        result = remover.process(df, mz_tolerance_ppm=20, rt_tolerance=0.1)

        assert result.success
        assert len(result.data) - 1 == 1

        merged_row = result.data.iloc[1]
        assert float(merged_row["Sample1"]) == 1000
        assert float(merged_row["Sample2"]) == 4500, "Overlapping sample should keep donor max"
        assert float(merged_row["Sample3"]) == 3000, "Lower donor overlap should not overwrite best row"
        assert result.statistics["data_points_recovered"] == 0
        assert result.statistics["data_points_upgraded"] == 1

    def test_merge_fill_gaps_mode_preserves_legacy_overlap_behavior(self, remover):
        """Legacy fill_gaps mode should only recover missing values, not overwrite overlaps."""
        data = {
            "Mz/RT": [
                "Sample_Type",
                "250.080914/8.571",
                "250.080914/8.571",
            ],
            "Sample1": ["case", 1000, 0],
            "Sample2": ["case", 2000, 4500],
            "Sample3": ["case", 1800, 2800],
            "Sample4": ["case", 0, 900],
            "Sample5": ["case", 700, 0],
        }
        df = pd.DataFrame(data)
        result = remover.process(
            df,
            mz_tolerance_ppm=20,
            rt_tolerance=0.1,
            merge_mode="fill_gaps",
        )

        assert result.success
        assert len(result.data) - 1 == 1

        merged_row = result.data.iloc[1]
        assert float(merged_row["Sample1"]) == 1000
        assert float(merged_row["Sample2"]) == 2000, "Legacy mode should keep original overlap value"
        assert float(merged_row["Sample3"]) == 1800, "Legacy mode should keep original overlap value"
        assert float(merged_row["Sample4"]) == 900, "Legacy mode should still fill missing values"
        assert float(merged_row["Sample5"]) == 700
        assert result.statistics["data_points_recovered"] == 1
        assert result.statistics["data_points_upgraded"] == 0

    def test_merge_four_way_duplicate_union(self, remover):
        """Four-way duplicate (like FH multi-trigger) should merge all coverage."""
        data = {
            "Mz/RT": [
                "Sample_Type",
                "250.080914/8.571",  # Row A: 5/5 samples
                "250.080914/8.571",  # Row B: 3/5 samples
                "250.080914/8.571",  # Row C: 1/5 samples
                "250.080914/8.571",  # Row D: 0/5 samples (empty)
            ],
            "S1": ["case", 100, 100, 0, 0],
            "S2": ["case", 200, 200, 0, 0],
            "S3": ["case", 300, 0, 0, 0],
            "S4": ["case", 400, 0, 400, 0],
            "S5": ["case", 500, 500, 0, 0],
        }
        df = pd.DataFrame(data)
        result = remover.process(df, mz_tolerance_ppm=20, rt_tolerance=0.1)

        assert result.success
        assert len(result.data) - 1 == 1
        merged = result.data.iloc[1]
        # Row A already has all 5, so no recovery needed in this case
        for col in ["S1", "S2", "S3", "S4", "S5"]:
            assert float(merged[col]) > 0, f"{col} should have data after merge"

    def test_merge_does_not_affect_non_duplicates(self, remover):
        """Non-duplicate features should pass through unchanged."""
        data = {
            "Mz/RT": [
                "Sample_Type",
                "100.0000/1.00",
                "200.0000/2.00",
                "300.0000/3.00",
            ],
            "S1": ["case", 100, 200, 300],
            "S2": ["case", 110, 0, 310],
        }
        df = pd.DataFrame(data)
        result = remover.process(df, mz_tolerance_ppm=20, rt_tolerance=0.1)

        assert result.success
        assert len(result.data) - 1 == 3
        assert result.statistics.get("data_points_recovered", 0) == 0

    def test_degeneracy_annotation_is_disabled_by_default(self, remover):
        df = pd.DataFrame(
            {
                "Mz/RT": ["Sample_Type", "242.1191/10.99", "264.1010/10.99"],
                "Sample1": ["case", 50000, 38000],
                "QC1": ["qc", 49000, 37000],
            }
        )

        result = remover.process(df, mz_tolerance_ppm=20, rt_tolerance=0.1)

        assert result.success
        assert "Degeneracy_Type" not in result.data.columns
        assert result.statistics["degeneracy_annotation_enabled"] is False
        assert result.statistics["degeneracy_adduct_count"] == 0

    def test_degeneracy_annotation_marks_base_and_adduct(self, remover):
        df = pd.DataFrame(
            {
                "Mz/RT": ["Sample_Type", "242.1191/10.99", "264.1010/11.00", "300.0000/20.00"],
                "Case1": ["case", 100, 50, 20],
                "Case2": ["case", 200, 100, 25],
                "Case3": ["case", 400, 200, 22],
                "Case4": ["case", 800, 400, 19],
                "QC1": ["qc", 51000, 38500, 11500],
            }
        )

        result = remover.process(
            df,
            mz_tolerance_ppm=5,
            rt_tolerance=0.1,
            enable_degeneracy_annotation=True,
            degeneracy_ppm_tolerance=20,
            degeneracy_rt_tolerance=0.05,
        )

        assert result.success
        assert "Degeneracy_Type" in result.data.columns
        assert result.statistics["degeneracy_annotation_enabled"] is True
        assert result.statistics["degeneracy_adduct_count"] == 1
        assert result.statistics["degeneracy_base_count"] == 1
        assert result.statistics["degeneracy_corr_rejected"] == 0

        base_row = result.data[result.data["Mz/RT"] == "242.1191/10.99"].iloc[0]
        adduct_row = result.data[result.data["Mz/RT"] == "264.1010/11.00"].iloc[0]
        singleton_row = result.data[result.data["Mz/RT"] == "300.0000/20.00"].iloc[0]

        assert base_row["Degeneracy_Group_Role"] == "base"
        assert base_row["Degeneracy_Type"] == "[M+H]+"
        assert base_row["Degeneracy_Group_ID"] == "DG0001"
        assert adduct_row["Degeneracy_Group_Role"] == "adduct"
        assert adduct_row["Degeneracy_Type"] == "[M+Na]+"
        assert adduct_row["Degeneracy_Base_mz"] == "242.1191"
        assert float(adduct_row["Degeneracy_Pearson_R"]) >= 0.99
        assert singleton_row["Degeneracy_Group_Role"] == "singleton"

    def test_degeneracy_annotation_rejects_low_correlation_pairs(self, remover):
        df = pd.DataFrame(
            {
                "Mz/RT": ["Sample_Type", "242.1191/10.99", "264.1010/11.00"],
                "Case1": ["case", 100, 400],
                "Case2": ["case", 200, 300],
                "Case3": ["case", 300, 200],
                "Case4": ["case", 400, 100],
            }
        )

        result = remover.process(
            df,
            enable_degeneracy_annotation=True,
            degeneracy_ppm_tolerance=20,
            degeneracy_rt_tolerance=0.05,
            degeneracy_correlation_threshold=0.8,
            degeneracy_min_correlation_points=3,
        )

        assert result.success
        assert result.statistics["degeneracy_adduct_count"] == 0
        assert result.statistics["degeneracy_corr_rejected"] >= 1
        pair_row = result.data[result.data["Mz/RT"] == "264.1010/11.00"].iloc[0]
        assert pair_row["Degeneracy_Group_Role"] == "singleton"
