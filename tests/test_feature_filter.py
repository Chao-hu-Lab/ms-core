"""Tests for FeatureFilter module."""

import pandas as pd
import pytest
from ms_core.preprocessing.ms_quality_filter import FeatureFilter


class TestFeatureFilter:
    """Test cases for FeatureFilter."""

    @pytest.fixture
    def filter_proc(self):
        """Create a FeatureFilter instance."""
        return FeatureFilter()

    @pytest.fixture
    def sample_data(self):
        """Create sample test data with proper structure."""
        data = {
            "Mz/RT": ["Sample_Type", "100.123/1.5", "200.456/2.5", "300.789/3.5"],
            "Tolerance": ["na", "na", "na", "na"],
            "Case1": ["case", 10000, 8000, 100],  # Signal above threshold
            "Case2": ["case", 9000, 7500, 200],
            "Control1": ["control", 500, 8500, 9000],
            "Control2": ["control", 400, 8000, 9500],
            "QC1": ["qc", 9500, 8200, 5000],
        }
        return pd.DataFrame(data)

    def test_validate_input_empty(self, filter_proc):
        """Test validation with empty DataFrame."""
        df = pd.DataFrame()
        is_valid, message = filter_proc.validate_input(df)
        assert not is_valid

    def test_validate_input_valid(self, filter_proc, sample_data):
        """Test validation with valid DataFrame."""
        is_valid, message = filter_proc.validate_input(sample_data)
        assert is_valid

    def test_detect_sample_types(self, filter_proc, sample_data):
        """Test sample type detection."""
        group_info = filter_proc._detect_sample_types(sample_data)

        assert "case" in group_info["groups"]
        assert "control" in group_info["groups"]
        assert group_info["has_qc"] is True
        assert len(group_info["qc_cols"]) == 1

    def test_get_group_summary(self, filter_proc, sample_data):
        """Test getting group summary."""
        summary = filter_proc.get_group_summary(sample_data)

        assert "groups" in summary
        assert summary["has_qc"] is True
        assert summary["qc_count"] == 1

    def test_process_filters_features(self, filter_proc, sample_data):
        """Test that processing filters features correctly."""
        result = filter_proc.process(
            sample_data,
            background_threshold=0.33,
        )

        assert result.success
        assert result.data is not None

    def test_max_ratio_diff_calculation(self, filter_proc):
        """Test max ratio difference calculation."""
        ratios = [0.1, 0.5, 0.9]
        max_diff = filter_proc._get_max_ratio_diff(ratios)

        # Max diff should be 0.9 - 0.1 = 0.8
        assert abs(max_diff - 0.8) < 0.001

    def test_qc_ratio_threshold_filters_low_qc_features(self, filter_proc):
        """Features with non-zero but low QC_ratio should be removed when threshold is set."""
        df = pd.DataFrame(
            {
                "Mz/RT": ["Sample_Type", "100.000/1.0", "200.000/2.0"],
                "Tolerance": ["na", "na", "na"],
                "Case1": ["case", 8000, 8000],
                "Case2": ["case", 8000, 8000],
                "Case3": ["case", 8000, 8000],
                "Case4": ["case", 8000, 8000],
                "Case5": ["case", 8000, 8000],
                "Case6": ["case", 8000, 8000],
                "Case7": ["case", 8000, 8000],
                "Case8": ["case", 8000, 8000],
                "Case9": ["case", 8000, 8000],
                "Case10": ["case", 8000, 8000],
                "Control1": ["control", 8000, 8000],
                "Control2": ["control", 8000, 8000],
                "Control3": ["control", 8000, 8000],
                "Control4": ["control", 8000, 8000],
                "Control5": ["control", 8000, 8000],
                "Control6": ["control", 8000, 8000],
                "Control7": ["control", 8000, 8000],
                "Control8": ["control", 8000, 8000],
                "Control9": ["control", 8000, 8000],
                "Control10": ["control", 8000, 8000],
                "QC1": ["qc", 8000, 8000],
                "QC2": ["qc", 100, 8000],
            }
        )

        result = filter_proc.process(
            df,
            background_threshold=0.33,
            qc_ratio_threshold=0.60,
        )

        assert result.success
        assert result.data is not None
        assert "100.000/1.0" not in result.data["Mz/RT"].tolist()
        assert "200.000/2.0" in result.data["Mz/RT"].tolist()
        assert result.statistics.get("qc_low_deleted", 0) >= 1

    def test_qc_ratio_threshold_zero_keeps_legacy_behavior(self, filter_proc):
        """qc_ratio_threshold=0 should keep legacy behavior (only QC_ratio==0 forced delete)."""
        df = pd.DataFrame(
            {
                "Mz/RT": ["Sample_Type", "100.000/1.0"],
                "Tolerance": ["na", "na"],
                "Case1": ["case", 8000],
                "Case2": ["case", 8000],
                "Case3": ["case", 8000],
                "Case4": ["case", 8000],
                "Case5": ["case", 8000],
                "Case6": ["case", 8000],
                "Case7": ["case", 8000],
                "Case8": ["case", 8000],
                "Case9": ["case", 8000],
                "Case10": ["case", 8000],
                "Control1": ["control", 8000],
                "Control2": ["control", 8000],
                "Control3": ["control", 8000],
                "Control4": ["control", 8000],
                "Control5": ["control", 8000],
                "Control6": ["control", 8000],
                "Control7": ["control", 8000],
                "Control8": ["control", 8000],
                "Control9": ["control", 8000],
                "Control10": ["control", 8000],
                "QC1": ["qc", 8000],
                "QC2": ["qc", 100],
            }
        )

        result = filter_proc.process(
            df,
            background_threshold=0.33,
            qc_ratio_threshold=0.0,
        )

        assert result.success
        assert result.data is not None
        assert "100.000/1.0" in result.data["Mz/RT"].tolist()
        assert result.statistics.get("qc_low_deleted", 0) == 0

    def test_disabling_background_rule_removes_feature_kept_only_by_stable_ratio(self, filter_proc):
        df = pd.DataFrame(
            {
                "Mz/RT": ["Sample_Type", "100.000/1.0"],
                "Tolerance": ["na", "na"],
                "Case1": ["case", 8000],
                "Case2": ["case", 8000],
                "Case3": ["case", 8000],
                "Case4": ["case", 8000],
                "Case5": ["case", 8000],
                "Case6": ["case", 100],
                "Case7": ["case", 100],
                "Case8": ["case", 100],
                "Case9": ["case", 100],
                "Case10": ["case", 100],
                "Control1": ["control", 8000],
                "Control2": ["control", 8000],
                "Control3": ["control", 8000],
                "Control4": ["control", 8000],
                "Control5": ["control", 8000],
                "Control6": ["control", 100],
                "Control7": ["control", 100],
                "Control8": ["control", 100],
                "Control9": ["control", 100],
                "Control10": ["control", 100],
                "QC1": ["qc", 8000],
            }
        )

        enabled = filter_proc.process(
            df,
            background_threshold=0.33,
            qc_ratio_threshold=0.0,
            enable_intensity_fc_threshold=False,
        )
        disabled = filter_proc.process(
            df,
            background_threshold=0.33,
            qc_ratio_threshold=0.0,
            enable_background_threshold=False,
            enable_intensity_fc_threshold=False,
        )

        assert enabled.success
        assert disabled.success
        assert "100.000/1.0" in enabled.data["Mz/RT"].tolist()
        assert "100.000/1.0" not in disabled.data["Mz/RT"].tolist()

    def test_mnar_gate_keeps_presence_absence_feature(self, filter_proc):
        """MNAR gate should keep a feature where case is fully detected, control is absent."""
        # case_ratio = 1.0 (all 10 samples above 5000) → ≥ 0.8 → high
        # control_ratio = 0.0 (all 10 samples below 5000) → ≤ 0.2 → low
        # → MNAR gate passes; stable gate fails (only 1 group ≥ 0.33)
        df = pd.DataFrame(
            {
                "Mz/RT": ["Sample_Type", "100.000/1.0"],
                "Tolerance": ["na", "na"],
                "Case1": ["case", 8000],
                "Case2": ["case", 9000],
                "Case3": ["case", 8000],
                "Case4": ["case", 9000],
                "Case5": ["case", 8000],
                "Case6": ["case", 9000],
                "Case7": ["case", 8000],
                "Case8": ["case", 9000],
                "Case9": ["case", 8000],
                "Case10": ["case", 9000],
                "Control1": ["control", 100],
                "Control2": ["control", 200],
                "Control3": ["control", 100],
                "Control4": ["control", 200],
                "Control5": ["control", 100],
                "Control6": ["control", 200],
                "Control7": ["control", 100],
                "Control8": ["control", 200],
                "Control9": ["control", 100],
                "Control10": ["control", 200],
                "QC1": ["qc", 8000],
            }
        )

        result = filter_proc.process(
            df,
            background_threshold=0.33,
            high_det_thresh=0.8,
            low_det_thresh=0.2,
            qc_ratio_threshold=0.0,
            enable_background_threshold=False,
            enable_intensity_fc_threshold=False,
        )

        assert result.success
        assert "100.000/1.0" in result.data["Mz/RT"].tolist()
        assert result.statistics.get("mnar_kept", 0) >= 1

    def test_mnar_feature_overrides_qc_zero_gate_only_for_presence_absence_marker(
        self, filter_proc
    ):
        """MNAR markers should survive QC_ratio==0, while non-MNAR features should still be deleted."""
        df = pd.DataFrame(
            {
                "Mz/RT": ["Sample_Type", "100.000/1.0", "200.000/2.0"],
                "Tolerance": ["na", "na", "na"],
                "Case1": ["case", 8000, 8000],
                "Case2": ["case", 9000, 9000],
                "Case3": ["case", 8000, 8000],
                "Case4": ["case", 9000, 9000],
                "Case5": ["case", 8000, 8000],
                "Case6": ["case", 9000, 9000],
                "Case7": ["case", 8000, 8000],
                "Case8": ["case", 9000, 9000],
                "Case9": ["case", 8000, 8000],
                "Case10": ["case", 9000, 9000],
                "Control1": ["control", 100, 8200],
                "Control2": ["control", 200, 9100],
                "Control3": ["control", 100, 8200],
                "Control4": ["control", 200, 9100],
                "Control5": ["control", 100, 8200],
                "Control6": ["control", 200, 9100],
                "Control7": ["control", 100, 8200],
                "Control8": ["control", 200, 9100],
                "Control9": ["control", 100, 8200],
                "Control10": ["control", 200, 9100],
                "QC1": ["qc", 100, 100],
                "QC2": ["qc", 200, 200],
            }
        )

        result = filter_proc.process(
            df,
            background_threshold=0.33,
            high_det_thresh=0.8,
            low_det_thresh=0.2,
            qc_ratio_threshold=0.0,
            enable_intensity_fc_threshold=False,
        )

        assert result.success
        assert result.data is not None
        assert "100.000/1.0" in result.data["Mz/RT"].tolist()
        assert "200.000/2.0" not in result.data["Mz/RT"].tolist()
        assert result.statistics.get("qc_zero_deleted", 0) == 1

    def test_mnar_feature_overrides_low_qc_ratio_gate_only_for_presence_absence_marker(
        self, filter_proc
    ):
        """MNAR markers should survive low non-zero QC_ratio, while non-MNAR features should still be deleted."""
        df = pd.DataFrame(
            {
                "Mz/RT": ["Sample_Type", "100.000/1.0", "200.000/2.0"],
                "Tolerance": ["na", "na", "na"],
                "Case1": ["case", 8000, 8000],
                "Case2": ["case", 9000, 9000],
                "Case3": ["case", 8000, 8000],
                "Case4": ["case", 9000, 9000],
                "Case5": ["case", 8000, 8000],
                "Case6": ["case", 9000, 9000],
                "Case7": ["case", 8000, 8000],
                "Case8": ["case", 9000, 9000],
                "Case9": ["case", 8000, 8000],
                "Case10": ["case", 9000, 9000],
                "Control1": ["control", 100, 8200],
                "Control2": ["control", 200, 9100],
                "Control3": ["control", 100, 8200],
                "Control4": ["control", 200, 9100],
                "Control5": ["control", 100, 8200],
                "Control6": ["control", 200, 9100],
                "Control7": ["control", 100, 8200],
                "Control8": ["control", 200, 9100],
                "Control9": ["control", 100, 8200],
                "Control10": ["control", 200, 9100],
                "QC1": ["qc", 8000, 8000],
                "QC2": ["qc", 100, 100],
            }
        )

        result = filter_proc.process(
            df,
            background_threshold=0.33,
            high_det_thresh=0.8,
            low_det_thresh=0.2,
            qc_ratio_threshold=0.75,
            enable_intensity_fc_threshold=False,
        )

        assert result.success
        assert result.data is not None
        assert "100.000/1.0" in result.data["Mz/RT"].tolist()
        assert "200.000/2.0" not in result.data["Mz/RT"].tolist()
        assert result.statistics.get("qc_low_deleted", 0) == 1

    def test_disabling_qc_rule_keeps_feature_even_when_qc_ratio_is_zero(self, filter_proc):
        df = pd.DataFrame(
            {
                "Mz/RT": ["Sample_Type", "100.000/1.0"],
                "Tolerance": ["na", "na"],
                "Case1": ["case", 8000],
                "Case2": ["case", 9000],
                "Control1": ["control", 8100],
                "Control2": ["control", 9200],
                "QC1": ["qc", 100],
            }
        )

        enabled = filter_proc.process(
            df,
            background_threshold=0.33,
            qc_ratio_threshold=0.0,
        )
        disabled = filter_proc.process(
            df,
            background_threshold=0.33,
            qc_ratio_threshold=0.0,
            enable_qc_ratio_threshold=False,
        )

        assert enabled.success
        assert disabled.success
        assert "100.000/1.0" not in enabled.data["Mz/RT"].tolist()
        assert "100.000/1.0" in disabled.data["Mz/RT"].tolist()
        assert disabled.metadata["enabled_thresholds"]["qc_ratio"] is False

    def test_disabling_qc_rule_ignores_nonzero_qc_threshold_cutoff(self, filter_proc):
        df = pd.DataFrame(
            {
                "Mz/RT": ["Sample_Type", "100.000/1.0"],
                "Tolerance": ["na", "na"],
                "Case1": ["case", 8000],
                "Case2": ["case", 9000],
                "Control1": ["control", 8100],
                "Control2": ["control", 9200],
                "QC1": ["qc", 8000],
                "QC2": ["qc", 100],
            }
        )

        enabled = filter_proc.process(
            df,
            background_threshold=0.33,
            qc_ratio_threshold=0.75,
        )
        disabled = filter_proc.process(
            df,
            background_threshold=0.33,
            qc_ratio_threshold=0.75,
            enable_qc_ratio_threshold=False,
        )

        assert enabled.success
        assert disabled.success
        assert "100.000/1.0" not in enabled.data["Mz/RT"].tolist()
        assert "100.000/1.0" in disabled.data["Mz/RT"].tolist()

    def test_intensity_fc_keeps_high_fold_change_feature(self, filter_proc):
        """A feature with high intensity fold-change between groups should be kept."""
        df = pd.DataFrame(
            {
                "Mz/RT": ["Sample_Type", "100.000/1.0"],
                "Tolerance": ["na", "na"],
                "Case1": ["case", 50000],
                "Case2": ["case", 60000],
                "Control1": ["control", 5000],
                "Control2": ["control", 6000],
                "QC1": ["qc", 30000],
            }
        )

        result = filter_proc.process(
            df,
            intensity_fc_threshold=2.0,
            enable_background_threshold=False,
            enable_intensity_fc_threshold=True,
        )

        assert result.success
        assert "100.000/1.0" in result.data["Mz/RT"].tolist()
        assert result.statistics.get("intensity_fc_kept", 0) >= 1

    def test_intensity_fc_does_not_keep_similar_intensity_feature(self, filter_proc):
        """A feature where groups have similar intensity should NOT pass intensity FC gate."""
        df = pd.DataFrame(
            {
                "Mz/RT": ["Sample_Type", "100.000/1.0"],
                "Tolerance": ["na", "na"],
                "Case1": ["case", 8000],
                "Case2": ["case", 8500],
                "Control1": ["control", 8200],
                "Control2": ["control", 8300],
                "QC1": ["qc", 8000],
            }
        )

        result = filter_proc.process(
            df,
            intensity_fc_threshold=2.0,
            enable_background_threshold=False,
            enable_intensity_fc_threshold=True,
        )

        assert result.success
        assert "100.000/1.0" not in result.data["Mz/RT"].tolist()

    def test_disabling_intensity_fc_rule_removes_feature_kept_only_by_fc(self, filter_proc):
        """Disabling the intensity FC gate should remove a feature kept only by that gate.

        Data design: Case ratio=0.5 (1/2 above 5000), Control ratio=0 (0/2 above 5000).
        - Stable gate: only 1 group >= 0.33 → FAIL (needs >=2)
        - MNAR gate: case_ratio=0.5 < 0.8 → FAIL (does not reach high threshold)
        - Intensity FC: Case mean=25050, Control mean=100 → FC=250.5 → PASS
        """
        df = pd.DataFrame(
            {
                "Mz/RT": ["Sample_Type", "100.000/1.0"],
                "Tolerance": ["na", "na"],
                "Case1": ["case", 50000],
                "Case2": ["case", 100],
                "Control1": ["control", 100],
                "Control2": ["control", 100],
                "QC1": ["qc", 8000],
            }
        )

        enabled = filter_proc.process(
            df,
            background_threshold=0.33,
            intensity_fc_threshold=2.0,
            enable_intensity_fc_threshold=True,
        )
        disabled = filter_proc.process(
            df,
            background_threshold=0.33,
            intensity_fc_threshold=2.0,
            enable_intensity_fc_threshold=False,
        )

        assert enabled.success
        assert disabled.success
        assert "100.000/1.0" in enabled.data["Mz/RT"].tolist()
        assert "100.000/1.0" not in disabled.data["Mz/RT"].tolist()

    def test_intensity_fc_rule_is_disabled_by_default(self, filter_proc):
        """High fold-change alone should not keep a feature unless the FC gate is explicitly enabled."""
        df = pd.DataFrame(
            {
                "Mz/RT": ["Sample_Type", "100.000/1.0"],
                "Tolerance": ["na", "na"],
                "Case1": ["case", 50000],
                "Case2": ["case", 100],
                "Control1": ["control", 100],
                "Control2": ["control", 100],
                "QC1": ["qc", 8000],
            }
        )

        result = filter_proc.process(
            df,
            background_threshold=0.33,
            intensity_fc_threshold=2.0,
        )

        assert result.success
        assert "100.000/1.0" not in result.data["Mz/RT"].tolist()

    def test_unique_stats_marginal_contribution(self, filter_proc):
        """Unique stats should reflect marginal contribution of each gate."""
        df = pd.DataFrame(
            {
                "Mz/RT": ["Sample_Type", "100.000/1.0", "200.000/2.0"],
                "Tolerance": ["na", "na", "na"],
                "Case1": ["case", 8000, 50000],
                "Case2": ["case", 8500, 60000],
                "Control1": ["control", 8200, 5000],
                "Control2": ["control", 8300, 6000],
                "QC1": ["qc", 8000, 30000],
            }
        )

        result = filter_proc.process(
            df,
            background_threshold=0.33,
            intensity_fc_threshold=2.0,
            enable_intensity_fc_threshold=True,
        )

        assert result.success
        stats = result.statistics
        assert "unique_stable_kept" in stats
        assert "unique_mnar_kept" in stats
        assert "unique_intensity_fc_kept" in stats

    def test_deprecated_skew_parameter_warns(self, filter_proc):
        """Passing removed skew_threshold should emit a DeprecationWarning."""
        import warnings

        df = pd.DataFrame(
            {
                "Mz/RT": ["Sample_Type", "100.000/1.0"],
                "Tolerance": ["na", "na"],
                "Case1": ["case", 8000],
                "Control1": ["control", 8000],
                "QC1": ["qc", 8000],
            }
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = filter_proc.process(df, skew_threshold=0.66)
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "skew_threshold" in str(deprecation_warnings[0].message)

        assert result.success

    def test_deprecated_diff_threshold_warns(self, filter_proc):
        """Passing removed diff_threshold should emit a DeprecationWarning."""
        import warnings

        df = pd.DataFrame(
            {
                "Mz/RT": ["Sample_Type", "100.000/1.0"],
                "Tolerance": ["na", "na"],
                "Case1": ["case", 8000],
                "Control1": ["control", 8000],
                "QC1": ["qc", 8000],
            }
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = filter_proc.process(df, diff_threshold=0.30)
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) >= 1
            assert "diff_threshold" in str(deprecation_warnings[0].message)

        assert result.success

    def test_output_contains_is_presence_absence_marker_column(self, filter_proc):
        """Output DataFrame must have is_Presence_Absence_Marker column."""
        df = pd.DataFrame(
            {
                "Mz/RT": ["Sample_Type", "100.000/1.0", "200.000/2.0"],
                "Tolerance": ["na", "na", "na"],
                "Case1": ["case", 8000, 8000],
                "Case2": ["case", 9000, 8500],
                "Control1": ["control", 100, 8000],
                "Control2": ["control", 200, 8500],
                "QC1": ["qc", 8000, 8000],
            }
        )

        result = filter_proc.process(df, background_threshold=0.33, qc_ratio_threshold=0.0)

        assert result.success
        assert "is_Presence_Absence_Marker" in result.data.columns

    def test_mnar_feature_is_marked_true_in_output(self, filter_proc):
        """A feature that passes the MNAR 80/20 rule must have is_Presence_Absence_Marker=True."""
        # case_ratio = 1.0 (all 10 above 5000), control_ratio = 0.0 (all 10 below 5000)
        df = pd.DataFrame(
            {
                "Mz/RT": ["Sample_Type", "100.000/1.0"],
                "Tolerance": ["na", "na"],
                "Case1": ["case", 8000],
                "Case2": ["case", 9000],
                "Case3": ["case", 8000],
                "Case4": ["case", 9000],
                "Case5": ["case", 8000],
                "Case6": ["case", 9000],
                "Case7": ["case", 8000],
                "Case8": ["case", 9000],
                "Case9": ["case", 8000],
                "Case10": ["case", 9000],
                "Control1": ["control", 100],
                "Control2": ["control", 200],
                "Control3": ["control", 100],
                "Control4": ["control", 200],
                "Control5": ["control", 100],
                "Control6": ["control", 200],
                "Control7": ["control", 100],
                "Control8": ["control", 200],
                "Control9": ["control", 100],
                "Control10": ["control", 200],
                "QC1": ["qc", 8000],
            }
        )

        result = filter_proc.process(
            df,
            high_det_thresh=0.8,
            low_det_thresh=0.2,
            qc_ratio_threshold=0.0,
        )

        assert result.success
        feature_rows = result.data[result.data["Mz/RT"] == "100.000/1.0"]
        assert not feature_rows.empty
        assert feature_rows["is_Presence_Absence_Marker"].iloc[0] is True

    def test_symmetric_feature_is_marked_false_in_output(self, filter_proc):
        """A feature with symmetric detection in both groups must have is_Presence_Absence_Marker=False."""
        # case_ratio = 1.0, control_ratio = 1.0 → no low group → MNAR=False
        df = pd.DataFrame(
            {
                "Mz/RT": ["Sample_Type", "100.000/1.0"],
                "Tolerance": ["na", "na"],
                "Case1": ["case", 8000],
                "Case2": ["case", 9000],
                "Control1": ["control", 8000],
                "Control2": ["control", 9000],
                "QC1": ["qc", 8000],
            }
        )

        result = filter_proc.process(
            df,
            background_threshold=0.33,
            high_det_thresh=0.8,
            low_det_thresh=0.2,
            qc_ratio_threshold=0.0,
        )

        assert result.success
        feature_rows = result.data[result.data["Mz/RT"] == "100.000/1.0"]
        assert not feature_rows.empty
        assert feature_rows["is_Presence_Absence_Marker"].iloc[0] is False

    # ------------------------------------------------------------------
    # Single-group (allow_single_group_stable) tests
    # ------------------------------------------------------------------

    def test_single_group_stable_gate_deletes_all_without_degradation(self, filter_proc):
        """With only 1 analysis group and allow_single_group_stable=False (default),
        the stable gate always fails because sum >= 2 can never be satisfied."""
        df = pd.DataFrame(
            {
                "Mz/RT": ["Sample_Type", "100.000/1.0", "200.000/2.0"],
                "Tolerance": ["na", "na", "na"],
                "Exposure1": ["exposure", 8000, 8000],
                "Exposure2": ["exposure", 9000, 9000],
                "QC1": ["qc", 8000, 8000],
            }
        )
        result = filter_proc.process(
            df,
            background_threshold=0.33,
            enable_intensity_fc_threshold=False,
            enable_mnar_gate=False,
        )
        assert result.success
        assert "100.000/1.0" not in result.data["Mz/RT"].tolist()
        assert "200.000/2.0" not in result.data["Mz/RT"].tolist()

    def test_single_group_stable_degradation_keeps_features_above_threshold(self, filter_proc):
        """With allow_single_group_stable=True and exactly 1 group, stable gate
        degrades to: group ratio >= background_threshold."""
        df = pd.DataFrame(
            {
                "Mz/RT": ["Sample_Type", "100.000/1.0", "200.000/2.0"],
                "Tolerance": ["na", "na", "na"],
                "Exposure1": ["exposure", 8000, 100],
                "Exposure2": ["exposure", 9000, 200],
                "QC1": ["qc", 8000, 8000],
            }
        )
        result = filter_proc.process(
            df,
            background_threshold=0.33,
            enable_intensity_fc_threshold=False,
            enable_mnar_gate=False,
            allow_single_group_stable=True,
        )
        assert result.success
        # 100.000/1.0: both exposure samples above 5000 → ratio=1.0 >= 0.33 → kept
        assert "100.000/1.0" in result.data["Mz/RT"].tolist()
        # 200.000/2.0: both exposure samples below 5000 → ratio=0.0 < 0.33 → deleted
        assert "200.000/2.0" not in result.data["Mz/RT"].tolist()

    def test_single_group_degradation_does_not_activate_for_two_or_more_groups(self, filter_proc):
        """allow_single_group_stable has no effect when there are >= 2 groups."""
        df = pd.DataFrame(
            {
                "Mz/RT": ["Sample_Type", "100.000/1.0"],
                "Tolerance": ["na", "na"],
                "Case1": ["case", 8000],
                "Case2": ["case", 9000],
                "Control1": ["control", 8000],
                "Control2": ["control", 9000],
                "QC1": ["qc", 8000],
            }
        )
        normal = filter_proc.process(
            df,
            background_threshold=0.33,
            qc_ratio_threshold=0.0,
        )
        degraded = filter_proc.process(
            df,
            background_threshold=0.33,
            qc_ratio_threshold=0.0,
            allow_single_group_stable=True,
        )
        assert normal.success and degraded.success
        # Both should keep the feature (2-group stable gate passes)
        assert "100.000/1.0" in normal.data["Mz/RT"].tolist()
        assert "100.000/1.0" in degraded.data["Mz/RT"].tolist()

    def test_single_group_degradation_metadata_records_flag(self, filter_proc):
        """allow_single_group_stable is reflected in enabled_thresholds metadata."""
        df = pd.DataFrame(
            {
                "Mz/RT": ["Sample_Type", "100.000/1.0"],
                "Tolerance": ["na", "na"],
                "Exposure1": ["exposure", 8000],
                "QC1": ["qc", 8000],
            }
        )
        result = filter_proc.process(df, allow_single_group_stable=True)
        assert result.success
        assert result.metadata["enabled_thresholds"]["single_group_stable"] is True

    # ------------------------------------------------------------------
    # enable_mnar_gate toggle tests
    # ------------------------------------------------------------------

    def test_disabling_mnar_gate_removes_presence_absence_feature(self, filter_proc):
        """When enable_mnar_gate=False, a feature that only passes via MNAR is removed."""
        df = pd.DataFrame(
            {
                "Mz/RT": ["Sample_Type", "100.000/1.0"],
                "Tolerance": ["na", "na"],
                "Case1": ["case", 8000],
                "Case2": ["case", 9000],
                "Control1": ["control", 100],
                "Control2": ["control", 200],
                "QC1": ["qc", 8000],
            }
        )
        result = filter_proc.process(
            df,
            background_threshold=0.33,
            high_det_thresh=0.8,
            low_det_thresh=0.2,
            qc_ratio_threshold=0.0,
            enable_background_threshold=False,
            enable_intensity_fc_threshold=False,
            enable_mnar_gate=False,
        )
        assert result.success
        assert "100.000/1.0" not in result.data["Mz/RT"].tolist()

    def test_enabling_mnar_gate_keeps_presence_absence_feature(self, filter_proc):
        """When enable_mnar_gate=True (default), a MNAR feature is kept."""
        df = pd.DataFrame(
            {
                "Mz/RT": ["Sample_Type", "100.000/1.0"],
                "Tolerance": ["na", "na"],
                "Case1": ["case", 8000],
                "Case2": ["case", 9000],
                "Case3": ["case", 8000],
                "Case4": ["case", 9000],
                "Case5": ["case", 8000],
                "Case6": ["case", 9000],
                "Case7": ["case", 8000],
                "Case8": ["case", 9000],
                "Case9": ["case", 8000],
                "Case10": ["case", 9000],
                "Control1": ["control", 100],
                "Control2": ["control", 200],
                "Control3": ["control", 100],
                "Control4": ["control", 200],
                "Control5": ["control", 100],
                "Control6": ["control", 200],
                "Control7": ["control", 100],
                "Control8": ["control", 200],
                "Control9": ["control", 100],
                "Control10": ["control", 200],
                "QC1": ["qc", 8000],
            }
        )
        result = filter_proc.process(
            df,
            background_threshold=0.33,
            high_det_thresh=0.8,
            low_det_thresh=0.2,
            qc_ratio_threshold=0.0,
            enable_background_threshold=False,
            enable_intensity_fc_threshold=False,
            enable_mnar_gate=True,
        )
        assert result.success
        assert "100.000/1.0" in result.data["Mz/RT"].tolist()

    def test_mnar_gate_metadata_records_enabled_state(self, filter_proc):
        """enable_mnar_gate is reflected in enabled_thresholds metadata."""
        df = pd.DataFrame(
            {
                "Mz/RT": ["Sample_Type", "100.000/1.0"],
                "Tolerance": ["na", "na"],
                "Case1": ["case", 8000],
                "Control1": ["control", 8000],
                "QC1": ["qc", 8000],
            }
        )
        result_off = filter_proc.process(df, enable_mnar_gate=False)
        result_on = filter_proc.process(df, enable_mnar_gate=True)
        assert result_off.metadata["enabled_thresholds"]["mnar_gate"] is False
        assert result_on.metadata["enabled_thresholds"]["mnar_gate"] is True
