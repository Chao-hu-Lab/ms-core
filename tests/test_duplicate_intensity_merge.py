import numpy as np
import pandas as pd

from ms_core.preprocessing.duplicate_remover import DuplicateRemover
from ms_core.preprocessing.duplicate_intensity_merge import DuplicateIntensityMerger


def test_per_sample_max_upgrades_overlapping_donor_values() -> None:
    df = pd.DataFrame(
        {
            "Sample1": [1000, 0],
            "Sample2": [2000, 4500],
            "Sample3": [3000, 2800],
        }
    )

    merged, stats = DuplicateIntensityMerger().merge(
        df,
        merge_groups=[[0, 1]],
        intensity_cols=["Sample1", "Sample2", "Sample3"],
        merge_mode="per_sample_max",
    )

    assert float(merged.at[0, "Sample1"]) == 1000
    assert float(merged.at[0, "Sample2"]) == 4500
    assert float(merged.at[0, "Sample3"]) == 3000
    assert stats["data_points_recovered"] == 0
    assert stats["data_points_upgraded"] == 1


def test_fill_gaps_preserves_legacy_overlap_behavior() -> None:
    df = pd.DataFrame(
        {
            "Sample1": [1000, 0],
            "Sample2": [2000, 4500],
            "Sample3": [0, 2800],
        }
    )

    merged, stats = DuplicateIntensityMerger().merge(
        df,
        merge_groups=[[0, 1]],
        intensity_cols=["Sample1", "Sample2", "Sample3"],
        merge_mode="fill_gaps",
    )

    assert float(merged.at[0, "Sample1"]) == 1000
    assert float(merged.at[0, "Sample2"]) == 2000
    assert float(merged.at[0, "Sample3"]) == 2800
    assert stats["data_points_recovered"] == 1
    assert stats["data_points_upgraded"] == 0


def test_zeros_blanks_and_nan_are_treated_as_missing() -> None:
    df = pd.DataFrame(
        {
            "Zero": [0, 10],
            "Blank": ["", "20"],
            "Nan": [np.nan, 30],
            "LowerOverlap": [40, 35],
        }
    )

    merged, stats = DuplicateIntensityMerger().merge(
        df,
        merge_groups=[[0, 1]],
        intensity_cols=["Zero", "Blank", "Nan", "LowerOverlap"],
        merge_mode="per_sample_max",
    )

    assert float(merged.at[0, "Zero"]) == 10
    assert float(merged.at[0, "Blank"]) == 20
    assert float(merged.at[0, "Nan"]) == 30
    assert float(merged.at[0, "LowerOverlap"]) == 40
    assert stats["data_points_recovered"] == 3
    assert stats["data_points_upgraded"] == 0


def test_merge_stats_report_recovered_and_upgraded_data_points() -> None:
    df = pd.DataFrame(
        {
            "Recovered": [0, 100],
            "Upgraded": [50, 90],
            "Preserved": [70, 60],
        }
    )

    _, stats = DuplicateIntensityMerger().merge(
        df,
        merge_groups=[[0, 1]],
        intensity_cols=["Recovered", "Upgraded", "Preserved"],
        merge_mode="per_sample_max",
    )

    assert stats == {
        "groups_merged": 1,
        "data_points_recovered": 1,
        "data_points_upgraded": 1,
    }


def test_protected_representatives_can_still_receive_per_sample_upgrades() -> None:
    df = pd.DataFrame(
        {
            "Sample1": [1000, 4500],
            "Sample2": [800, 700],
            "Sample3": [600, 0],
        }
    )

    merged, stats = DuplicateIntensityMerger().merge(
        df,
        merge_groups=[[0, 1]],
        intensity_cols=["Sample1", "Sample2", "Sample3"],
        merge_mode="per_sample_max",
    )

    assert float(merged.at[0, "Sample1"]) == 4500
    assert float(merged.at[0, "Sample2"]) == 800
    assert float(merged.at[0, "Sample3"]) == 600
    assert stats["data_points_upgraded"] == 1


def test_duplicate_remover_facade_reports_merge_stats_through_process() -> None:
    df = pd.DataFrame(
        {
            "Mz/RT": ["Sample_Type", "100.0000/1.00", "100.0001/1.01"],
            "Sample1": ["case", 0, 100],
            "Sample2": ["case", 50, 90],
            "Sample3": ["case", 1000, 60],
        }
    )

    result = DuplicateRemover().process(
        df,
        mz_tolerance_ppm=20,
        rt_tolerance=0.1,
        merge_mode="per_sample_max",
        protected_rows={1},
    )

    assert result.success
    assert len(result.data) == 2
    assert result.statistics["duplicates_removed"] == 1
    assert result.statistics["groups_merged"] == 1
    assert result.statistics["data_points_recovered"] == 1
    assert result.statistics["data_points_upgraded"] == 1

    merged_row = result.data.iloc[1]
    assert float(merged_row["Sample1"]) == 100
    assert float(merged_row["Sample2"]) == 90
    assert float(merged_row["Sample3"]) == 1000
