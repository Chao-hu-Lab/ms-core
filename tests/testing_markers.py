"""Central pytest marker classification rules for the ms-core suite."""

from __future__ import annotations

from pathlib import Path

ALGORITHM_TEST_FILES = frozenset(
    {
        "test_combined_tsv_preprocessor.py",
        "test_data_organizer_facade_contract.py",
        "test_data_organizer_false_positive_fix.py",
        "test_data_organizer_layout.py",
        "test_data_organizer_method.py",
        "test_data_organizer_step1.py",
        "test_degeneracy_annotation.py",
        "test_detection_ratios.py",
        "test_duplicate_intensity_merge.py",
        "test_feature_filter_decisions.py",
        "test_feature_filter_output.py",
        "test_feature_filter_small_n.py",
        "test_feature_groups.py",
    }
)

IO_TEST_FILES = frozenset(
    {
        "test_cache_path_policy.py",
        "test_intermediate_store.py",
    }
)

PIPELINE_TEST_FILES = frozenset(
    {
        "test_pipeline.py",
    }
)

CONTRACT_TEST_FILES = frozenset(
    {
        "test_dataset.py",
        "test_intermediate_store.py",
        "test_pipeline.py",
    }
)

HYGIENE_TEST_FILES = frozenset(
    {
        "test_root_hygiene.py",
    }
)

SERIAL_TEST_FILES = frozenset(
    {
        "test_root_hygiene.py",
    }
)


def classify_test_markers(path: Path | str) -> set[str]:
    """Return pytest markers implied by a test file path."""
    test_path = Path(path)
    file_name = test_path.name
    markers: set[str] = set()

    if file_name in ALGORITHM_TEST_FILES:
        markers.add("algorithm")

    if file_name in IO_TEST_FILES:
        markers.add("io")

    if file_name in PIPELINE_TEST_FILES:
        markers.add("pipeline")

    if file_name in CONTRACT_TEST_FILES:
        markers.add("contract")

    if file_name in HYGIENE_TEST_FILES:
        markers.add("hygiene")

    if file_name in SERIAL_TEST_FILES:
        markers.add("serial")

    return markers
