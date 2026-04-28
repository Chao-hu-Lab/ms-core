"""Tests for ms-core pytest marker classification."""

from __future__ import annotations

from pathlib import Path

from tests.testing_markers import classify_test_markers


def test_algorithm_marker_selects_processing_regressions() -> None:
    combined_tsv = classify_test_markers(Path("tests") / "test_combined_tsv_preprocessor.py")
    feature_filter = classify_test_markers(Path("tests") / "test_feature_filter_small_n.py")
    data_organizer = classify_test_markers(
        Path("tests") / "test_data_organizer_false_positive_fix.py"
    )

    assert combined_tsv == {"algorithm"}
    assert feature_filter == {"algorithm"}
    assert data_organizer == {"algorithm"}


def test_io_marker_selects_storage_and_cache_contracts() -> None:
    intermediate_store = classify_test_markers(Path("tests") / "test_intermediate_store.py")
    cache_policy = classify_test_markers(Path("tests") / "test_cache_path_policy.py")

    assert intermediate_store == {"contract", "io"}
    assert cache_policy == {"io"}


def test_pipeline_marker_overlaps_with_public_contract() -> None:
    markers = classify_test_markers(Path("tests") / "test_pipeline.py")

    assert markers == {"contract", "pipeline"}


def test_dataset_marker_is_public_contract() -> None:
    markers = classify_test_markers(Path("tests") / "test_dataset.py")

    assert markers == {"contract"}


def test_root_hygiene_is_serial() -> None:
    markers = classify_test_markers(Path("tests") / "test_root_hygiene.py")

    assert markers == {"hygiene", "serial"}
