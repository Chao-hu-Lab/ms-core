from __future__ import annotations

from ms_core.preprocessing.data_organizer_method import extract_sample_id
from ms_core.preprocessing.data_organizer_method import _sample_type_from_text
from ms_core.preprocessing.data_organizer_layout import normalize_sample_type_value


def test_method_sample_type_labels_normalize_to_toolkit_canonical_values() -> None:
    assert normalize_sample_type_value(_sample_type_from_text("Tumor tissue BC2257_DNA")) == "Exposure"
    assert normalize_sample_type_value(_sample_type_from_text("Benign tissue BC2257_DNA")) == "Control"
    assert normalize_sample_type_value(_sample_type_from_text("pooled QC 1")) == "QC"


def test_method_extract_sample_id_preserves_existing_patterns() -> None:
    assert extract_sample_id("Tumor tissue BC2257_DNA") == "BC2257_DNA"
    assert extract_sample_id("Sample EC301") == "EC301"
