from __future__ import annotations

import pytest

from ms_core.preprocessing.sample_identity import (
    SampleInfoIdentity,
    build_sample_info_identity,
    normalize_sample_token,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("BC2286_2", "bc2286"),
        ("DNA_program1_TumorBC2257_DNA", "tumorbc2257_dna"),
        ("program2_program1_TumorBC2257_DNA", "tumorbc2257_dna"),
        ("program2_DNA_program1_TumorBC2257_DNA", "tumorbc2257_dna"),
        ("EC013_2", "ec013"),
        ("EC301", "ec0301"),
        ("ZBEE000070", "u00070zbee"),
        ("ZBEE000070_2", "u00070zbee"),
        ("U00070ZBEE_2", "u00070zbee"),
        ("QC sample 1", "qc_sample_1"),
        ("DNAandRNA", "dnaandrna"),
    ],
)
def test_normalize_sample_token_documents_matching_keys(raw: str, expected: str) -> None:
    assert normalize_sample_token(raw) == expected


def test_sample_info_identity_preserves_raw_column_name() -> None:
    identity = build_sample_info_identity("BC2286_2", "BC2286")

    assert identity == SampleInfoIdentity(
        sample_name="BC2286_2",
        method_sample_name="BC2286",
    )


@pytest.mark.parametrize("method_name", [None, "", "   "])
def test_sample_info_identity_uses_empty_method_name_for_unmatched_rows(
    method_name: str | None,
) -> None:
    identity = build_sample_info_identity("BC9999_2", method_name)

    assert identity.sample_name == "BC9999_2"
    assert identity.method_sample_name == ""
