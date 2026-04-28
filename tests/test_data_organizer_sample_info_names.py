from __future__ import annotations

import pandas as pd

from ms_core.preprocessing.data_organizer import DataOrganizer, InjectionInfo


def test_sample_info_sample_name_matches_raw_intensity_header_for_reruns() -> None:
    organizer = DataOrganizer()
    df = pd.DataFrame(
        [
            ["Sample_Type", "sample", "sample"],
            ["100.1000/1.00", 1.0, 2.0],
        ],
        columns=["Mz/RT", "BC2286_2", "BC2287"],
    )
    injection_info = [
        InjectionInfo(1, "BC2286", "BC2286", 5.0),
        InjectionInfo(2, "BC2287", "BC2287", 5.0),
    ]

    sample_info = organizer._build_sample_info(df, injection_info)

    assert list(sample_info.columns[:2]) == ["Sample_Name", "Method_Sample_Name"]
    assert list(sample_info["Sample_Name"]) == ["BC2286_2", "BC2287"]
    assert list(sample_info["Method_Sample_Name"]) == ["BC2286", "BC2287"]
    assert list(sample_info["Injection_Order"]) == [1, 2]


def test_sample_info_matches_identity_variants_without_renaming_raw_columns() -> None:
    organizer = DataOrganizer()
    df = pd.DataFrame(
        [
            ["Sample_Type", "sample", "sample", "QC", "sample"],
            ["100.1000/1.00", 1.0, 2.0, 3.0, 4.0],
        ],
        columns=[
            "Mz/RT",
            "program2_program1_TumorBC2257_DNA",
            "ZBEE000070_2",
            "QC_4",
            "EC301_2",
        ],
    )
    injection_info = [
        InjectionInfo(1, "ZBEE000070", "ZBEE000070", 5.0),
        InjectionInfo(2, "Tumor tissue BC2257_DNA", "Tumor tissue BC2257_DNA", 5.0),
        InjectionInfo(3, "EC301", "EC301", 5.0),
        InjectionInfo(4, "QC sample 4", "QC sample 4", 5.0),
    ]

    sample_info = organizer._build_sample_info(df, injection_info)

    assert list(sample_info["Sample_Name"]) == [
        "ZBEE000070_2",
        "program2_program1_TumorBC2257_DNA",
        "EC301_2",
        "QC_4",
    ]
    assert list(sample_info["Method_Sample_Name"]) == [
        "ZBEE000070",
        "Tumor tissue BC2257_DNA",
        "EC301",
        "QC sample 4",
    ]
    assert list(sample_info["Injection_Order"]) == [1, 2, 3, 4]


def test_step1_output_sample_info_names_match_raw_intensity_columns(monkeypatch) -> None:
    organizer = DataOrganizer()
    df = pd.DataFrame(
        {
            "Mz": ["Sample_Type", 100.1],
            "RT": ["na", 1.0],
            "BC2287": ["sample", 2.0],
            "BC2286_2": ["sample", 1.0],
        }
    )
    injection_info = [
        InjectionInfo(1, "BC2286", "BC2286", 5.0),
        InjectionInfo(2, "BC2287", "BC2287", 5.0),
    ]
    monkeypatch.setattr(organizer, "_parse_method_file", lambda _path: {})
    monkeypatch.setattr(organizer, "_parse_injection_sequence", lambda _path: injection_info)

    result = organizer.process(df, method_file="dummy.docx")

    assert result.success
    sample_info = result.metadata["sample_info"]
    raw_sample_columns = list(result.data.columns[1:])
    assert list(sample_info["Sample_Name"]) == raw_sample_columns
    assert list(sample_info["Method_Sample_Name"]) == ["BC2286", "BC2287"]
