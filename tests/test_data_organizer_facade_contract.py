from __future__ import annotations

import pandas as pd

from ms_core.preprocessing.data_organizer import DataOrganizer, InjectionInfo


def test_normalization_process_preserves_step1_output_contract(monkeypatch) -> None:
    organizer = DataOrganizer()
    df = pd.DataFrame(
        {
            "Mz": [100.123456, 200.2],
            "RT": [1.234, 2.345],
            "BC2287": [20, 40],
            "BC2286_2": [10, 30],
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
    assert list(result.data.columns) == ["Mz/RT", "BC2286_2", "BC2287"]
    assert result.data.iloc[0].to_dict() == {
        "Mz/RT": "Sample_Type",
        "BC2286_2": "sample",
        "BC2287": "sample",
    }
    assert result.statistics["original_rows"] == 2
    assert result.statistics["original_cols"] == 4
    assert result.statistics["mz_rt_merged"] == 2
    assert result.statistics["sample_info_rows"] == 2
    assert result.statistics["final_rows"] == 3
    assert result.metadata["mode"] == "normalization"
    assert result.metadata["header_mapping"] == {"BC2287": "BC2287", "BC2286_2": "BC2286_2"}
    assert result.metadata["sample_mapping"] == {}

    sample_info = result.metadata["sample_info"]
    assert list(sample_info.columns) == [
        "Sample_Name",
        "Method_Sample_Name",
        "Sample_Type",
        "Injection_Order",
        "Batch",
        "Injection_Volume",
        "DNA_mg/20uL",
    ]
    assert list(sample_info["Sample_Name"]) == ["BC2286_2", "BC2287"]
    assert list(sample_info["Method_Sample_Name"]) == ["BC2286", "BC2287"]


def test_statistics_process_preserves_separate_mz_rt_and_metadata(monkeypatch) -> None:
    organizer = DataOrganizer()
    df = pd.DataFrame(
        {
            "Mz": [100.123456, 200.2],
            "RT": [1.234, 2.345],
            "BC2287": [20, 40],
            "BC2286_2": [10, 30],
        }
    )
    injection_info = [
        InjectionInfo(1, "BC2286", "BC2286", 5.0),
        InjectionInfo(2, "BC2287", "BC2287", 5.0),
    ]
    monkeypatch.setattr(organizer, "_parse_method_file", lambda _path: {})
    monkeypatch.setattr(organizer, "_parse_injection_sequence", lambda _path: injection_info)

    result = organizer.process(df, method_file="dummy.docx", mode="statistics")

    assert result.success
    assert list(result.data.columns) == ["Mz", "RT", "BC2286_2", "BC2287"]
    assert result.data.iloc[0].to_dict() == {
        "Mz": "Sample_Type",
        "RT": "na",
        "BC2286_2": "sample",
        "BC2287": "sample",
    }
    assert result.data["Mz"].tolist()[1:] == [100.123456, 200.2]
    assert result.data["RT"].tolist()[1:] == [1.234, 2.345]
    assert result.statistics["mode"] == "statistics"
    assert result.statistics["merge_skipped_in_output"] is True
    assert result.statistics["mz_rt_merged"] == 0
    assert result.statistics["sample_info_rows"] == 2
    assert result.metadata["mode"] == "statistics"
    assert result.metadata["header_mapping"] == {"BC2287": "BC2287", "BC2286_2": "BC2286_2"}
    assert result.metadata["sample_mapping"] == {}

    sample_info = result.metadata["sample_info"]
    assert list(sample_info["Sample_Name"]) == ["BC2286_2", "BC2287"]
    assert list(sample_info["Method_Sample_Name"]) == ["BC2286", "BC2287"]


def test_input_sample_type_row_overrides_detected_sample_types() -> None:
    organizer = DataOrganizer()
    df = pd.DataFrame(
        [
            ["Sample Type", "na", "control", "QC"],
            [100.123456, 1.234, 10, 20],
        ],
        columns=["Mz", "RT", "TumorBC2257_DNA", "pooled_QC1"],
    )

    result = organizer.process(df)

    assert result.success
    assert result.data.iloc[0].to_dict() == {
        "Mz/RT": "Sample_Type",
        "TumorBC2257_DNA": "Control",
        "pooled_QC_1": "QC",
    }
    assert result.metadata["header_mapping"]["pooled_QC1"] == "pooled_QC_1"
    assert result.statistics["sample_types_from_input"] is True
    assert result.statistics["input_sample_type_count"] == 2
    assert result.statistics["types_from_input"] == 2
