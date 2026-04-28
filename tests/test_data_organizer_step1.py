from __future__ import annotations

import pandas as pd

from ms_core.preprocessing.data_organizer import InjectionInfo
from ms_core.preprocessing.data_organizer_step1 import (
    StatisticsMzRtRestore,
    assemble_step1_output,
    prepare_step1_input,
    simplify_input_sample_type_overrides,
)


def test_prepare_step1_input_extracts_common_state_and_statistics_restore() -> None:
    df = pd.DataFrame(
        [
            ["Sample Type", "na", "control"],
            ["100.1234/1.23", "ignored", 10],
        ],
        columns=["Mz/RT", "Meta", "TumorBC1_DNA"],
    )
    injection_info = [InjectionInfo(1, "TumorBC1_DNA", "TumorBC1_DNA", 5.0)]

    prepared = prepare_step1_input(
        df,
        method_file="dummy.docx",
        sample_type_mapping={"Manual": "QC"},
        mode="statistics",
        parse_method_file=lambda _path: {"TumorBC1": "Exposure"},
        parse_injection_sequence=lambda _path: injection_info,
    )

    assert list(prepared.data.columns) == ["Mz", "RT", "Meta", "TumorBC1_DNA"]
    assert prepared.input_sample_types == {"TumorBC1_DNA": "Control"}
    assert prepared.sample_mapping == {"TumorBC1": "Exposure", "Manual": "QC"}
    assert prepared.injection_info_list == injection_info
    assert prepared.statistics["mode"] == "statistics"
    assert prepared.statistics["original_rows"] == 2
    assert prepared.statistics["sample_types_from_input"] is True
    assert prepared.statistics["method_file_samples"] == 2
    assert prepared.statistics_restore == StatisticsMzRtRestore(
        mz_column_name="Mz",
        rt_column_name="RT",
        mz_values=[100.1234],
        rt_values=[1.23],
    )


def test_simplify_input_sample_type_overrides_uses_header_mapping_and_normalizes() -> None:
    overrides = simplify_input_sample_type_overrides(
        {"raw/TumorBC1.tsv": "benign", "QC1": "quality control"},
        {"raw/TumorBC1.tsv": "TumorBC1_DNA"},
        extract_sample_name=lambda raw: f"simplified-{raw}",
    )

    assert overrides == {
        "TumorBC1_DNA": "Control",
        "simplified-QC1": "QC",
    }


def test_assemble_step1_output_cleans_sample_info_and_restores_statistics_columns() -> None:
    df = pd.DataFrame(
        {
            "Mz/RT": ["Sample_Type", "100.1234/1.23"],
            "SampleB": ["sample", "20"],
            "SampleA": ["sample", "10"],
        }
    )
    sample_info = pd.DataFrame(
        {
            "Sample_Name": ["SampleA", "SampleB"],
            "Injection_Order": [1, 2],
            "_col_name": ["SampleA", "SampleB"],
        }
    )
    restore = StatisticsMzRtRestore(
        mz_column_name="Mz",
        rt_column_name="RT",
        mz_values=[100.1234],
        rt_values=[1.23],
    )

    assembled = assemble_step1_output(
        df,
        injection_info_list=[],
        build_sample_info=lambda _df, _info: sample_info,
        reorder_columns_by_injection=lambda frame, _info: frame[["Mz/RT", "SampleA", "SampleB"]],
        finalize_structure=lambda frame: frame,
        statistics_restore=restore,
    )

    assert list(assembled.data.columns) == ["Mz", "RT", "SampleA", "SampleB"]
    assert assembled.data.iloc[0].to_dict() == {
        "Mz": "Sample_Type",
        "RT": "na",
        "SampleA": "sample",
        "SampleB": "sample",
    }
    assert "_col_name" not in assembled.sample_info.columns
    assert assembled.statistics == {
        "sample_info_rows": 2,
        "final_rows": 2,
        "final_cols": 4,
    }


def test_assemble_step1_output_reports_cancellation_before_reorder() -> None:
    df = pd.DataFrame({"Mz/RT": ["Sample_Type", "100.0000/1.00"], "SampleA": ["sample", "10"]})
    sample_info = pd.DataFrame({"Sample_Name": ["SampleA"], "_col_name": ["SampleA"]})

    assembled = assemble_step1_output(
        df,
        injection_info_list=[],
        build_sample_info=lambda _df, _info: sample_info,
        reorder_columns_by_injection=lambda _df, _info: (_ for _ in ()).throw(
            AssertionError("reorder should not run after cancellation")
        ),
        finalize_structure=lambda frame: frame,
        cancellation_requested=lambda: True,
    )

    assert assembled.cancelled is True
    assert assembled.statistics == {"sample_info_rows": 1}
