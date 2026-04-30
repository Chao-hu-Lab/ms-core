import math
from pathlib import Path
from typing import Optional

import pandas as pd

from ms_core.preprocessing.base import ProcessingResult
from ms_core.preprocessing.combined_tsv import CombinedTsvPreprocessor


def test_process_combined_splits_sides_and_restores_mzmine_id_order() -> None:
    calls: list[pd.DataFrame] = []

    def process_statistics(
        *,
        df: pd.DataFrame,
        method_file: Optional[str | Path],
        mz_decimals: int,
        rt_decimals: int,
        sample_type_mapping: Optional[dict[str, str]],
    ) -> ProcessingResult:
        calls.append(df.copy())
        if "MZmine ID" in df.columns:
            return ProcessingResult(
                success=True,
                data=pd.DataFrame(
                    [["na", "na", "na", "na"], [501.1, 7.2, 40, "row_1"]],
                    columns=["MZmine m/z", "MZmine RT", "SampleA", "MZmine ID"],
                ),
                metadata={"side": "mz"},
            )
        return ProcessingResult(
            success=True,
            data=pd.DataFrame(
                [["na", "na", "na"], [245.1, 12.2, 4]],
                columns=["Mz", "RT", "SampleA"],
            ),
            metadata={"side": "fh", "sample_info": pd.DataFrame({"Sample_Name": ["SampleA"]})},
        )

    df = pd.DataFrame(
        [[245.1, 12.2, 4, "row_1", 501.1, 7.2, 40, None]],
        columns=[
            "Mz",
            "RT",
            "SampleA",
            "MZmine ID",
            "MZmine m/z",
            "MZmine RT",
            "SampleA",
            "Unnamed: 99",
        ],
    )

    result = CombinedTsvPreprocessor(process_statistics).process_combined(df)

    assert result.success
    assert len(calls) == 2
    assert list(calls[0].columns) == ["Mz", "RT", "SampleA"]
    assert list(calls[1].columns) == ["MZmine ID", "MZmine m/z", "MZmine RT", "SampleA"]
    assert list(result.data.columns) == [
        "Mz",
        "RT",
        "SampleA",
        "MZmine ID",
        "MZmine m/z",
        "MZmine RT",
        "SampleA",
    ]
    assert result.metadata["mode"] == "combined"
    assert result.metadata["sample_info"].to_dict("list") == {"Sample_Name": ["SampleA"]}


def test_process_combined_and_fix_preserves_mzmine_id_values_for_filtering() -> None:
    from ms_core.preprocessing.data_organizer import DataOrganizer

    df = pd.DataFrame(
        {
            "Mz": [100.0, 200.0],
            "RT": [1.0, 2.0],
            "SampleA": [10.0, 20.0],
            "SampleB": [0.0, 30.0],
            "MZmine ID": ["mz1", pd.NA],
            "MZmine m/z": [100.001, pd.NA],
            "MZmine RT (min)": [1.1, pd.NA],
            "SampleA.mzML Peak area": [100.0, pd.NA],
            "SampleB.mzML Peak area": [0.0, pd.NA],
        }
    )

    result = DataOrganizer().process(df, mode="combined_fix")

    assert result.success
    assert result.data.shape[0] == 1
    assert result.data.iloc[0]["Mz"] == 100.001
    assert result.data.iloc[0]["RT"] == 1.1
    assert result.data.iloc[0]["SampleA"] == 6000.0
    assert pd.isna(result.data.iloc[0]["SampleB"])


def test_false_positive_fix_treats_zero_as_missing_and_clears_zero_outputs() -> None:
    preprocessor = CombinedTsvPreprocessor()
    df = pd.DataFrame(
        [
            [245.1332, 12.21, 0, 5, "row_1", 245.133243, 12.279866, 398734.3, 14731746.6],
            [260.0000, 13.50, "", 0, "row_2", 260.000100, 13.600000, 10.0, 0.0],
        ],
        columns=[
            "Mz",
            "RT",
            "SampleA",
            "SampleB",
            "MZmine ID",
            "MZmine m/z",
            "MZmine RT (min)",
            "SampleA",
            "SampleB",
        ],
    )

    result = preprocessor.false_positive_fix(df)

    assert list(result.columns) == ["Mz", "RT", "SampleA", "SampleB"]
    assert result.shape == (2, 4)

    assert math.isclose(result.iloc[0, 0], 245.133243)
    assert math.isclose(result.iloc[0, 1], 12.279866)

    assert pd.isna(result.iloc[0, 2])
    assert math.isclose(
        result.iloc[0, 3],
        14731746.6 * preprocessor.MZMINE_AREA_UNIT_FACTOR,
    )

    assert pd.isna(result.iloc[1, 2])
    assert pd.isna(result.iloc[1, 3])

    sample_data = result.iloc[:, 2:]
    assert not sample_data.eq(0).any().any()


def test_process_combined_and_fix_returns_failed_result_for_unexpected_fix_error(
    monkeypatch,
) -> None:
    preprocessor = CombinedTsvPreprocessor()

    def process_combined(*_args, **_kwargs) -> ProcessingResult:
        return ProcessingResult(
            success=True,
            data=pd.DataFrame({"Mz": [1.0], "RT": [2.0]}),
        )

    def false_positive_fix(_df: pd.DataFrame) -> pd.DataFrame:
        raise RuntimeError("unexpected pandas failure")

    monkeypatch.setattr(preprocessor, "process_combined", process_combined)
    monkeypatch.setattr(preprocessor, "false_positive_fix", false_positive_fix)

    result = preprocessor.process_combined_and_fix(pd.DataFrame())

    assert not result.success
    assert result.errors == ["unexpected pandas failure"]
    assert result.message == "False-positive fix failed: unexpected pandas failure"
