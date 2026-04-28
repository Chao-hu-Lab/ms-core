from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

import pandas as pd

from ms_core.preprocessing.degeneracy_annotation import DegeneracyAnnotator
from ms_core.preprocessing.duplicate_remover import DuplicateRemover

TempDirFactory = Callable[[str], AbstractContextManager[Path]]


def _annotation_input() -> tuple[pd.DataFrame, dict[str, object], pd.Series]:
    df = pd.DataFrame(
        {
            "Mz/RT": ["242.1191/10.99", "264.1010/11.00", "300.0000/20.00"],
            "Case1": [100, 50, 20],
            "Case2": [200, 100, 25],
            "Case3": [400, 200, 22],
            "Case4": [800, 400, 19],
            "QC1": [51000, 38500, 11500],
            "_mz": [242.1191, 264.1010, 300.0000],
            "_rt": [10.99, 11.00, 20.00],
            "_total_intensity": [51500, 39250, 11586],
        }
    )
    col_info = {"intensity_cols": ["Case1", "Case2", "Case3", "Case4", "QC1"]}
    sample_type_row = pd.Series(
        {
            "Mz/RT": "Sample_Type",
            "Case1": "case",
            "Case2": "case",
            "Case3": "case",
            "Case4": "case",
            "QC1": "qc",
        }
    )
    return df, col_info, sample_type_row


def test_annotates_base_and_sodium_adduct_rows() -> None:
    df, col_info, sample_type_row = _annotation_input()

    annotated, stats, source = DegeneracyAnnotator().annotate(
        df,
        col_info=col_info,
        sample_type_row=sample_type_row,
        ppm_tolerance=20,
        rt_tolerance=0.05,
        correlation_threshold=0.8,
        min_correlation_points=3,
        adduct_table_file=None,
    )

    base_row = annotated[annotated["Mz/RT"] == "242.1191/10.99"].iloc[0]
    adduct_row = annotated[annotated["Mz/RT"] == "264.1010/11.00"].iloc[0]
    singleton_row = annotated[annotated["Mz/RT"] == "300.0000/20.00"].iloc[0]

    assert source == "built-in"
    assert stats["degeneracy_matches"] == 1
    assert stats["degeneracy_base_count"] == 1
    assert stats["degeneracy_adduct_count"] == 1
    assert base_row["Degeneracy_Group_Role"] == "base"
    assert base_row["Degeneracy_Type"] == "[M+H]+"
    assert adduct_row["Degeneracy_Group_Role"] == "adduct"
    assert adduct_row["Degeneracy_Type"] == "[M+Na]+"
    assert adduct_row["Degeneracy_Base_mz"] == "242.1191"
    assert float(adduct_row["Degeneracy_Pearson_R"]) >= 0.99
    assert singleton_row["Degeneracy_Group_Role"] == "singleton"


def test_rejects_low_correlation_adduct_like_pairs() -> None:
    df = pd.DataFrame(
        {
            "Mz/RT": ["242.1191/10.99", "264.1010/11.00"],
            "Case1": [100, 400],
            "Case2": [200, 300],
            "Case3": [300, 200],
            "Case4": [400, 100],
            "_mz": [242.1191, 264.1010],
            "_rt": [10.99, 11.00],
            "_total_intensity": [1000, 1000],
        }
    )

    annotated, stats, _ = DegeneracyAnnotator().annotate(
        df,
        col_info={"intensity_cols": ["Case1", "Case2", "Case3", "Case4"]},
        sample_type_row=pd.Series(
            {"Case1": "case", "Case2": "case", "Case3": "case", "Case4": "case"}
        ),
        ppm_tolerance=20,
        rt_tolerance=0.05,
        correlation_threshold=0.8,
        min_correlation_points=3,
        adduct_table_file=None,
    )

    pair_row = annotated[annotated["Mz/RT"] == "264.1010/11.00"].iloc[0]
    assert stats["degeneracy_adduct_count"] == 0
    assert stats["degeneracy_corr_rejected"] == 1
    assert pair_row["Degeneracy_Group_Role"] == "singleton"


def test_returns_empty_stats_for_empty_input() -> None:
    annotated, stats, source = DegeneracyAnnotator().annotate(
        pd.DataFrame(columns=["Mz/RT", "_mz", "_rt", "_total_intensity"]),
        col_info={"intensity_cols": []},
        sample_type_row=pd.Series(dtype=object),
        ppm_tolerance=20,
        rt_tolerance=0.05,
        correlation_threshold=0.8,
        min_correlation_points=3,
        adduct_table_file=None,
    )

    assert annotated.empty
    assert stats == {
        "degeneracy_annotation_enabled": True,
        "degeneracy_matches": 0,
        "degeneracy_groups": 0,
        "degeneracy_base_count": 0,
        "degeneracy_adduct_count": 0,
        "degeneracy_corr_rejected": 0,
    }
    assert source == "empty"


def test_loads_valid_custom_adduct_table(project_temp_dir: TempDirFactory) -> None:
    df, col_info, sample_type_row = _annotation_input()
    with project_temp_dir("degeneracy-adducts-") as temp_dir:
        custom_table = temp_dir / "custom_adducts.csv"
        custom_table.write_text("To,Delta_Da\n[M+Custom]+,21.981943\n", encoding="utf-8")

        annotated, stats, source = DegeneracyAnnotator().annotate(
            df,
            col_info=col_info,
            sample_type_row=sample_type_row,
            ppm_tolerance=20,
            rt_tolerance=0.05,
            correlation_threshold=0.8,
            min_correlation_points=3,
            adduct_table_file=str(custom_table),
        )

    adduct_row = annotated[annotated["Mz/RT"] == "264.1010/11.00"].iloc[0]
    assert source == str(custom_table)
    assert stats["degeneracy_adduct_count"] == 1
    assert adduct_row["Degeneracy_Type"] == "[M+Custom]+"


def test_falls_back_to_built_in_table_for_missing_or_invalid_custom_table(
    project_temp_dir: TempDirFactory,
) -> None:
    df, col_info, sample_type_row = _annotation_input()
    with project_temp_dir("degeneracy-adducts-") as temp_dir:
        invalid_table = temp_dir / "invalid_adducts.csv"
        invalid_table.write_text("name,delta\nbad,21.981943\n", encoding="utf-8")

        for custom_table in [temp_dir / "missing.csv", invalid_table]:
            annotated, stats, source = DegeneracyAnnotator().annotate(
                df,
                col_info=col_info,
                sample_type_row=sample_type_row,
                ppm_tolerance=20,
                rt_tolerance=0.05,
                correlation_threshold=0.8,
                min_correlation_points=3,
                adduct_table_file=str(custom_table),
            )

            adduct_row = annotated[annotated["Mz/RT"] == "264.1010/11.00"].iloc[0]
            assert source == "built-in"
            assert stats["degeneracy_adduct_count"] == 1
            assert adduct_row["Degeneracy_Type"] == "[M+Na]+"


def test_falls_back_to_built_in_table_when_custom_table_reader_raises(
    monkeypatch,
    project_temp_dir: TempDirFactory,
) -> None:
    df, col_info, sample_type_row = _annotation_input()
    with project_temp_dir("degeneracy-adducts-") as temp_dir:
        corrupt_workbook = temp_dir / "corrupt_adducts.xlsx"
        corrupt_workbook.write_bytes(b"not a valid workbook")

        def raise_parser_error(*_args, **_kwargs):
            raise RuntimeError("workbook parser failed")

        monkeypatch.setattr(pd, "read_excel", raise_parser_error)

        annotated, stats, source = DegeneracyAnnotator().annotate(
            df,
            col_info=col_info,
            sample_type_row=sample_type_row,
            ppm_tolerance=20,
            rt_tolerance=0.05,
            correlation_threshold=0.8,
            min_correlation_points=3,
            adduct_table_file=str(corrupt_workbook),
        )

    adduct_row = annotated[annotated["Mz/RT"] == "264.1010/11.00"].iloc[0]
    assert source == "built-in"
    assert stats["degeneracy_adduct_count"] == 1
    assert adduct_row["Degeneracy_Type"] == "[M+Na]+"


def test_duplicate_remover_facade_preserves_degeneracy_process_contract() -> None:
    df = pd.DataFrame(
        {
            "Mz/RT": ["Sample_Type", "242.1191/10.99", "264.1010/11.00", "300.0000/20.00"],
            "Case1": ["case", 100, 50, 20],
            "Case2": ["case", 200, 100, 25],
            "Case3": ["case", 400, 200, 22],
            "Case4": ["case", 800, 400, 19],
            "QC1": ["qc", 51000, 38500, 11500],
        }
    )

    result = DuplicateRemover().process(
        df,
        mz_tolerance_ppm=5,
        rt_tolerance=0.1,
        enable_degeneracy_annotation=True,
        degeneracy_ppm_tolerance=20,
        degeneracy_rt_tolerance=0.05,
    )

    assert result.success
    assert result.statistics["degeneracy_annotation_enabled"] is True
    assert result.statistics["degeneracy_adduct_count"] == 1
    assert result.statistics["degeneracy_base_count"] == 1
    assert result.metadata["degeneracy_adduct_table_source"] == "built-in"
    assert {
        "Degeneracy_Type",
        "Degeneracy_Description",
        "Degeneracy_Base_mz",
        "Degeneracy_PPM_Error",
        "Degeneracy_Group_Role",
        "Degeneracy_Group_ID",
        "Degeneracy_Pearson_R",
    }.issubset(result.data.columns)

    base_row = result.data[result.data["Mz/RT"] == "242.1191/10.99"].iloc[0]
    adduct_row = result.data[result.data["Mz/RT"] == "264.1010/11.00"].iloc[0]
    assert base_row["Degeneracy_Group_Role"] == "base"
    assert adduct_row["Degeneracy_Group_Role"] == "adduct"
    assert adduct_row["Degeneracy_Type"] == "[M+Na]+"
