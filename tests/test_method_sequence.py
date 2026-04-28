from __future__ import annotations

from ms_core.preprocessing.data_organizer import InjectionInfo as DataOrganizerInjectionInfo
from ms_core.preprocessing.method_sequence import (
    InjectionInfo,
    extract_injection_rows_from_table,
    parse_injection_tables,
    parse_injection_volume_from_cells,
)


def test_data_organizer_reexports_method_sequence_injection_info() -> None:
    assert DataOrganizerInjectionInfo is InjectionInfo


def test_extract_injection_rows_from_table_parses_method_rows() -> None:
    rows = [
        ["ID", "File Name", "Instrument Method", "Inj Vol"],
        ["1", "ZBEE000070", "LC method A", "5"],
        ["2", "QC sample 4", "LC method B", "10"],
    ]

    injections = extract_injection_rows_from_table(rows)

    assert injections == [
        InjectionInfo(1, "ZBEE000070", "ZBEE000070", 5.0, "LC method A"),
        InjectionInfo(2, "QC sample 4", "QC sample 4", 10.0, "LC method B"),
    ]


def test_parse_injection_tables_renumbers_reused_bc_ids_by_source_order() -> None:
    target_table = [["ID", "File Name", "Instrument Method", "Inj Vol"]]
    target_table.extend(
        [
            [str((idx % 2) + 1), f"Tumor tissue BC22{idx:02d}_DNA", "LC method", "5"]
            for idx in range(6)
        ]
    )
    target_table.extend(
        [
            ["7", "not an injection", "LC method", "5"],
            ["8", "header continuation", "LC method", "5"],
            ["9", "page break", "LC method", "5"],
            ["10", "footer", "LC method", "5"],
        ]
    )

    injections = parse_injection_tables([target_table])

    assert [info.injection_order for info in injections] == [1, 2, 3, 4, 5, 6]
    assert [info.file_name for info in injections] == [
        "Tumor tissue BC2200_DNA",
        "Tumor tissue BC2201_DNA",
        "Tumor tissue BC2202_DNA",
        "Tumor tissue BC2203_DNA",
        "Tumor tissue BC2204_DNA",
        "Tumor tissue BC2205_DNA",
    ]


def test_parse_injection_volume_from_cells_ignores_large_numeric_ids() -> None:
    assert parse_injection_volume_from_cells(["1234", "ZBEE000070", "150", "7.5"]) == 7.5
