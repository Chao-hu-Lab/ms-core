"""MS-specific preprocessing — data organization, ISTD marking, deduplication, quality filtering.

Source: ms-preprocessing-toolkit
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ms_core.preprocessing.data_organizer import DataOrganizer
    from ms_core.preprocessing.istd_marker import ISTDMarker
    from ms_core.preprocessing.duplicate_remover import DuplicateRemover
    from ms_core.preprocessing.ms_quality_filter import FeatureFilter
    from ms_core.preprocessing.settings import (
        DataOrganizerConfig,
        ISTDConfig,
        DuplicateRemovalConfig,
        FeatureFilterConfig,
        Settings,
    )

__all__ = [
    "DataOrganizer",
    "ISTDMarker",
    "DuplicateRemover",
    "FeatureFilter",
    "DataOrganizerConfig",
    "ISTDConfig",
    "DuplicateRemovalConfig",
    "FeatureFilterConfig",
    "Settings",
]

_LAZY_MAP: dict[str, str] = {
    "DataOrganizer": "ms_core.preprocessing.data_organizer",
    "ISTDMarker": "ms_core.preprocessing.istd_marker",
    "DuplicateRemover": "ms_core.preprocessing.duplicate_remover",
    "FeatureFilter": "ms_core.preprocessing.ms_quality_filter",
    "DataOrganizerConfig": "ms_core.preprocessing.settings",
    "ISTDConfig": "ms_core.preprocessing.settings",
    "DuplicateRemovalConfig": "ms_core.preprocessing.settings",
    "FeatureFilterConfig": "ms_core.preprocessing.settings",
    "Settings": "ms_core.preprocessing.settings",
}


def __getattr__(name: str) -> object:
    if name in _LAZY_MAP:
        import importlib

        module = importlib.import_module(_LAZY_MAP[name])
        obj = getattr(module, name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
