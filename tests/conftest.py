from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import shutil
import uuid

import pytest

from tests.testing_markers import classify_test_markers


ROOT = Path(__file__).resolve().parents[1]
TMP_ROOT = ROOT / ".tmp"
PROJECT_TEST_TEMP_ROOT = TMP_ROOT / "tests"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply suite markers from the central path-based classification rules."""
    root = ROOT.resolve()
    for item in items:
        item_fs_path = getattr(item, "path", None)
        if item_fs_path is None:
            item_fs_path = Path(item.fspath)
        else:
            item_fs_path = Path(item_fs_path)

        try:
            item_path = item_fs_path.resolve().relative_to(root)
        except ValueError:
            item_path = item_fs_path

        for marker in sorted(classify_test_markers(item_path)):
            item.add_marker(getattr(pytest.mark, marker))


@pytest.fixture(scope="session")
def project_temp_root() -> Path:
    PROJECT_TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    return PROJECT_TEST_TEMP_ROOT


@pytest.fixture
def project_temp_dir(project_temp_root: Path):
    @contextmanager
    def _factory(prefix: str = "case-"):
        temp_dir = project_temp_root / f"{prefix}{uuid.uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=False)
        try:
            yield temp_dir
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return _factory
