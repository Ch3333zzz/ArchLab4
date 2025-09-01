"""File for tests."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml


def pytest_configure(config: Any) -> None:
    """Configure the tests."""
    config.addinivalue_line(
        "markers",
        "golden_test(pattern): parameterize test with YAML files matching pattern",
    )


def _iter_marker_patterns(node: Any) -> Iterator[str]:
    """Yield pattern strings from golden_test markers on `node`."""
    for m in node.iter_markers(name="golden_test"):
        if m.args:
            yield m.args[0]
        else:
            yield "golden/*.yaml"


def pytest_generate_tests(metafunc: Any) -> None:
    """Generate tests (parametrization) from YAML golden files."""
    if "golden" not in metafunc.fixturenames:
        return

    patterns: list[str] = list(_iter_marker_patterns(metafunc.definition))
    if not patterns:
        patterns = ["golden/*.yaml"]

    root = Path(metafunc.config.rootpath)

    files: list[Path] = []
    for pat in patterns:
        files.extend(sorted(root.glob(pat)))

    if not files:
        return

    params: list[dict[str, Any]] = []
    ids: list[str] = []
    for p in files:
        try:
            with p.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            data = {"__yaml_load_error__": str(e), "__path__": str(p)}
        if isinstance(data, dict):
            data.setdefault("__path__", str(p))
            data.setdefault("__name__", p.name)
        params.append(data)
        ids.append(p.name)

    metafunc.parametrize("golden", params, ids=ids)
