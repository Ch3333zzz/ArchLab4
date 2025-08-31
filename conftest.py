import yaml
from pathlib import Path
import pytest

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "golden_test(pattern): parameterize test with YAML files matching pattern"
    )

def _iter_marker_patterns(node):
    for m in node.iter_markers(name="golden_test"):
        if m.args:
            yield m.args[0]
        else:
            yield "golden/*.yaml"

def pytest_generate_tests(metafunc):
    
    if "golden" not in metafunc.fixturenames:
        return

    patterns = list(_iter_marker_patterns(metafunc.definition))
    if not patterns:
        patterns = ["golden/*.yaml"]

    root = Path(metafunc.config.rootpath)

    files = []
    for pat in patterns:
        files.extend(sorted(root.glob(pat)))

    if not files:
        return

    params = []
    ids = []
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
