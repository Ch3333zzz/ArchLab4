from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

"""Config loader and validator.

Provides `load_config` which accepts either a path to a YAML file,
a dictionary or None and returns a normalized configuration dict
using DEFAULTS for missing values.
"""


DEFAULTS: dict[str, Any] = {
    "mmio_in": 0xFFF0,
    "mmio_out": 0xFFF1,
    "mem_cells": 65536,
    "pause_tick": None,
    "tick_limit": 100000,
    "string_pool_cells": 1024,
    "lenient_log": False,
}


class ConfigError(ValueError):
    """Raised when configuration is invalid or cannot be loaded."""

    pass


def _convert_types(cfg: dict[str, Any]) -> None:
    """Normalize types for configuration values in-place.

    Raises ConfigError on conversion failure.
    """
    try:
        # mmio_in
        v = cfg.get("mmio_in")
        if v is None:
            cfg["mmio_in"] = None
        else:
            cfg["mmio_in"] = int(v)

        # mmio_out
        v = cfg.get("mmio_out")
        if v is None:
            cfg["mmio_out"] = None
        else:
            cfg["mmio_out"] = int(v)

        # mem_cells (required to exist by callers)
        cfg["mem_cells"] = int(cfg.get("mem_cells", DEFAULTS["mem_cells"]))

        # pause_tick
        v = cfg.get("pause_tick")
        if v is None:
            cfg["pause_tick"] = None
        else:
            cfg["pause_tick"] = int(v)

        # tick_limit
        tl = cfg.get("tick_limit", DEFAULTS["tick_limit"])
        # tl should not be None because DEFAULTS provides a value
        cfg["tick_limit"] = int(tl)

        # string_pool_cells
        spc = cfg.get("string_pool_cells", None)
        if spc is None:
            cfg["string_pool_cells"] = int(DEFAULTS["string_pool_cells"])
        else:
            cfg["string_pool_cells"] = int(spc)

        # lenient_log (bool coercion)
        cfg["lenient_log"] = bool(cfg.get("lenient_log", DEFAULTS["lenient_log"]))
    except Exception as e:
        msg = f"Bad types in config: {e}"
        raise ConfigError(msg) from e


def _validate_cfg(cfg: dict[str, Any]) -> None:
    """Perform semantic validation on normalized config dict.

    Raises ConfigError on invalid values.
    """
    if cfg["mem_cells"] <= 0:
        msg = "mem_cells must be positive"
        raise ConfigError(msg)

    # Check mmio_in range
    if cfg["mmio_in"] is not None and not (0 <= cfg["mmio_in"] < cfg["mem_cells"]):
        max_idx = cfg["mem_cells"] - 1
        msg = f"mmio_in ({cfg['mmio_in']}) out of memory range (0..{max_idx})"
        raise ConfigError(msg)

    # Check mmio_out range
    if cfg["mmio_out"] is not None and not (0 <= cfg["mmio_out"] < cfg["mem_cells"]):
        max_idx = cfg["mem_cells"] - 1
        msg = f"mmio_out ({cfg['mmio_out']}) out of memory range (0..{max_idx})"
        raise ConfigError(msg)

    if cfg["pause_tick"] is not None and cfg["pause_tick"] < 0:
        msg = "pause_tick must be non-negative or null"
        raise ConfigError(msg)

    if cfg["string_pool_cells"] < 0 or cfg["string_pool_cells"] > cfg["mem_cells"]:
        msg = "string_pool_cells must be in range 0..mem_cells"
        raise ConfigError(msg)

    if not isinstance(cfg["lenient_log"], bool):
        msg = "lenient_log must be boolean"
        raise ConfigError(msg)


def load_config(path_or_dict: str | dict[str, Any] | None = None) -> dict[str, Any]:
    """Load and normalize configuration.

    Accepts:
      - None -> returns DEFAULTS copy
      - dict -> overlay DEFAULTS with provided dict
      - str (path) -> load YAML and overlay DEFAULTS

    Returns a normalized dict or raises ConfigError.
    """
    if path_or_dict is None:
        cfg: dict[str, Any] = dict(DEFAULTS)
    elif isinstance(path_or_dict, dict):
        cfg = dict(DEFAULTS)
        cfg.update(path_or_dict)
    elif isinstance(path_or_dict, str):
        p = Path(path_or_dict)
        if not p.exists():
            msg = f"Config file not found: {path_or_dict}"
            raise ConfigError(msg)
        try:
            with p.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            msg = f"Failed to load config file {path_or_dict}: {e}"
            raise ConfigError(msg) from e
        if not isinstance(data, dict):
            msg = f"Config file {path_or_dict} does not contain a mapping"
            raise ConfigError(msg)
        cfg = dict(DEFAULTS)
        cfg.update(data)
    else:
        msg = "Unsupported config input"
        raise ConfigError(msg)

    # convert types and validate semantics
    _convert_types(cfg)
    _validate_cfg(cfg)

    return cfg
