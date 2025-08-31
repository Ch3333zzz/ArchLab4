import yaml
import os

DEFAULTS = {
    "mmio_in": 0xFFF0,
    "mmio_out": 0xFFF1,
    "mem_cells": 65536,
    "pause_tick": None,
    "tick_limit": 100000,
    "string_pool_cells": 1024,
    "lenient_log": False,
}

class ConfigError(ValueError):
    pass

def load_config(path_or_dict=None):

    if path_or_dict is None:
        cfg = dict(DEFAULTS)
    elif isinstance(path_or_dict, dict):
        cfg = dict(DEFAULTS)
        cfg.update(path_or_dict)
    elif isinstance(path_or_dict, str):
        if not os.path.exists(path_or_dict):
            raise ConfigError(f"Config file not found: {path_or_dict}")
        with open(path_or_dict, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg = dict(DEFAULTS)
        cfg.update(data)
    else:
        raise ConfigError("Unsupported config input")

    try:
        cfg["mmio_in"] = None if cfg["mmio_in"] is None else int(cfg["mmio_in"])
        cfg["mmio_out"] = None if cfg["mmio_out"] is None else int(cfg["mmio_out"])
        cfg["mem_cells"] = int(cfg["mem_cells"])
        cfg["pause_tick"] = None if cfg["pause_tick"] is None else int(cfg["pause_tick"])
        cfg["tick_limit"] = int(cfg.get("tick_limit", DEFAULTS["tick_limit"]))
        spc = cfg.get("string_pool_cells", None)
        cfg["string_pool_cells"] = int(spc) if spc is not None else int(DEFAULTS["string_pool_cells"])
        cfg["lenient_log"] = bool(cfg.get("lenient_log", DEFAULTS["lenient_log"]))
    except Exception as e:
        raise ConfigError("Bad types in config: " + str(e))

    if cfg["mem_cells"] <= 0:
        raise ConfigError("mem_cells must be positive")

    if cfg["mmio_in"] is not None and not (0 <= cfg["mmio_in"] < cfg["mem_cells"]):
        raise ConfigError(f"mmio_in ({cfg['mmio_in']}) out of memory range (0..{cfg['mem_cells']-1})")
    if cfg["mmio_out"] is not None and not (0 <= cfg["mmio_out"] < cfg["mem_cells"]):
        raise ConfigError(f"mmio_out ({cfg['mmio_out']}) out of memory range (0..{cfg['mem_cells']-1})")

    if cfg["pause_tick"] is not None and cfg["pause_tick"] < 0:
        raise ConfigError("pause_tick must be non-negative or null")
    if cfg["string_pool_cells"] < 0 or cfg["string_pool_cells"] > cfg["mem_cells"]:
        raise ConfigError("string_pool_cells must be in range 0..mem_cells")

    if not isinstance(cfg["lenient_log"], bool):
        raise ConfigError("lenient_log must be boolean")

    return cfg
