"""Golden-test runner for the lisp -> VM pipeline.

This test loads golden YAML records and runs the compilation -> VM pipeline,
then compares produced outputs (stdout, memory, code, hex, binary, logs)
against the expectations in the golden files.
"""

from __future__ import annotations

import logging
import os
import struct
import tempfile
from parser import Compiler, parse, tokenize
from pathlib import Path
from typing import Any

import processor
import pytest
from isa import INSTR_SIZE, decode_instr, mnemonic
from processor import ControlUnit, Datapath


@pytest.mark.golden_test("golden/*.yaml")
def test_translator_and_vm(golden: Any, caplog: Any) -> None:  # noqa: C901
    """Run one golden record: compile, run and compare outputs.

    The function intentionally bundles a number of checks (stdout, log,
    binary, disassembly, memory) to match the teammate's golden runner.
    """
    logging.getLogger().setLevel(logging.DEBUG)

    src_spec = golden.get("source")
    if not src_spec and "in_source" in golden:
        src_spec = {"language": "lisp", "code": golden["in_source"]}

    input_schedule = None
    if "input_schedule" in golden:
        input_schedule = golden["input_schedule"]
    elif "in_stdin" in golden:
        # turn in_stdin string into schedule: deliver chars at ticks 1,2,3...
        s = golden["in_stdin"]
        schedule = []
        tick = 1
        for ch in s:
            schedule.append((tick, ch))
            tick += 1
        input_schedule = schedule

    cfg = None
    # Prefer explicit config blocks or in_config
    if "config" in golden:
        cfg = golden["config"]
    elif "in_config" in golden:
        cfg = golden["in_config"]

    # Build code_bytes/data_bytes
    code_bytes = b""
    data_bytes = b""
    if src_spec:
        lang = src_spec.get("language", "lisp")
        if lang == "lisp":
            src_code = src_spec.get("code", "")
            toks = tokenize(src_code)
            ast = parse(toks)
            comp = Compiler(ast)
            comp.compile()
            code_bytes = bytes(comp.code)
            data_bytes = bytes(comp.data)
        else:
            pytest.skip(f"Unsupported source language in golden: {lang}")
    else:
        pytest.skip("No source provided in golden record")

    # create a temporary working dir to store artifacts (optional files)
    with tempfile.TemporaryDirectory() as tmp:
        # Use tmp for all artifact paths to avoid cross-test interference.
        code_path = os.path.join(tmp, "code.bin")
        data_path = os.path.join(tmp, "data.bin")
        code_hex_path = os.path.join(tmp, "code.hex")

        # processor log and optional debug artifacts are placed in tmp too
        proc_log_path = os.path.join(tmp, "processor.log")
        proc_bin_path = os.path.join(tmp, "out.bin")
        proc_hex_path = os.path.join(tmp, "out.hex")

        # persist produced code/data for golden checkers that expect files
        with open(code_path, "wb") as code_f:
            code_f.write(code_bytes)
        with open(data_path, "wb") as data_f:
            data_f.write(data_bytes)

        # produce a simple disassembly (code_hex) for human-readable
        # comparison. Left-align numeric offset (tests compare stripped text).
        code_hex_lines = []
        pc = 0
        code_len = len(code_bytes)
        while pc + INSTR_SIZE <= code_len:
            try:
                opcode, arg = decode_instr(code_bytes, pc)
                instr_bytes = code_bytes[pc : pc + INSTR_SIZE]
                hexbytes = instr_bytes.hex().upper()
                try:
                    mnem = mnemonic(opcode, arg)
                except Exception:
                    mnem = f"{opcode.name} {arg}"
                code_hex_lines.append(f"{pc} - {hexbytes} - {mnem}")
            except Exception as e:
                # if decode fails — dump remainder as hex
                rest = code_bytes[pc:].hex().upper()
                code_hex_lines.append(f"{pc} - {rest} - <decode error: {e}>")
                break
            pc += INSTR_SIZE
        code_hex = "\n".join(code_hex_lines)
        with open(code_hex_path, "w", encoding="utf-8") as hex_f:
            hex_f.write(code_hex)

        # Initialize processor logging so that processor.log gets created and
        # filled. Tests expect deterministic debug output, so enable debug
        # logging for golden runs.
        processor.init_logging(logfile=proc_log_path, debug=True, console=False)

        # Prepare Datapath (was VM)
        mmio_in = cfg.get("mmio_in") if isinstance(cfg, dict) else None
        mmio_out = cfg.get("mmio_out") if isinstance(cfg, dict) else None
        mem_cells = cfg.get("mem_cells") if isinstance(cfg, dict) else None
        tick_limit = cfg.get("tick_limit") if isinstance(cfg, dict) else None
        pause_tick = cfg.get("pause_tick") if isinstance(cfg, dict) else None
        string_pool_cells = cfg.get("string_pool_cells") if isinstance(cfg, dict) else None
        lenient_log = cfg.get("lenient_log") if isinstance(cfg, dict) else None

        # mark dp/cu as Any to avoid mypy complaining about untyped methods in processor.py
        dp: Any = Datapath(
            code_bytes,
            data_bytes,
            mem_cells=mem_cells or 65536,
            mmio_in=mmio_in if mmio_in is not None else 0xFFF0,
            mmio_out=mmio_out if mmio_out is not None else 0xFFF1,
            tick_limit=tick_limit or 100000,
            pause_tick=pause_tick,
            string_pool_cells=(string_pool_cells if string_pool_cells is not None else 1024),
            lenient_log=(lenient_log if lenient_log is not None else False),
        )

        if input_schedule:
            # ensure schedule is list of (tick, char)
            dp.schedule_input(input_schedule)

        cu: Any = ControlUnit(dp)

        # run datapath
        out, ticks, state = cu.run()

        # Read processor.log if exists (many of your tools log to this file)
        proc_log_text: str = ""
        if Path(proc_log_path).exists():
            try:
                with open(proc_log_path, encoding="utf-8") as log_f:
                    proc_log_text = log_f.read()
            except Exception:
                proc_log_text = ""

        # If processor produced out.bin/out.hex in debug mode, read them now
        proc_out_code_bytes: bytes | None = None
        proc_out_code_hex: str | None = None
        if Path(proc_bin_path).exists():
            try:
                with open(proc_bin_path, "rb") as proc_bin_f:
                    proc_out_code_bytes = proc_bin_f.read()
            except Exception:
                proc_out_code_bytes = None
        if Path(proc_hex_path).exists():
            try:
                with open(proc_hex_path, encoding="utf-8") as proc_hex_f:
                    proc_out_code_hex = proc_hex_f.read()
            except Exception:
                proc_out_code_hex = None

        try:
            root = logging.getLogger()
            for h in list(root.handlers):
                try:
                    h.flush()
                    h.close()
                except Exception:
                    pass
                try:
                    root.removeHandler(h)
                except Exception:
                    pass
            logging.shutdown()
        except Exception:
            # don't let logging cleanup break tests
            pass

        expect = golden.get("out") or golden.get("expect") or {}

        # helper to produce long mismatch messages
        def _mismatch(msg_title: str, got_text: str, expected_text: str) -> str:
            return f"{msg_title}\n--- got ---\n{got_text}\n--- expected ---\n{expected_text}"

        # 1) code bytes (binary) if provided in golden.out.out_code (YAML !!binary)
        if "out_code" in expect:
            expected_code = expect["out_code"]
            assert isinstance(expected_code, (bytes, bytearray)), "golden.out_code must be binary"
            assert bytes(expected_code) == code_bytes, "machine code bytes mismatch"

        # if processor produced out.bin verify it matches
        if proc_out_code_bytes is not None:
            error = "out.bin must be binary"
            assert isinstance(proc_out_code_bytes, (bytes, bytearray)), error
            assert bytes(proc_out_code_bytes) == code_bytes, "out.bin doesn't match generated code_bytes"

        if "out_data" in expect:
            expected_data = expect["out_data"]
            assert isinstance(expected_data, (bytes, bytearray)), "golden.out_data must be binary"
            assert bytes(expected_data) == data_bytes, "data section mismatch"

        # 2) code hex
        if "out_code_hex" in expect:
            exp_code_hex = expect["out_code_hex"].strip()
            got = code_hex.strip()
            if got != exp_code_hex:
                raise AssertionError(_mismatch("code hex mismatch", got, exp_code_hex))

        # if processor produced out.hex verify it matches
        if proc_out_code_hex is not None:
            got_code_hex = proc_out_code_hex or ""
            assert isinstance(got_code_hex, str), "out.hex must be a string"
            if got_code_hex.strip() != code_hex.strip():
                err = "out.hex doesn't match generated disassembly"
                raise AssertionError(err)

        # 3) stdout
        if "out_stdout" in expect:
            exp_stdout = expect["out_stdout"]
            if isinstance(exp_stdout, str):
                if out.strip() != exp_stdout.strip():
                    raise AssertionError(_mismatch("stdout mismatch", out, exp_stdout))

        # 4) ticks/state
        if "ticks" in expect:
            try:
                exp_ticks = int(expect["ticks"])
                assert ticks == exp_ticks, f"ticks mismatch: got {ticks} expected {exp_ticks}"
            except Exception as e:
                msg = f"bad expected ticks value: {expect['ticks']} ({e})"
                raise AssertionError(msg) from e

        if "state" in expect:
            assert state == expect["state"], f"state mismatch: got {state} expected {expect['state']}"

        # 5) processor log
        if "out_log" in expect:
            exp_log = expect["out_log"]
            # compare verbatim (you may want to normalize timestamps in real repo)
            if proc_log_text.strip() != exp_log.strip():
                got_snip = proc_log_text[:4000]
                exp_snip = exp_log[:4000]
                raise AssertionError(_mismatch("processor.log mismatch", got_snip, exp_snip))

        # 6) additional: if golden expect.memory provided as dict addr->int, verify
        if "memory" in expect:
            mem_expect = expect["memory"]
            assert isinstance(mem_expect, dict), "expect.memory must be dict"
            for k, v in mem_expect.items():
                addr = int(k)
                off = addr * 4
                if off < 0 or off + 4 > len(dp.memory):
                    err = f"memory addr {addr} out of bounds"
                    raise AssertionError(err)
                actual = struct.unpack("<i", dp.memory[off : off + 4])[0]
                exp_val = int(v)
                assert actual == exp_val, f"memory[{addr}] mismatch: got {actual} expected {exp_val}"
