# tests/test_translator_and_vm.py
import contextlib
import io
import logging
import os
import tempfile
import struct

import pytest

from parser import tokenize, parse, Compiler
import processor
from processor import Datapath, ControlUnit, run_bytes
from isa import decode_instr, INSTR_SIZE, mnemonic

# marker: your test runner should provide this parametrization (same as colleague)
@pytest.mark.golden_test("golden/*.yaml")
def test_translator_and_vm(golden, caplog):
    """
    Golden-test runner for the lisp -> VM pipeline.

    Accepts YAMLs with either:
      - 'source': { language: 'lisp', code: '...' }, and optional input_schedule, config
      - OR 'in_source' / 'in_stdin' style (colleague style)
    And expected outputs in golden['out'] or golden['expect'].
    """
    logging.getLogger().setLevel(logging.DEBUG)

    # Normalise different possible YAML layouts used by different authors
    src_spec = golden.get("source")
    if not src_spec and "in_source" in golden:
        # colleague-style
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
    # Prefer explicit config blocks (user style) else use colleague in_config
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
        # Use tmp for all artifact paths to avoid cross-test interference
        code_path = os.path.join(tmp, "code.bin")
        data_path = os.path.join(tmp, "data.bin")
        code_hex_path = os.path.join(tmp, "code.hex")
        proc_log_path = os.path.join(tmp, "processor.log")  # processor uses this filename by default
        proc_bin_path = os.path.join(tmp, "out.bin")
        proc_hex_path = os.path.join(tmp, "out.hex")

        # persist produced code/data for golden checkers that expect files
        with open(code_path, "wb") as f:
            f.write(code_bytes)
        with open(data_path, "wb") as f:
            f.write(data_bytes)

        # produce a simple disassembly (code_hex) for human-readable comparison
        code_hex_lines = []
        pc = 0
        code_len = len(code_bytes)
        while pc + INSTR_SIZE <= code_len:
            try:
                opcode, arg = decode_instr(code_bytes, pc)
                instr_bytes = code_bytes[pc:pc + INSTR_SIZE]
                hexbytes = instr_bytes.hex().upper()
                try:
                    mnem = mnemonic(opcode, arg)
                except Exception:
                    mnem = f"{opcode.name} {arg}"
                # left-aligned numeric offset (tests compare stripped text)
                code_hex_lines.append(f"{pc} - {hexbytes} - {mnem}")
            except Exception as e:
                # if decode fails — dump remainder as hex
                rest = code_bytes[pc:].hex().upper()
                code_hex_lines.append(f"{pc} - {rest} - <decode error: {e}>")
                break
            pc += INSTR_SIZE
        code_hex = "\n".join(code_hex_lines)
        with open(code_hex_path, "w", encoding="utf-8") as f:
            f.write(code_hex)

        # Initialize processor logging so that processor.log gets created and filled.
        # Tests expect deterministic debug output, so enable debug logging for golden runs.
        processor.init_logging(logfile=proc_log_path, debug=True, console=False)

        # Prepare Datapath (was VM)
        mmio_in = cfg.get("mmio_in") if isinstance(cfg, dict) else None
        mmio_out = cfg.get("mmio_out") if isinstance(cfg, dict) else None
        mem_cells = cfg.get("mem_cells") if isinstance(cfg, dict) else None
        tick_limit = cfg.get("tick_limit") if isinstance(cfg, dict) else None
        pause_tick = cfg.get("pause_tick") if isinstance(cfg, dict) else None
        string_pool_cells = cfg.get("string_pool_cells") if isinstance(cfg, dict) else None
        lenient_log = cfg.get("lenient_log") if isinstance(cfg, dict) else None

        dp = Datapath(
            code_bytes,
            data_bytes,
            mem_cells=mem_cells or 65536,
            mmio_in=mmio_in if mmio_in is not None else 0xFFF0,
            mmio_out=mmio_out if mmio_out is not None else 0xFFF1,
            tick_limit=tick_limit or 100000,
            pause_tick=pause_tick,
            string_pool_cells=(string_pool_cells if string_pool_cells is not None else 1024),
            lenient_log=(lenient_log if lenient_log is not None else False)
        )

        if input_schedule:
            # ensure schedule is list of (tick, char)
            dp.schedule_input(input_schedule)

        cu = ControlUnit(dp)

        # run datapath
        out, ticks, state = cu.run()

        # Read processor.log if exists (many of your tools log to this file)
        proc_log_text = ""
        if os.path.exists(proc_log_path):
            try:
                with open(proc_log_path, "r", encoding="utf-8") as f:
                    proc_log_text = f.read()
            except Exception:
                proc_log_text = ""

        # If processor produced out.bin/out.hex in debug mode, read them now
        proc_out_code_bytes = None
        proc_out_code_hex = None
        if os.path.exists(proc_bin_path):
            try:
                with open(proc_bin_path, "rb") as f:
                    proc_out_code_bytes = f.read()
            except Exception:
                proc_out_code_bytes = None
        if os.path.exists(proc_hex_path):
            try:
                with open(proc_hex_path, "r", encoding="utf-8") as f:
                    proc_out_code_hex = f.read()
            except Exception:
                proc_out_code_hex = None

        # IMPORTANT: make sure logging handlers are closed before leaving tmp dir (Windows locks files)
        try:
            # flush/close and remove any handlers attached to root logger
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
            # final shutdown to be extra-safe
            logging.shutdown()
        except Exception:
            # don't let logging cleanup break tests
            pass

        # Now compare with golden expectations. Support both 'out' and 'expect' styles.
        expect = golden.get("out") or golden.get("expect") or {}

        # 1) code bytes (binary) if provided in golden.out.out_code (YAML !!binary)
        if "out_code" in expect:
            expected_code = expect["out_code"]
            assert isinstance(expected_code, (bytes, bytearray)), "golden.out_code must be binary"
            assert bytes(expected_code) == code_bytes, "machine code bytes mismatch"

        # additionally: if processor produced out.bin verify it matches
        if proc_out_code_bytes is not None:
            assert isinstance(proc_out_code_bytes, (bytes, bytearray)), "out.bin must be binary"
            assert bytes(proc_out_code_bytes) == code_bytes, "out.bin doesn't match generated code_bytes"

        if "out_data" in expect:
            expected_data = expect["out_data"]
            assert isinstance(expected_data, (bytes, bytearray)), "golden.out_data must be binary"
            assert bytes(expected_data) == data_bytes, "data section mismatch"

        # 2) code hex
        if "out_code_hex" in expect:
            exp_code_hex = expect["out_code_hex"].strip()
            got = code_hex.strip()
            assert got == exp_code_hex, f"code hex mismatch\n--- got ---\n{got}\n--- expected ---\n{exp_code_hex}"

        # additionally: if processor produced out.hex verify it matches
        if proc_out_code_hex is not None:
            got_code_hex = proc_out_code_hex
            if got_code_hex is None:
                got_code_hex = ""
            assert isinstance(got_code_hex, str), "out.hex must be a string"
            # normalize whitespace
            assert got_code_hex.strip() == code_hex.strip(), "out.hex doesn't match generated disassembly"

        # 3) stdout
        if "out_stdout" in expect:
            exp_stdout = expect["out_stdout"]
            # normalize trailing whitespace to be less brittle
            if isinstance(exp_stdout, str):
                if out.strip() != exp_stdout.strip():
                    raise AssertionError(f"stdout mismatch\n--- got ---\n{out!r}\n--- expected ---\n{exp_stdout!r}")

        # 4) ticks/state
        if "ticks" in expect:
            try:
                exp_ticks = int(expect["ticks"])
                assert ticks == exp_ticks, f"ticks mismatch: got {ticks} expected {exp_ticks}"
            except Exception as e:
                raise AssertionError(f"bad expected ticks value: {expect['ticks']} ({e})")

        if "state" in expect:
            assert state == expect["state"], f"state mismatch: got {state} expected {expect['state']}"

        # 5) processor log
        if "out_log" in expect:
            exp_log = expect["out_log"]
            # compare verbatim (you may want to normalize timestamps in real repo)
            if proc_log_text.strip() != exp_log.strip():
                raise AssertionError(f"processor.log mismatch\n--- got ---\n{proc_log_text[:4000]!r}\n--- expected ---\n{exp_log[:4000]!r}")

        # 6) additional: if golden expect.memory provided as dict addr->int, verify
        if "memory" in expect:
            mem_expect = expect["memory"]
            assert isinstance(mem_expect, dict), "expect.memory must be dict"
            for k, v in mem_expect.items():
                addr = int(k)
                off = addr * 4
                if off < 0 or off + 4 > len(dp.memory):
                    raise AssertionError(f"memory addr {addr} out of bounds")
                actual = struct.unpack("<i", dp.memory[off:off+4])[0]
                exp_val = int(v)
                assert actual == exp_val, f"memory[{addr}] mismatch: got {actual} expected {exp_val}"
