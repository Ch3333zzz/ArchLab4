"""Processor (Datapath + ControlUnit) and CLI wrapper.

Provides VM execution, logging initialization and optional debug output
files (out.bin / out.hex) emitted when debug logging is enabled.
"""

from __future__ import annotations

import logging
import struct
import sys
from pathlib import Path
from typing import Any, TextIO

from config import ConfigError, load_config
from isa import INSTR_SIZE, OpCode, decode_instr, mnemonic

LOGFILE = "processor.log"


def init_logging(logfile: str = LOGFILE, debug: bool = False, console: bool = False) -> None:
    """Configure root logger to write to `logfile`.

    If debug=True set DEBUG level. If console=True also echo logs to stdout.

    NOTE: when debug=True we use a compact log format without timestamp so that
    entries look like:
        DEBUG root:processor.py:197 Datapath: heap_ptr initialized to word 1027
    which matches the requested style.
    """
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    lvl = logging.DEBUG if debug else logging.CRITICAL
    root.setLevel(lvl)

    # Choose formats. In debug mode use compact format (no timestamp) so lines match:
    if debug:
        file_fmt = "%(levelname)s %(name)s:%(filename)s:%(lineno)d %(message)s"
    else:
        file_fmt = "%(levelname)-5s %(message)s"

    # Custom formatter that indents all but the first formatted record.
    # This reproduces the example where the first debug line appears without
    # indentation and subsequent lines are indented for readability.
    class _IndentOnceFormatter(logging.Formatter):
        def __init__(self, fmt: str | None = None):
            super().__init__(fmt)
            self._seen_first = False

        def format(self, record: logging.LogRecord) -> str:
            s = super().format(record)
            # Only the very first formatted record is left unindented; all
            # subsequent records are prefixed with 4 spaces to match the
            # requested style.
            if not self._seen_first:
                self._seen_first = True
                return s
            return "    " + s

    # always create a FileHandler even when not debug to allow easier inspection if asked
    fh = logging.FileHandler(logfile, mode="w", encoding="utf-8")
    fh.setLevel(lvl)
    if debug:
        fh.setFormatter(_IndentOnceFormatter(file_fmt))
    else:
        fh.setFormatter(logging.Formatter(file_fmt))
    root.addHandler(fh)

    if debug and console:
        # Console: keep concise but readable
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(lvl)
        # console shows level and message (no filename by default)
        ch.setFormatter(logging.Formatter("%(levelname)5s %(message)s"))
        root.addHandler(ch)


# --- debug output helpers (split out to reduce complexity) ---
def _flush_logging_handlers() -> None:
    try:
        for h in list(logging.getLogger().handlers):
            try:
                if hasattr(h, "flush"):
                    h.flush()
            except Exception:
                pass
    except Exception:
        pass


def _write_out_bin(code_bytes: bytes) -> None:
    try:
        with open("out.bin", "wb") as f:
            f.write(code_bytes)
    except Exception as e:
        logging.debug("Failed to write out.bin: %s", e)


def _write_out_hex(code_bytes: bytes) -> None:
    try:
        code_hex_lines: list[str] = []
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
                rest = code_bytes[pc:].hex().upper()
                code_hex_lines.append(f"{pc} - {rest} - <decode error: {e}>")
                break
            pc += INSTR_SIZE
        code_hex = "\n".join(code_hex_lines)
        with open("out.hex", "w", encoding="utf-8") as f:
            f.write(code_hex)
    except Exception as e:
        logging.debug("Failed to write out.hex: %s", e)


def _write_debug_out_files(code_bytes: bytes) -> None:
    _flush_logging_handlers()
    _write_out_bin(code_bytes)
    _write_out_hex(code_bytes)


class Datapath:
    """Datapath (memory + registers + MMIO buffers) for the VM."""

    # configuration
    code_bytes: bytes
    data_bytes: bytes
    mem_cells: int
    mem_bytes: int
    memory: bytearray

    string_pool_cells: int
    lenient_log: bool

    code_start: int
    code_len: int

    PC: int
    ACC: int

    # runtime-stack
    runtime_base: int
    runtime_len: int
    SP: int  # physical next-free word index for runtime stack (word index)
    FP: int  # logical frame pointer (index into logical runtime stack)
    # frame-stack
    frame_stack_cells: int
    FSP: int  # physical next-free index for frame-stack (word index)
    # call-stack
    call_stack_cells: int
    CP: int  # physical next-free index for call-stack (word index)
    interrupt_stack_cells: int  # interrupt-stack (ISP)
    ISP: int  # physical next-free index for interrupt-stack (word index)

    tick: int
    interrupt_ready: bool
    interruption_allowed: bool
    interruption_vector: int | None
    in_interruption: bool
    input_schedule: list[tuple[int, str]]
    input_closed: bool
    mmio_in: int
    mmio_out: int
    mmio_in_buffer: list[int]
    mmio_out_buffer: list[int]
    mmio_in_cell: int
    output_buffer: list[int]
    tick_limit: int
    pause_tick: int | None
    last_interrupt_source: str | None

    def __init__(
        self,
        code_bytes: bytes,
        data_bytes: bytes,
        mem_cells: int = 65536,
        mmio_in: int = 0xFFF0,
        mmio_out: int = 0xFFF1,
        tick_limit: int = 100000,
        pause_tick: int | None = None,
        string_pool_cells: int = 1024,
        call_stack_cells: int = 1024,
        frame_stack_cells: int = 256,
        interrupt_stack_cells: int = 64,
        lenient_log: bool = False,
    ) -> None:
        """Initialize Datapath state and memory layout."""
        # keep raw code/data
        self.code_bytes = code_bytes or b""
        self.data_bytes = data_bytes or b""

        # memory: mem_cells words (4 bytes each)
        self.mem_cells = int(mem_cells)
        self.mem_bytes = self.mem_cells * 4
        self.memory = bytearray(b"\x00" * self.mem_bytes)

        # write data bytes at start of memory (offset 0)
        data_len = len(self.data_bytes)
        if data_len > self.mem_bytes:
            err = "Initial data doesn't fit into memory"
            raise MemoryError(err)
        self.memory[0:data_len] = self.data_bytes

        # string pool size (in words)
        self.string_pool_cells = int(string_pool_cells)

        # lenient logging flag
        self.lenient_log = bool(lenient_log)

        # write code bytes right after data
        code_len = len(self.code_bytes)
        if data_len + code_len > self.mem_bytes:
            err = "Code and data don't fit into memory"
            raise MemoryError(err)
        self.code_start = data_len
        self.memory[self.code_start : self.code_start + code_len] = self.code_bytes
        self.code_len = code_len

        # heap pointer (word index): start allocation at first free word after code
        # code_start and code_len are bytes; convert to word index
        code_end_word = (self.code_start + self.code_len) // 4
        self.heap_ptr = int(code_end_word)
        logging.debug("Datapath: heap_ptr initialized to word %d", self.heap_ptr)

        # registers/state
        self.PC = self.code_start
        self.ACC = 0

        # reserve areas at top for interrupt-stack, frame-stack and call-stack (words)
        self.call_stack_cells = int(call_stack_cells)
        if self.call_stack_cells < 1:
            self.call_stack_cells = 1
        self.frame_stack_cells = int(frame_stack_cells)
        if self.frame_stack_cells < 0:
            self.frame_stack_cells = 0
        self.interrupt_stack_cells = int(interrupt_stack_cells)
        if self.interrupt_stack_cells < 0:
            self.interrupt_stack_cells = 0

        top_reserved = self.call_stack_cells + self.frame_stack_cells + self.interrupt_stack_cells
        if top_reserved >= self.mem_cells:
            err = "Not enough memory to reserve stacks"
            raise MemoryError(err)

        # runtime_base is the first word after reserved top area
        self.runtime_base = self.mem_cells - top_reserved

        # runtime-stack initially empty
        self.runtime_len = 0
        self.SP = self.runtime_base  # physical next-free index (stack grows downward)
        self.FP = 0  # logical frame pointer

        # frame-stack pointer (physical next-free index)
        # frame-stack area starts at runtime_base + interrupt_stack_cells and has frame_stack_cells words
        # empty FSP == runtime_base + interrupt_stack_cells + frame_stack_cells
        self.FSP = self.runtime_base + self.interrupt_stack_cells + self.frame_stack_cells

        # interrupt-stack pointer (physical next-free index)
        # interrupt-stack area is runtime_base .. runtime_base + interrupt_stack_cells - 1
        # empty ISP == runtime_base + interrupt_stack_cells
        self.ISP = self.runtime_base + self.interrupt_stack_cells

        # call-stack pointer (physical next-free index at very top)
        self.CP = int(self.mem_cells)

        # ticks / interrupts / mmio
        self.tick = 0
        self.interrupt_ready = False
        self.interruption_allowed = True
        self.interruption_vector = None
        self.in_interruption = False
        self.input_schedule = []
        self.input_closed = False
        self.mmio_in = mmio_in
        self.mmio_out = mmio_out
        self.mmio_in_buffer = []
        self.mmio_out_buffer = []
        self.mmio_in_cell = 0
        self.output_buffer = []
        self.tick_limit = tick_limit
        self.pause_tick = pause_tick
        self.last_interrupt_source = None

    # helpers to read/write 4-byte words in little-endian
    def _word_byte_offset(self, word_addr: int) -> int:
        return int(word_addr) * 4

    def read_word(self, word_addr: int) -> int:
        """Read a 4-byte little-endian signed word from memory.

        Returns 0 on out-of-range accesses.
        """
        off = self._word_byte_offset(word_addr)
        if off < 0 or off + 4 > len(self.memory):
            return 0
        return int.from_bytes(self.memory[off : off + 4], byteorder="little", signed=True)

    def write_word(self, word_addr: int, value: int) -> None:
        """Write a 4-byte little-endian signed word to memory.

        Raises MemoryError for out-of-range writes.
        """
        off = self._word_byte_offset(word_addr)
        if off < 0 or off + 4 > len(self.memory):
            err = f"write_word out of memory: word {word_addr} (byte {off})"
            raise MemoryError(err)
        v = int(value) & 0xFFFFFFFF
        if v & 0x80000000:
            v = v - (1 << 32)
        self.memory[off : off + 4] = struct.pack("<i", int(v))

    def _write_pstr_to_mem(self, base_word_addr: int, buffer_list: list[int]) -> None:
        """Write pstr (len + chars-as-words) starting at base_word_addr.

        If buffer_list is empty write header 0.
        """
        try:
            if not buffer_list:
                self.write_word(base_word_addr, 0)
                return
            ln = len(buffer_list)
            if base_word_addr + 1 + ln > self.mem_cells:
                logging.debug(
                    "mmio pstr write would overflow memory at %d (len %d) -> skipped",
                    base_word_addr,
                    ln,
                )
                return
            self.write_word(base_word_addr, ln)
            for i, ch in enumerate(buffer_list):
                self.write_word(base_word_addr + 1 + i, ch & 0xFF)
        except MemoryError as e:
            logging.debug("_write_pstr_to_mem failed: %s", e)

    def mem_read_word(self, word_addr: int) -> int:
        """Read a word, with MMIO IN behaviour for the mmio_in address."""
        if word_addr == self.mmio_in:
            if self.mmio_in_buffer:
                return int(self.mmio_in_buffer[0])
            return 0
        return self.read_word(word_addr)

    # break mmio handling into helpers to reduce mem_write_word complexity
    def _mmio_out_write(self, value: int) -> None:
        ch = value & 0xFF
        self.mmio_out_buffer.append(ch)
        self.output_buffer.append(ch)
        self.last_interrupt_source = "OUT"
        try:
            ch_repr = chr(ch) if 0 <= ch < 256 else "?"
        except Exception:
            ch_repr = "?"
        logging.debug("[MMIO OUT] wrote %d -> char: %s", value, ch_repr)
        self._write_pstr_to_mem(self.mmio_out, self.mmio_out_buffer)

    def _mmio_in_write(self, value: int) -> None:
        ch = value & 0xFF
        if getattr(self, "input_closed", False):
            logging.debug("[MMIO IN] input closed, ignoring incoming char %d", ch)
            return
        self.mmio_in_buffer.append(ch)
        self.mmio_in_cell = ch
        logging.debug("[MMIO IN] input cell set to %d", ch)
        self._write_pstr_to_mem(self.mmio_in, self.mmio_in_buffer)

    def _mmio_in_consume(self) -> None:
        popped = self.mmio_in_buffer.pop(0)
        logging.debug("[MMIO IN] consumed %d", popped)
        self._write_pstr_to_mem(self.mmio_in, self.mmio_in_buffer)
        if popped == 10:  # newline -> close input
            self.input_closed = True
            if getattr(self, "input_schedule", None):
                self.input_schedule = []
                logging.debug("[MMIO IN] newline -> in closed, schedule cleared")

    def mem_write_word(self, word_addr: int, value: int) -> None:
        """Implement special MMIO behavior.

        - mmio_out: append to output_buffer and mmio_out_buffer;
          write pstr to memory at mmio_out.
        - mmio_in: nonzero -> append new char to mmio_in_buffer;
          zero -> consume first char.
        """
        try:
            value = int(value) & 0xFFFFFFFF
        except Exception:
            value = 0

        # MMIO OUT
        if word_addr == self.mmio_out:
            self._mmio_out_write(value)
            return

        # MMIO IN
        if word_addr == self.mmio_in:
            if value != 0:
                self._mmio_in_write(value)
                return
            # consume one char
            if self.mmio_in_buffer:
                self._mmio_in_consume()
                return
            # nothing to consume -> ensure header zero
            self._write_pstr_to_mem(self.mmio_in, [])
            self.mmio_in_cell = 0
            return

        try:
            self.write_word(word_addr, value)
        except MemoryError:
            logging.debug(
                "mem_write_word: attempt to write out-of-range word %s; ignored",
                word_addr,
            )

    # --- runtime-stack helpers (values/locals/args) ---
    def _phys_addr_for_logical_index(self, logical_idx: int) -> int:
        """Map logical runtime index -> physical word address."""
        return int(self.runtime_base - 1 - logical_idx)

    def mem_push(self, value: int) -> None:
        """Push a signed 32-bit word onto the runtime stack (stack in memory)."""
        try:
            v = int(value) & 0xFFFFFFFF
        except Exception:
            v = 0
        if v & 0x80000000:
            v_signed = v - (1 << 32)
        else:
            v_signed = v
        new_len = self.runtime_len + 1
        new_sp = self.runtime_base - new_len
        # Prevent overlap with frame-stack and call-stack: new_sp must be < FSP
        if new_sp < 0 or new_sp >= self.mem_cells or not (new_sp < self.FSP):
            logging.debug("mem_push: stack overflow or collision -> new_sp=%s FSP=%s; push ignored", new_sp, self.FSP)
            return
        self.runtime_len = new_len
        self.SP = new_sp
        try:
            self.write_word(self.SP, v_signed)
        except MemoryError:
            logging.debug("mem_push: write out of range at %d", self.SP)

    def mem_pop(self) -> int:
        """Pop a word from the runtime stack. Return 0 on underflow."""
        if self.runtime_len == 0:
            logging.debug("mem_pop: stack underflow -> returning 0")
            return 0
        try:
            v = self.read_word(self.SP)
        except Exception:
            v = 0
        # shrink
        self.runtime_len -= 1
        self.SP = self.runtime_base - self.runtime_len
        return int(v)

    # --- frame-stack helpers (saved FP values) ---
    def frame_push(self, value: int) -> None:
        """Push a word onto the frame-stack (FSP decreases)."""
        try:
            v = int(value) & 0xFFFFFFFF
        except Exception:
            v = 0
        if v & 0x80000000:
            v_signed = v - (1 << 32)
        else:
            v_signed = v
        new_fsp = self.FSP - 1
        # Prevent overlap with runtime-stack: new_fsp must be > SP
        if new_fsp < 0 or new_fsp >= self.mem_cells or not (new_fsp > self.SP):
            logging.debug(
                "frame_push: frame-stack overflow or collision -> new_fsp=%s SP=%s; push ignored", new_fsp, self.SP
            )
            return
        self.FSP = new_fsp
        try:
            self.write_word(self.FSP, v_signed)
        except MemoryError:
            logging.debug("frame_push: write out of range at %d", self.FSP)

    def frame_pop(self) -> int:
        """Pop a word from the frame-stack. Return 0 on underflow."""
        # empty when FSP == runtime_base + interrupt_stack_cells + frame_stack_cells
        empty_fsp = self.runtime_base + self.interrupt_stack_cells + self.frame_stack_cells
        if self.FSP >= empty_fsp:
            logging.debug("frame_pop: frame-stack underflow -> returning 0")
            return 0
        try:
            v = self.read_word(self.FSP)
        except Exception:
            v = 0
        self.FSP += 1
        return int(v)

    # --- call-stack helpers (return-address stack) ---
    def call_push(self, value: int) -> None:
        """Push a word onto the call stack (CP decreases)."""
        try:
            v = int(value) & 0xFFFFFFFF
        except Exception:
            v = 0
        if v & 0x80000000:
            v_signed = v - (1 << 32)
        else:
            v_signed = v
        new_cp = self.CP - 1
        # Prevent overlap: new_cp must be > FSP (physical). If new_cp <= FSP -> collision
        if new_cp < 0 or new_cp >= self.mem_cells or not (new_cp > self.FSP):
            logging.debug(
                "call_push: call-stack overflow or collision -> new_cp=%s FSP=%s; push ignored", new_cp, self.FSP
            )
            return
        self.CP = new_cp
        try:
            self.write_word(self.CP, v_signed)
        except MemoryError:
            logging.debug("call_push: write out of range at %d", self.CP)

    def call_pop(self) -> int:
        """Pop a word from the call stack. Return 0 on underflow."""
        if self.CP >= self.mem_cells:
            logging.debug("call_pop: call-stack underflow -> returning 0")
            return 0
        try:
            v = self.read_word(self.CP)
        except Exception:
            v = 0
        self.CP += 1
        return int(v)

    # --- interrupt-stack helpers (ISP) ---
    def isp_push(self, value: int) -> None:
        """Push a word onto the interrupt-stack (ISP decreases)."""
        try:
            v = int(value) & 0xFFFFFFFF
        except Exception:
            v = 0
        if v & 0x80000000:
            v_signed = v - (1 << 32)
        else:
            v_signed = v
        new_isp = self.ISP - 1
        # Prevent overflow or collision with runtime-stack: new_isp must be >= runtime_base and > SP
        if new_isp < 0 or new_isp >= self.mem_cells or not (new_isp >= self.runtime_base and new_isp > self.SP):
            logging.debug(
                "isp_push: interrupt-stack overflow or collision -> new_isp=%s SP=%s; push ignored", new_isp, self.SP
            )
            return
        self.ISP = new_isp
        try:
            self.write_word(self.ISP, v_signed)
        except MemoryError:
            logging.debug("isp_push: write out of range at %d", self.ISP)

    def isp_pop(self) -> int:
        """Pop a word from the interrupt-stack. Return 0 on underflow."""
        empty_isp = self.runtime_base + self.interrupt_stack_cells
        if self.ISP >= empty_isp:
            logging.debug("isp_pop: interrupt-stack underflow -> returning 0")
            return 0
        try:
            v = self.read_word(self.ISP)
        except Exception:
            v = 0
        self.ISP += 1
        return int(v)

    def isp_empty(self) -> bool:
        """Return True if interrupt-stack is empty."""
        empty_isp = self.runtime_base + self.interrupt_stack_cells
        return self.ISP >= empty_isp

    # Frame helpers: push_frame and pop_frame operate on runtime-stack and frame-stack in memory
    def push_frame(self, total_locals: int, num_args: int) -> None:
        """Reserve and set up a new frame on the runtime stack.

        Old FP is stored on the frame-stack (in memory).
        The compiler's original convention is preserved: FP = len(runtime_stack) - num_args.
        """
        # save previous FP on frame-stack (in memory)
        self.frame_push(self.FP)
        # base = start index for locals (args are expected to be at the top of stack)
        base = self.runtime_len - num_args
        if base < 0:
            base = 0
        self.FP = base
        # ensure stack has space for all locals
        need_len = self.FP + total_locals
        while self.runtime_len < need_len:
            self.mem_push(0)

    def pop_frame(self) -> None:
        """Pop current frame and restore previous FP."""
        # truncate stack to FP (drop locals and arguments)
        if self.FP < 0:
            self.FP = 0
        while self.runtime_len > self.FP:
            self.mem_pop()
        # restore previous FP from frame-stack (in memory)
        prev = self.frame_pop()
        try:
            self.FP = int(prev)
        except Exception:
            self.FP = 0

    def schedule_input(self, schedule: list[tuple[int, str]]) -> None:
        """Attach an input schedule (list of (tick, char))."""
        self.input_schedule = list(schedule)
        logging.debug(
            "Datapath.schedule_input called, %d events attached",
            len(self.input_schedule),
        )

    def get_nearest_input_moment(self) -> int:
        """Return tick of next scheduled input or -1 when none available."""
        if len(self.input_schedule) == 0 or getattr(self, "input_closed", False):
            return -1
        return int(self.input_schedule[0][0])


class ControlUnit:
    """Control unit implementing the FETCH-DECODE-EXEC loop for the Datapath."""

    dp: Datapath

    def __init__(self, dp: Datapath) -> None:
        """Create a ControlUnit bound to `dp`."""
        self.dp = dp

    def _format_instr(self, opcode: OpCode, arg: int) -> str:
        try:
            return mnemonic(opcode, arg)
        except Exception:
            return f"{opcode.name} {arg}"

    def _mem_at_word(self, word_addr: int | None) -> int:
        try:
            if word_addr is None:
                return 0
            return self.dp.read_word(word_addr)
        except Exception:
            return 0

    def _log_step(
        self,
        state: str,
        step: str,
        tick: int,
        pc: int,
        addr_word: int | None,
        acc: int,
        instr: str,
    ) -> None:
        # skip verbose per-step logs in lenient mode to reduce log size
        if getattr(self.dp, "lenient_log", False):
            return

        mem_val = self._mem_at_word(addr_word) if addr_word is not None else 0
        left = f"STATE: {state:<10} STEP: {step:<15} TICK: {tick:4d} PC: {pc:5d} "
        right = (
            f"ADDR: {addr_word if addr_word is not None else 0:5d} "
            f"MEM[ADDR]: {mem_val:10d} ACC: {acc:10d} SP: {self.dp.SP:5d} "
            f"FSP: {self.dp.FSP:5d} CP: {self.dp.CP:5d} ISP: {self.dp.ISP:5d} "
            f"heap_ptr: {self.dp.heap_ptr:5d} C: 0 V: 0\tINSTR: {instr}"
        )
        logging.debug(left + right)

    def run(self) -> tuple[str, int, str]:  # noqa: C901
        """Execute the datapath until halt/pause or tick limit."""
        dp = self.dp
        while dp.tick < dp.tick_limit:
            # paused check (log a PAUSED step)
            if dp.pause_tick is not None and dp.tick == dp.pause_tick:
                self._log_step(
                    "PAUSED",
                    "PAUSE_CHECK",
                    dp.tick,
                    dp.PC,
                    (dp.PC - dp.code_start) // 4,
                    dp.ACC,
                    "pause",
                )
                break

            # If there are no more scheduled input events and in-buffer is empty,
            # treat input as closed (EOF). This prevents programs from blocking
            # forever on READ when nothing else will arrive.
            no_input = dp.get_nearest_input_moment() == -1 and not dp.mmio_in_buffer and not dp.input_closed
            if no_input:
                dp.input_closed = True
                logging.debug("No more scheduled input and mmio_in_buffer empty -> input closed (EOF)")

            # input arrival
            if dp.get_nearest_input_moment() == dp.tick:
                t, ch = dp.input_schedule.pop(0)
                if not isinstance(ch, str):
                    ch = str(ch)
                if ch == "":
                    ch = " "
                ch0 = ch[0]
                # write to mmio input cell
                dp.mem_write_word(dp.mmio_in, ord(ch0))
                dp.last_interrupt_source = "IN"
                logging.debug("[tick %d] input arrived: %r (source=IN)", dp.tick, ch0)
                # Handle interrupt immediately
                self.handle_interrupt()

            # handle interrupt (hardwired): keep the existing path
            interrupt_ready = dp.interrupt_ready
            interruption_allowed = dp.interruption_allowed
            in_interruption = dp.in_interruption
            interrupt_pending = interrupt_ready and interruption_allowed and not in_interruption
            if interrupt_pending:
                src = getattr(dp, "last_interrupt_source", None)
                if src is not None:
                    logging.debug(
                        "[tick %d] ** INTERRUPT (%s) -> vector %s **",
                        dp.tick,
                        src,
                        dp.interruption_vector,
                    )
                else:
                    logging.debug(
                        "[tick %d] ** INTERRUPT -> vector %s **",
                        dp.tick,
                        dp.interruption_vector,
                    )
                # Log interruption COMMAND_FETCH step before jumping into handler
                self._log_step(
                    "INTERRUPTION",
                    "COMMAND_FETCH",
                    dp.tick,
                    dp.PC,
                    (dp.PC - dp.code_start) // 4,
                    dp.ACC,
                    "interrupt",
                )
                self.handle_interrupt()

            try:
                self._log_step(
                    "RUNNING",
                    "COMMAND_FETCH",
                    dp.tick,
                    dp.PC,
                    (dp.PC - dp.code_start) // 4,
                    dp.ACC,
                    "fetch",
                )
                # decode_instr expects bytes; dp.memory is bytearray -> convert
                opcode, arg = decode_instr(bytes(dp.memory), dp.PC)
            except Exception:
                logging.debug("PC out of range / decode error -> HALT")
                break

            # operand-fetch step
            instr_str = self._format_instr(opcode, arg)
            self._log_step(
                "RUNNING",
                "OPERAND_FETCH",
                dp.tick,
                dp.PC,
                (dp.PC - dp.code_start) // 4,
                dp.ACC,
                instr_str,
            )

            if not getattr(dp, "lenient_log", False):
                logging.debug(
                    "Tick %d PC %d: %s %s ACC=%s SP=%s FSP=%s CP=%s",
                    dp.tick,
                    dp.PC,
                    opcode.name,
                    arg,
                    dp.ACC,
                    dp.SP,
                    dp.FSP,
                    dp.CP,
                )

            dp.PC += INSTR_SIZE

            # execute
            self.exec(opcode, arg)

            # execution step log (after exec)
            self._log_step(
                "RUNNING",
                "EXECUTION",
                dp.tick,
                dp.PC,
                (dp.PC - dp.code_start) // 4,
                dp.ACC,
                instr_str,
            )

            dp.tick += 1

            if opcode == OpCode.HALT:
                logging.debug("HALT encountered")
                break

        # build ascii output: prefer output_buffer; if empty fallback to mmio_out_buffer
        out_chars: list[str] = []
        source_buf = dp.output_buffer if dp.output_buffer else dp.mmio_out_buffer
        for v in source_buf:
            out_chars.append(chr(v) if 0 <= v < 256 else "?")

        try:
            if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                self._dump_memory_to_file("memory_dump_nice.txt")
        except Exception as e:
            logging.debug("Failed to write memory dump: %s", e)

        return (
            "".join(out_chars),
            dp.tick,
            ("paused" if (dp.pause_tick is not None and dp.tick == dp.pause_tick) else "stopped"),
        )

    def handle_interrupt(self) -> None:
        """Handle an interrupt by jumping to the configured vector.

        Save full return state into interrupt-stack (ISP) so IRET/RET can restore execution.
        """
        dp = self.dp
        # Check vector first
        if dp.interruption_vector is None:
            logging.debug("No interruption_vector set -> ignore interrupt")
            return

        # Save full context needed to restore execution (PC points to the instruction about to be executed)
        # We push a compact context onto ISP in this order (push CP first, PC last),
        # so pop order will be: PC, ACC, FP, runtime_len, FSP, CP
        try:
            # Prevent nested saves from being interrupted in the middle: disallow further interrupts
            dp.interruption_allowed = False

            # Push context fields onto ISP. Use ints; ISP writes directly to memory (doesn't change runtime_len)
            dp.isp_push(dp.CP)
            dp.isp_push(dp.FSP)
            dp.isp_push(dp.runtime_len)
            dp.isp_push(dp.FP)
            dp.isp_push(dp.ACC)
            dp.isp_push(dp.PC)

            dp.in_interruption = True
            dp.interrupt_ready = False
            # jump to handler (arg is absolute code_start + vector)
            dp.PC = int(dp.interruption_vector)
            logging.debug(
                "[handle_interrupt] jumping to vector %s, context saved to ISP (ISP=%s)",
                dp.interruption_vector,
                dp.ISP,
            )
        except Exception as e:
            logging.debug("handle_interrupt: failed to save context: %s", e)
            # if failed, ensure we don't leave interrupts disabled permanently
            dp.interruption_allowed = True

    def _restore_interrupt_return_state(self) -> None:
        """Restore saved context from interrupt-stack (ISP) on IRET or RET-from-interrupt."""
        dp = self.dp
        # If ISP empty -> nothing to restore
        if dp.isp_empty():
            logging.debug("_restore_interrupt_return_state: ISP empty -> nothing to restore")
            dp.in_interruption = False
            dp.interruption_allowed = True
            return
        try:
            # pop in reverse order of push: PC, ACC, FP, runtime_len, FSP, CP
            pc = dp.isp_pop()
            acc = dp.isp_pop()
            fp = dp.isp_pop()
            rlen = dp.isp_pop()
            fsp = dp.isp_pop()
            cp = dp.isp_pop()

            dp.PC = int(pc)
            dp.ACC = int(acc)
            dp.FP = int(fp)
            dp.runtime_len = int(rlen)
            dp.SP = dp.runtime_base - dp.runtime_len
            dp.FSP = int(fsp)
            dp.CP = int(cp)
            dp.in_interruption = False
            dp.interruption_allowed = True
            logging.debug(
                "[restore_interrupt_return_state] restored PC=%s ACC=%s FP=%s runtime_len=%s FSP=%s CP=%s",
                dp.PC,
                dp.ACC,
                dp.FP,
                dp.runtime_len,
                dp.FSP,
                dp.CP,
            )
        except Exception as e:
            logging.debug("_restore_interrupt_return_state: exception %s", e)
            # best-effort restore: clear interrupt flags
            dp.in_interruption = False
            dp.interruption_allowed = True

    def exec(self, opcode: OpCode, arg: int) -> None:  # noqa: C901
        """Execute a single instruction (hardwired control unit)."""
        dp = self.dp

        def to_s32(x: int) -> int:
            x = int(x) & 0xFFFFFFFF
            return x if x < 0x80000000 else x - 0x100000000

        # Hardwired realisation of CU
        if opcode == OpCode.NOP:
            return
        if opcode == OpCode.HALT:
            return
        if opcode == OpCode.LOAD_IMM:
            dp.ACC = arg
            return
        if opcode == OpCode.LOAD_CONST:
            dp.ACC = arg
            return
        if opcode == OpCode.LOAD_MEM:
            dp.ACC = dp.mem_read_word(arg)
            return
        if opcode == OpCode.STORE_MEM:
            dp.mem_write_word(arg, dp.ACC)
            return
        if opcode == OpCode.STORE_IND:
            addr = dp.mem_pop()
            dp.mem_write_word(addr, dp.ACC)
            return
        if opcode == OpCode.LOAD_IND:
            addr = dp.mem_pop()
            dp.ACC = dp.mem_read_word(addr)
            return
        if opcode == OpCode.ADD_IMM:
            dp.ACC = to_s32(dp.ACC + arg)
            return
        if opcode == OpCode.SUB_IMM:
            dp.ACC = to_s32(dp.ACC - arg)
            return
        if opcode == OpCode.MUL_IMM:
            dp.ACC = to_s32(dp.ACC * arg)
            return
        if opcode == OpCode.DIV_IMM:
            dp.ACC = int(dp.ACC / arg) if arg != 0 else 0
            dp.ACC = to_s32(dp.ACC)
            return

        if opcode == OpCode.JMP:
            dp.PC = dp.code_start + arg
            return
        if opcode == OpCode.BEQZ:
            if dp.ACC == 0:
                dp.PC = dp.code_start + arg
            return
        if opcode == OpCode.BNEZ:
            if dp.ACC != 0:
                dp.PC = dp.code_start + arg
            return
        if opcode == OpCode.PUSH_IMM:
            dp.mem_push(dp.ACC)
            return
        if opcode == OpCode.POP:
            dp.ACC = dp.mem_pop()
            return
        if opcode == OpCode.CALL:
            dp.call_push(dp.PC)
            dp.PC = dp.code_start + arg
            return
        if opcode == OpCode.RET:
            # normal return via call-stack
            if dp.CP < dp.mem_cells:
                retaddr = dp.call_pop()
                dp.PC = int(retaddr)
                # If we were in interruption and ISP contains a saved context,
                # support returning from interrupt via RET (fallback behaviour)
                if dp.in_interruption and not dp.isp_empty():
                    self._restore_interrupt_return_state()
                return
            # returning from interrupt (if in_interruption and ISP not empty)
            if dp.in_interruption and not dp.isp_empty():
                self._restore_interrupt_return_state()
                return
            dp.PC = len(dp.memory)
            return
        if opcode == OpCode.SETVEC:
            dp.interruption_vector = dp.code_start + arg
            logging.debug("SETVEC -> %s", dp.interruption_vector)
            return
        if opcode == OpCode.EI:
            dp.interruption_allowed = True
            logging.debug("EI -> interrupts enabled")
            return
        if opcode == OpCode.DI:
            dp.interruption_allowed = False
            logging.debug("DI -> interrupts disabled")
            return
        if opcode == OpCode.IRET:
            # restore saved state fully from ISP
            self._restore_interrupt_return_state()
            logging.debug("IRET -> returned from interrupt")
            return
        if opcode == OpCode.SOFTINT:
            # software interrupt: invoke handler immediately (internal)
            dp.last_interrupt_source = "SW"
            logging.debug("SOFTINT executed -> invoking handler")
            self.handle_interrupt()
            return
        if opcode == OpCode.BINOP_POP:
            v = dp.mem_pop()
            code = int(arg)
            if code == 1:  # ADD
                dp.ACC = to_s32(v + dp.ACC)
            elif code == 2:  # SUB
                dp.ACC = to_s32(v - dp.ACC)
            elif code == 3:  # MUL
                dp.ACC = to_s32(v * dp.ACC)
            elif code == 4:  # DIV
                if dp.ACC != 0:
                    dp.ACC = to_s32(int(v / dp.ACC))
                else:
                    dp.ACC = 0
            else:
                logging.debug("BINOP_POP: unknown code %s", code)
            return

        if opcode == OpCode.CMP_POP:
            v = dp.mem_pop()
            code = int(arg)

            def as_signed(x: int) -> int:
                x &= 0xFFFFFFFF
                return x if x < 0x80000000 else x - 0x100000000

            lv = as_signed(v)
            rv = as_signed(dp.ACC)
            res = 0
            if code == 1:
                res = 1 if lv == rv else 0
            elif code == 2:
                res = 1 if lv != rv else 0
            elif code == 3:
                res = 1 if lv < rv else 0
            elif code == 4:
                res = 1 if lv <= rv else 0
            elif code == 5:
                res = 1 if lv > rv else 0
            elif code == 6:
                res = 1 if lv >= rv else 0
            else:
                logging.debug("CMP_POP: unknown code %s", code)
            dp.ACC = res
            return

        if opcode == OpCode.LOAD_LOCAL:
            idx = int(arg)
            pos = dp.FP + idx
            if 0 <= pos < dp.runtime_len:
                phys = dp._phys_addr_for_logical_index(pos)
                dp.ACC = dp.read_word(phys)
            else:
                dp.ACC = 0
            return

        if opcode == OpCode.STORE_LOCAL:
            idx = int(arg)
            pos = dp.FP + idx
            if pos >= dp.runtime_len:
                while dp.runtime_len <= pos:
                    dp.mem_push(0)
            phys = dp._phys_addr_for_logical_index(pos)
            dp.mem_write_word(phys, dp.ACC)
            return

        if opcode == OpCode.ENTER:
            word = int(arg) & 0xFFFFFFFF
            total_locals = (word >> 16) & 0xFFFF
            num_args = word & 0xFFFF
            dp.push_frame(total_locals, num_args)
            return

        if opcode == OpCode.LEAVE:
            dp.pop_frame()
            return

        if opcode == OpCode.ALLOC:
            # allocate arg words if arg != 0, else take size from ACC
            size = int(arg) if int(arg) != 0 else int(dp.ACC)
            try:
                size = int(size)
            except Exception:
                size = 0
            if size <= 0:
                dp.ACC = 0
                logging.debug("ALLOC: size <= 0 -> return 0")
                return
            # check heap vs runtime_base collision
            base = int(dp.heap_ptr)
            if base + size > dp.runtime_base:
                logging.debug(
                    "ALLOC: out of memory: requested %d words, heap_ptr=%d, runtime_base=%d",
                    size,
                    base,
                    dp.runtime_base,
                )
                dp.ACC = 0
                return
            # zero memory for safety
            for i in range(size):
                try:
                    dp.write_word(base + i, 0)
                except MemoryError:
                    logging.debug("ALLOC: write failed at %d", base + i)
            dp.heap_ptr = base + size
            dp.ACC = base
            logging.debug("ALLOC: allocated %d words at base %d (new heap_ptr=%d)", size, base, dp.heap_ptr)
            return

        if opcode == OpCode.ASET:
            # If arg != 0: arg is base; pop index from stack
            # If arg == 0: pop index then pop base from stack
            try:
                if int(arg) != 0:
                    base = int(arg)
                    idx = dp.mem_pop()
                else:
                    idx = dp.mem_pop()
                    base = dp.mem_pop()
                if idx is None:
                    idx = 0
                addr = int(base) + int(idx)
                # basic bounds check: ensure writing inside memory and not in reserved stack area
                if addr < 0 or addr >= dp.mem_cells or addr >= dp.runtime_base:
                    logging.debug("ASET: out-of-bounds write attempt at word %d (base=%s idx=%s)", addr, base, idx)
                    return
                dp.mem_write_word(addr, dp.ACC)
                logging.debug("ASET: wrote ACC=%s to word %d (base=%s idx=%s)", dp.ACC, addr, base, idx)
            except Exception as e:
                logging.debug("ASET: exception %s", e)
            return

        if opcode == OpCode.AGET:
            # If arg != 0: arg is base; pop index
            # If arg == 0: pop index then base
            try:
                if int(arg) != 0:
                    base = int(arg)
                    idx = dp.mem_pop()
                else:
                    idx = dp.mem_pop()
                    base = dp.mem_pop()
                addr = int(base) + int(idx)
                if addr < 0 or addr >= dp.mem_cells or addr >= dp.runtime_base:
                    logging.debug("AGET: out-of-bounds read attempt at word %d (base=%s idx=%s)", addr, base, idx)
                    dp.ACC = 0
                    return
                dp.ACC = dp.mem_read_word(addr)
                logging.debug("AGET: loaded ACC=%s from word %d (base=%s idx=%s)", dp.ACC, addr, base, idx)
            except Exception as e:
                logging.debug("AGET: exception %s", e)
                dp.ACC = 0
            return

        if opcode == OpCode.PRINT:
            acc = dp.ACC
            try:
                ai = int(acc)
            except Exception:
                ai = None

            if ai is not None and 0 <= ai < dp.string_pool_cells:
                try:
                    ln = dp.read_word(ai)
                except Exception:
                    ln = -1
                if isinstance(ln, int) and ln > 0 and ln < 100000 and (ai + 1 + ln) * 4 <= dp.code_start:
                    ok = True
                    chars: list[int] = []
                    for i in range(1, ln + 1):
                        ch_code = dp.read_word(ai + i) & 0xFF
                        if ch_code in (9, 10, 13) or 32 <= ch_code <= 126:
                            chars.append(ch_code)
                        else:
                            ok = False
                            break
                    if ok:
                        logging.debug("PRINT: Detected pstr at %s len=%s", ai, ln)
                        for ch_code in chars:
                            dp.mem_write_word(dp.mmio_out, ch_code)
                        return
            # numeric printing
            x = dp.ACC & 0xFFFFFFFF
            if x & 0x80000000:
                x = -((~x + 1) & 0xFFFFFFFF)
            s = str(x)
            logging.debug("PRINT: numeric -> %s", s)
            for ch in s:
                dp.mem_write_word(dp.mmio_out, ord(ch))
            return

        if opcode == OpCode.READ:
            dp.ACC = dp.mem_read_word(dp.mmio_in)
            if dp.ACC == 0:
                if not getattr(dp, "input_closed", False):
                    logging.debug("READ: no data -> retrying next tick")
                    dp.PC -= INSTR_SIZE
                    return
                logging.debug("READ: no data -> returning 0 (EOF or closed)")
                return
            try:
                chrepr = chr(dp.ACC)
            except Exception:
                chrepr = repr(dp.ACC)
            logging.debug("READ: got %s ('%s')", dp.ACC, chrepr)
            dp.mem_write_word(dp.mmio_in, 0)
            return
        logging.debug("Unhandled opcode: %s", opcode)

    # split memory dump into parts to reduce complexity of one function
    def _dump_data_section(self, f: TextIO, dp: Datapath) -> None:
        data_words = dp.code_start // 4
        f.write("=== DATA (words) ===\n")
        for i in range(data_words):
            off = i * 4
            w_bytes = dp.memory[off : off + 4]
            if len(w_bytes) < 4:
                break
            unsigned = struct.unpack("<I", w_bytes)[0]
            signed = struct.unpack("<i", w_bytes)[0]
            txt = f"{i:08d}: {unsigned:08X}  ({signed})"
            ln = signed
            if 0 <= ln and (i + 1 + ln) * 4 <= dp.code_start and ln < 100000:
                chars: list[str] = []
                ok = True
                for j in range(ln):
                    offj = (i + 1 + j) * 4
                    if offj + 4 > dp.code_start:
                        ok = False
                        break
                    ch = struct.unpack("<I", dp.memory[offj : offj + 4])[0] & 0xFF
                    chars.append(chr(ch) if 32 <= ch < 127 else f"\\x{ch:02X}")
                if ok:
                    s = "".join(chars)
                    txt += f"   (pstr len={ln} '{s}')"
            f.write(txt + "\n")

    def _dump_code_section(self, f: TextIO, dp: Datapath) -> None:
        f.write("\n=== CODE (instructions) ===\n")
        pc = dp.code_start
        code_end = dp.code_start + dp.code_len
        while pc < code_end:
            try:
                opcode, arg = decode_instr(bytes(dp.memory), pc)
                instr_bytes = dp.memory[pc : pc + INSTR_SIZE]
                hexbytes = instr_bytes.hex().upper()
                try:
                    mnem = mnemonic(opcode, arg)
                except Exception:
                    mnem = f"{opcode.name} {arg}"
                f.write(f"{pc:08d}: {hexbytes:<12} - {mnem}\n")
            except Exception as e:
                rem = dp.memory[pc:code_end].hex().upper()
                f.write(f"{pc:08d}: {rem} - <incomplete or decode error: {e}>\n")
                break
            pc += INSTR_SIZE

    def _dump_memory_to_file(self, path: str) -> None:
        dp = self.dp
        with open(path, "w", encoding="utf-8") as f:
            f.write("=== MEMORY DUMP ===\n")
            f.write(f"mem_cells: {dp.mem_cells}  mem_bytes: {len(dp.memory)}\n")
            f.write(f"data_bytes_len (code_start): {dp.code_start}  code_bytes_len: {dp.code_len}\n\n")
            self._dump_data_section(f, dp)
            self._dump_code_section(f, dp)
            f.write("\n=== END DUMP ===\n")


# ---------- Public API ----------
def run_bytes(code_bytes: bytes, data_bytes: bytes, config: dict[str, Any] | None) -> tuple[str, int, str]:
    """Run VM on given bytes and config and return (stdout, ticks, state)."""
    cfg = dict(config) if config is not None else {}
    mmio_in = cfg.get("mmio_in") if "mmio_in" in cfg else 0xFFF0
    mmio_in = 0xFFF0 if mmio_in is None else mmio_in
    mmio_out = cfg.get("mmio_out") if "mmio_out" in cfg else 0xFFF1
    mmio_out = 0xFFF1 if mmio_out is None else mmio_out

    mem_cells = cfg.get("mem_cells", 65536)
    tick_limit = cfg.get("tick_limit", 100000)
    pause_tick = cfg.get("pause_tick", None)
    spc = cfg.get("string_pool_cells", None)
    if spc is None:
        spc = 1024
    call_stack_cells = cfg.get("call_stack_cells", 1024)
    frame_stack_cells = cfg.get("frame_stack_cells", 256)
    interrupt_stack_cells = cfg.get("interrupt_stack_cells", 64)
    lenient = cfg.get("lenient_log", False)

    dp = Datapath(
        code_bytes,
        data_bytes,
        mem_cells=mem_cells,
        mmio_in=mmio_in,
        mmio_out=mmio_out,
        tick_limit=tick_limit,
        pause_tick=pause_tick,
        string_pool_cells=spc,
        call_stack_cells=call_stack_cells,
        frame_stack_cells=frame_stack_cells,
        interrupt_stack_cells=interrupt_stack_cells,
        lenient_log=lenient,
    )
    cu = ControlUnit(dp)
    out, ticks, state = cu.run()

    # raw memory dump writing kept but silent
    try:
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            with open("memory_dump.txt", "w", encoding="utf-8") as f:
                for i in range(dp.mem_cells):
                    off = i * 4
                    w = struct.unpack("<I", dp.memory[off : off + 4])[0]
                    si = struct.unpack("<i", dp.memory[off : off + 4])[0]
                    f.write(f"{i:08d}: {w:08X}  ({si})\n")
    except Exception as e:
        logging.debug("Failed to write memory_dump.txt: %s", e)

    # if debug -> write out.bin / out.hex (from original code_bytes)
    try:
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            _write_debug_out_files(code_bytes)
    except Exception as e:
        logging.debug("Failed to write out.hex/out.bin: %s", e)

    return out, ticks, state


# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    from parser import Compiler, parse, tokenize

    ap = argparse.ArgumentParser(
        description="Processor VM runner. Accepts Lisp source (.lisp) or binary (.bin). "
        "Input for MMIO must be provided via --input-schedule (tick char per line)."
    )
    ap.add_argument("program", help="program.lisp or program.bin (binary code).")
    ap.add_argument(
        "--input-schedule",
        help="input schedule file (tick char per line). Each non-empty line: '<tick> <char>'",
        default=None,
    )
    ap.add_argument("--config", help="path to yaml config", default=None)

    help_debug = "enable debug logging to logfile (detailed per-step state)."
    help_logfile = "path to processor log"
    help_console = "also echo logs to console (only when --debug)"
    ap.add_argument("--debug", action="store_true", help=help_debug)
    ap.add_argument("--logfile", default=LOGFILE, help=help_logfile)
    ap.add_argument("--console", action="store_true", help=help_console)
    args = ap.parse_args()

    # initialize logging according to CLI flags
    # note: original CLI previously referenced args.trace in some environments;
    # keep behavior conservative and only use args.debug here.
    debug_enabled = args.debug
    init_logging(logfile=args.logfile, debug=debug_enabled, console=args.console)

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print("Bad config:", e)
        sys.exit(2)

    # helper to parse schedule file with lines "<tick> <char>".
    def parse_schedule_file(path: str) -> list[tuple[int, str]]:
        """Parse schedule file with lines "<tick> <char>".

        Returns list of (tick, char).
        """
        result: list[tuple[int, str]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                try:
                    tick = int(parts[0])
                except Exception as e:
                    err = f"Bad schedule line (bad tick): {line!r}"
                    raise ValueError(err) from e
                token = parts[1] if len(parts) > 1 else " "
                try:
                    token_unescaped = bytes(token, "utf-8").decode("unicode_escape")
                except Exception:
                    token_unescaped = token
                if token_unescaped == "":
                    ch = " "
                else:
                    if len(token_unescaped) > 1:
                        logging.debug(
                            "Schedule token %r decoded to %r (len>1); using first char",
                            token,
                            token_unescaped,
                        )
                    ch = token_unescaped[0]
                result.append((tick, ch))
        return result

    sched: list[tuple[int, str]] = []
    if args.input_schedule:
        sched_path = Path(args.input_schedule)
        if not sched_path.exists():
            print("Input schedule file not found:", args.input_schedule)
            sys.exit(2)
        try:
            sched = parse_schedule_file(args.input_schedule)
            logging.debug("CLI: parsed schedule from %s: %r", args.input_schedule, sched)
        except Exception:
            logging.exception("Failed to parse --input-schedule: %s")
            sys.exit(2)

    # Read program: if .lisp compile, otherwise load binary
    if args.program.endswith(".lisp"):
        src = open(args.program, encoding="utf-8").read()
        toks = tokenize(src)
        ast = parse(toks)
        comp = Compiler(ast)
        comp.compile()
        code_bytes = bytes(comp.code)
        data_bytes = bytes(comp.data)
    else:
        code_path = Path(args.program)
        if not code_path.exists():
            print("Program file not found:", args.program)
            sys.exit(2)
        code_bytes = code_path.read_bytes()
        data_path = args.program + ".data"
        data_bytes = Path(data_path).read_bytes() if Path(data_path).exists() else b""

    # Decide Datapath creation: if schedule explicitly provided,
    # create Datapath with schedule and run
    if sched:
        vm_cfg = cfg.copy()
        mmio_in = vm_cfg["mmio_in"] if vm_cfg.get("mmio_in") is not None else 0xFFF0
        mmio_out = vm_cfg["mmio_out"] if vm_cfg.get("mmio_out") is not None else 0xFFF1
        spc = vm_cfg.get("string_pool_cells", None)
        if spc is None:
            spc = 1024
        call_stack_cells = vm_cfg.get("call_stack_cells", 1024)
        frame_stack_cells = vm_cfg.get("frame_stack_cells", 256)
        interrupt_stack_cells = vm_cfg.get("interrupt_stack_cells", 64)
        lenient = vm_cfg.get("lenient_log", False)
        mem_cells_cfg = vm_cfg.get("mem_cells", 65536)
        tick_limit_cfg = vm_cfg.get("tick_limit", 100000)
        pause_tick_cfg = vm_cfg.get("pause_tick", None)
        dp = Datapath(
            code_bytes,
            data_bytes,
            mem_cells=mem_cells_cfg,
            mmio_in=mmio_in,
            mmio_out=mmio_out,
            tick_limit=tick_limit_cfg,
            pause_tick=pause_tick_cfg,
            string_pool_cells=spc,
            call_stack_cells=call_stack_cells,
            frame_stack_cells=frame_stack_cells,
            interrupt_stack_cells=interrupt_stack_cells,
            lenient_log=lenient,
        )
        dp.schedule_input(sched)
        cu = ControlUnit(dp)
        out, ticks, state = cu.run()
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            try:
                _write_debug_out_files(code_bytes)
            except Exception as e:
                logging.debug("Failed to write debug out files (CLI sched): %s", e)
    else:
        out, ticks, state = run_bytes(code_bytes, data_bytes, cfg)

    # print VM output to stdout
    sys.stdout.write(out)
    sys.stdout.write("\n")
    sys.stdout.write("TIKS: " + str(ticks))
    sys.stdout.write("\n")
