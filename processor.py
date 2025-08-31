import struct
import logging
import os
import sys

from isa import decode_instr, INSTR_SIZE, OpCode, mnemonic
from config import load_config, ConfigError

LOGFILE = "processor.log"


def init_logging(logfile: str = LOGFILE, debug: bool = False, console: bool = False):
    """
    Configure root logger to write to `logfile`. If debug=True set DEBUG level.
    If console=True also echo logs to stdout.

    IMPORTANT: by default (debug=False) logging is effectively disabled
    (root level = CRITICAL) so normal runs produce no logging output.
    """
    root = logging.getLogger()
    # remove existing handlers
    for h in list(root.handlers):
        root.removeHandler(h)

    if debug:
        lvl = logging.DEBUG
    else:
        # suppress ordinary logging by setting very high threshold
        lvl = logging.CRITICAL

    root.setLevel(lvl)

    if debug:
        # file handler (write debug logs only when debug=True)
        fh = logging.FileHandler(logfile, mode="w", encoding="utf-8")
        fh.setLevel(lvl)
        fh.setFormatter(logging.Formatter("%(levelname)-5s %(name)s:%(filename)s:%(lineno)d %(message)s"))
        root.addHandler(fh)

        if console:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(lvl)
            ch.setFormatter(logging.Formatter("%(levelname)5s %(message)s"))
            root.addHandler(ch)

    return


def _write_debug_out_files(code_bytes: bytes):
    """
    When debug logging is on, write out.bin (raw machine code) and out.hex (disassembly).
    Flush logging handlers first to ensure processor.log is complete.
    """
    # flush logging handlers
    try:
        for h in logging.getLogger().handlers:
            try:
                if hasattr(h, "flush"):
                    h.flush()
            except Exception:
                pass
    except Exception:
        pass

    try:
        # write binary
        with open("out.bin", "wb") as f:
            f.write(code_bytes)
    except Exception as e:
        logging.debug(f"Failed to write out.bin: {e}")

    # produce code_hex (disassembly) from raw code_bytes (same format as tests)
    try:
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
        logging.debug(f"Failed to write out.hex: {e}")


class Datapath:
    def __init__(
        self,
        code_bytes: bytes,
        data_bytes: bytes,
        mem_cells=65536,
        mmio_in=0xFFF0,
        mmio_out=0xFFF1,
        tick_limit=100000,
        pause_tick=None,
        string_pool_cells=1024,
        lenient_log=False,
    ):
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
            raise MemoryError("Initial data doesn't fit into memory")
        self.memory[0:data_len] = self.data_bytes

        # string pool size (in words)
        self.string_pool_cells = int(string_pool_cells)

        # lenient logging flag
        self.lenient_log = bool(lenient_log)

        # warn if data uses more words than pool
        data_words = data_len // 4
        if data_words > self.string_pool_cells:
            logging.debug(
                f"Warning: data section uses {data_words} words but string_pool_cells={self.string_pool_cells}. "
                "Some static strings may live outside the pool and won't be treated as pstr by PRINT."
            )

        # write code bytes right after data
        code_len = len(self.code_bytes)
        if data_len + code_len > self.mem_bytes:
            raise MemoryError("Code and data don't fit into memory")
        self.code_start = data_len  # byte offset where code begins
        self.memory[self.code_start : self.code_start + code_len] = self.code_bytes
        self.code_len = code_len

        # initialize registers/state
        self.PC = self.code_start
        self.ACC = 0
        self.runtime_stack = []
        self.call_stack = []
        self.tick = 0
        self.interrupt_ready = False
        self.interruption_allowed = True
        self.interruption_vector = None
        self.in_interruption = False
        self.interrupt_return_state = None
        self.input_schedule = []
        self.input_closed = False  # when newline read or no more scheduled input -> stop consuming schedule
        self.mmio_in = mmio_in
        self.mmio_out = mmio_out
        # MMIO buffers for pstr semantics
        self.mmio_in_buffer = []   # ints (queued input chars)
        self.mmio_out_buffer = []  # ints (what was written to mmio_out pstr)
        self.mmio_in_cell = 0
        self.output_buffer = []    # ints - canonical "printed" bytes
        self.tick_limit = tick_limit
        self.pause_tick = pause_tick
        self.last_interrupt_source = None  # 'IN' or 'OUT'

        # frame pointer and frame stack
        self.FP = 0
        self.frame_stack = []

    # helpers to read/write 4-byte words in little-endian
    def _word_byte_offset(self, word_addr):
        return int(word_addr) * 4

    def read_word(self, word_addr):
        off = self._word_byte_offset(word_addr)
        if off < 0 or off + 4 > len(self.memory):
            return 0
        return struct.unpack("<i", self.memory[off : off + 4])[0]

    def write_word(self, word_addr, value):
        off = self._word_byte_offset(word_addr)
        if off < 0 or off + 4 > len(self.memory):
            raise MemoryError(f"write_word out of memory: word {word_addr} (byte {off})")
        v = int(value) & 0xFFFFFFFF
        if v & 0x80000000:
            v = v - (1 << 32)
        self.memory[off : off + 4] = struct.pack("<i", int(v))

    def _write_pstr_to_mem(self, base_word_addr, buffer_list):
        """
        Write pstr (len + chars-as-words) starting at base_word_addr.
        If buffer_list is empty write header 0.
        """
        try:
            if not buffer_list:
                self.write_word(base_word_addr, 0)
                return
            L = len(buffer_list)
            if base_word_addr + 1 + L > self.mem_cells:
                logging.debug(f"mmio pstr write would overflow memory at {base_word_addr} (len {L}) -> skipped")
                return
            self.write_word(base_word_addr, L)
            for i, ch in enumerate(buffer_list):
                self.write_word(base_word_addr + 1 + i, ch & 0xFF)
        except MemoryError as e:
            logging.debug(f"_write_pstr_to_mem failed: {e}")

    def mem_read_word(self, word_addr):
        # MMIO IN returns next available char or 0
        if word_addr == self.mmio_in:
            if self.mmio_in_buffer:
                return int(self.mmio_in_buffer[0])
            return 0
        return self.read_word(word_addr)

    def mem_write_word(self, word_addr, value):
        """
        Special MMIO behavior:
         - mmio_out: append to output_buffer and mmio_out_buffer; write pstr to memory at mmio_out.
         - mmio_in: nonzero -> append new char to mmio_in_buffer; zero -> consume first char.
        """
        try:
            value = int(value) & 0xFFFFFFFF
        except Exception:
            value = 0

        # MMIO OUT
        if word_addr == self.mmio_out:
            ch = value & 0xFF
            self.mmio_out_buffer.append(ch)
            self.output_buffer.append(ch)
            self.last_interrupt_source = 'OUT'
            logging.debug(f"[MMIO OUT] wrote {value} -> char: {chr(ch) if 0<=ch<256 else '?'}")
            # update pstr in memory at mmio_out (best-effort)
            self._write_pstr_to_mem(self.mmio_out, self.mmio_out_buffer)
            return

        # MMIO IN
        if word_addr == self.mmio_in:
            if value != 0:
                ch = value & 0xFF
                if getattr(self, "input_closed", False):
                    logging.debug(f"[MMIO IN] input closed, ignoring incoming char {ch}")
                    return
                self.mmio_in_buffer.append(ch)
                self.mmio_in_cell = ch
                logging.debug(f"[MMIO IN] input cell set to {ch}")
                self._write_pstr_to_mem(self.mmio_in, self.mmio_in_buffer)
                return
            else:
                # consume one char
                if self.mmio_in_buffer:
                    popped = self.mmio_in_buffer.pop(0)
                    logging.debug(f"[MMIO IN] consumed {popped}")
                    self._write_pstr_to_mem(self.mmio_in, self.mmio_in_buffer)
                    if popped == 10:  # newline -> close input
                        self.input_closed = True
                        if getattr(self, "input_schedule", None):
                            self.input_schedule = []
                            logging.debug("[MMIO IN] newline consumed -> input closed, schedule cleared")
                    return
                else:
                    # nothing to consume -> ensure header zero
                    self._write_pstr_to_mem(self.mmio_in, [])
                    self.mmio_in_cell = 0
                    return

        try:
            self.write_word(word_addr, value)
        except MemoryError:
            logging.debug(f"mem_write_word: attempt to write out-of-range word {word_addr}; ignored")

    # Frame helpers: push_frame and pop_frame operate on stack
    def push_frame(self, total_locals, num_args):
        """
        total_locals: total slots to reserve (including arg slots)
        num_args: number of arguments the caller pushed (must be <= total_locals)
        """

        self.frame_stack.append(self.FP)
        # base = start index for locals (args are expected to be at the top of stack)
        base = len(self.runtime_stack) - num_args
        if base < 0:
            base = 0
        self.FP = base
        # ensure stack has space for all locals
        need_len = self.FP + total_locals
        if need_len > len(self.runtime_stack):
            self.runtime_stack.extend([0] * (need_len - len(self.runtime_stack)))

    def pop_frame(self):
        # truncate stack to FP (drop locals and arguments)
        if self.FP < 0:
            self.FP = 0
        self.runtime_stack = self.runtime_stack[: self.FP]
        # restore previous FP
        self.FP = self.frame_stack.pop() if self.frame_stack else 0

    def schedule_input(self, schedule):
        self.input_schedule = list(schedule)
        logging.debug(f"Datapath.schedule_input called, {len(self.input_schedule)} events attached")

    def get_nearest_input_moment(self):
        if len(self.input_schedule) == 0 or getattr(self, "input_closed", False):
            return -1
        return int(self.input_schedule[0][0])


class ControlUnit:
    def __init__(self, dp: Datapath):
        self.dp = dp

    def _format_instr(self, opcode, arg):
        try:
            return mnemonic(opcode, arg)
        except Exception:
            return f"{opcode.name} {arg}"

    def _mem_at_word(self, word_addr):
        try:
            return self.dp.read_word(word_addr)
        except Exception:
            return 0

    def _log_step(self, state, step, tick, pc, addr_word, acc, instr):
        # skip verbose per-step logs in lenient mode to reduce log size
        if getattr(self.dp, "lenient_log", False):
            return

        mem_val = self._mem_at_word(addr_word) if addr_word is not None else 0
        logging.debug(
            f"STATE: {state:<10} STEP: {step:<15} TICK: {tick:4d} PC: {pc:5d} ADDR: {addr_word if addr_word is not None else 0:5d} "
            f"MEM[ADDR]: {mem_val:10d} ACC: {acc:10d} C: 0 V: 0\tINSTR: {instr}"
        )

    def run(self):
        dp = self.dp
        while dp.tick < dp.tick_limit:
            # paused check (log a PAUSED step)
            if dp.pause_tick is not None and dp.tick == dp.pause_tick:
                self._log_step("PAUSED", "PAUSE_CHECK", dp.tick, dp.PC, (dp.PC - dp.code_start)//4, dp.ACC, "pause")
                break

            # If there are no more scheduled input events and in-buffer is empty,
            # treat input as closed (EOF). This prevents programs from blocking
            # forever on READ when nothing else will arrive.
            if dp.get_nearest_input_moment() == -1 and not dp.mmio_in_buffer and not dp.input_closed:
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
                dp.mem_write_word(dp.mmio_in, ord(ch0))
                dp.interrupt_ready = True
                dp.last_interrupt_source = 'IN'
                logging.debug(f"[tick {dp.tick}] input arrived: {ch0!r} (source=IN)")

            # handle interrupt (hardwired)
            if dp.interrupt_ready and dp.interruption_allowed and not dp.in_interruption:
                src = getattr(dp, "last_interrupt_source", None)
                if src is not None:
                    logging.debug(f"[tick {dp.tick}] ** INTERRUPT ({src}) -> vector {dp.interruption_vector} **")
                else:
                    logging.debug(f"[tick {dp.tick}] ** INTERRUPT -> vector {dp.interruption_vector} **")
                # Log interruption COMMAND_FETCH step before jumping into handler
                self._log_step("INTERRUPTION", "COMMAND_FETCH", dp.tick, dp.PC, (dp.PC - dp.code_start)//4, dp.ACC, "interrupt")
                self.handle_interrupt()

            # fetch (COMMAND_FETCH)
            try:
                self._log_step("RUNNING", "COMMAND_FETCH", dp.tick, dp.PC, (dp.PC - dp.code_start)//4, dp.ACC, "fetch")
                opcode, arg = decode_instr(dp.memory, dp.PC)
            except Exception:
                logging.debug("PC out of range / decode error -> HALT")
                break

            # operand-fetch step
            instr_str = self._format_instr(opcode, arg)
            self._log_step("RUNNING", "OPERAND_FETCH", dp.tick, dp.PC, (dp.PC - dp.code_start)//4, dp.ACC, instr_str)

            if not getattr(dp, "lenient_log", False):
                logging.debug(f"Tick {dp.tick} PC {dp.PC}: {opcode.name} {arg} ACC={dp.ACC}")

            dp.PC += INSTR_SIZE

            # execute
            self.exec(opcode, arg)

            # execution step log (after exec)
            self._log_step("RUNNING", "EXECUTION", dp.tick, dp.PC, (dp.PC - dp.code_start)//4, dp.ACC, instr_str)

            dp.tick += 1

            if opcode == OpCode.HALT:
                logging.debug("HALT encountered")
                break

        # build ascii output: prefer output_buffer; if empty fall back to mmio_out_buffer
        out_chars = []
        source_buf = dp.output_buffer if dp.output_buffer else dp.mmio_out_buffer
        for v in source_buf:
            out_chars.append(chr(v) if 0 <= v < 256 else "?")

        try:
            if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
                self._dump_memory_to_file("memory_dump_nice.txt")
        except Exception as e:
            logging.debug(f"Failed to write memory dump: {e}")

        return (
            "".join(out_chars),
            dp.tick,
            ("paused" if (dp.pause_tick is not None and dp.tick == dp.pause_tick) else "stopped"),
        )

    def handle_interrupt(self):
        dp = self.dp
        dp.interrupt_return_state = (dp.PC, dp.ACC)
        dp.in_interruption = True
        dp.interrupt_ready = False
        dp.interruption_allowed = False
        if dp.interruption_vector is None:
            logging.debug("No interruption_vector set -> ignore interrupt")
            return
        # interruption_vector is stored as absolute PC already
        dp.PC = int(dp.interruption_vector)

    def exec(self, opcode, arg):
        dp = self.dp

        def to_s32(x):
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
            addr = dp.runtime_stack.pop() if dp.runtime_stack else 0
            dp.mem_write_word(addr, dp.ACC)
            return
        if opcode == OpCode.LOAD_IND:
            addr = dp.runtime_stack.pop() if dp.runtime_stack else 0
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
            dp.runtime_stack.append(dp.ACC)
            return
        if opcode == OpCode.POP:
            dp.ACC = dp.runtime_stack.pop() if dp.runtime_stack else 0
            return
        if opcode == OpCode.CALL:
            dp.call_stack.append(dp.PC)
            dp.PC = dp.code_start + arg
            return
        if opcode == OpCode.RET:
            if dp.call_stack:
                dp.PC = dp.call_stack.pop()
                return
            else:
                # returning from interrupt (if in interruption)
                if dp.in_interruption and dp.interrupt_return_state:
                    dp.PC, dp.ACC = dp.interrupt_return_state
                    dp.in_interruption = False
                    dp.interruption_allowed = True
                    dp.interrupt_return_state = None
                    return
                dp.PC = len(dp.memory)
                return
        if opcode == OpCode.SETVEC:
            # store absolute PC for the vector (arg is code-relative offset)
            dp.interruption_vector = dp.code_start + arg
            logging.debug(f"SETVEC -> {dp.interruption_vector}")
            return
        if opcode == OpCode.EI:
            dp.interruption_allowed = True
            logging.debug(f"EI -> interrupts enabled")
            return
        if opcode == OpCode.DI:
            dp.interruption_allowed = False
            logging.debug(f"DI -> interrupts disabled")
            return
        if opcode == OpCode.IRET:
            if dp.interrupt_return_state:
                dp.PC, dp.ACC = dp.interrupt_return_state
            dp.in_interruption = False
            dp.interruption_allowed = True
            dp.interrupt_return_state = None
            logging.debug("IRET -> returned from interrupt")
            return
        if opcode == OpCode.BINOP_POP:
            v = dp.runtime_stack.pop() if dp.runtime_stack else 0
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
                logging.debug(f"BINOP_POP: unknown code {code}")
            return

        if opcode == OpCode.CMP_POP:
            v = dp.runtime_stack.pop() if dp.runtime_stack else 0
            code = int(arg)

            def as_signed(x):
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
                logging.debug(f"CMP_POP: unknown code {code}")
            dp.ACC = res
            return

        if opcode == OpCode.LOAD_LOCAL:
            idx = int(arg)
            pos = dp.FP + idx
            if 0 <= pos < len(dp.runtime_stack):
                dp.ACC = dp.runtime_stack[pos]
            else:
                dp.ACC = 0
            return

        if opcode == OpCode.STORE_LOCAL:
            idx = int(arg)
            pos = dp.FP + idx
            if pos >= len(dp.runtime_stack):
                dp.runtime_stack.extend([0] * (pos + 1 - len(dp.runtime_stack)))
            dp.runtime_stack[pos] = int(dp.ACC) & 0xFFFFFFFF
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

        if opcode == OpCode.PRINT:
            acc = dp.ACC
            try:
                ai = int(acc)
            except Exception:
                ai = None

            if ai is not None and 0 <= ai < dp.string_pool_cells:
                try:
                    L = dp.read_word(ai)
                except Exception:
                    L = -1
                if (
                    isinstance(L, int)
                    and L > 0
                    and L < 100000
                    and (ai + 1 + L) * 4 <= dp.code_start
                ):
                    ok = True
                    chars = []
                    for i in range(1, L + 1):
                        ch_code = dp.read_word(ai + i) & 0xFF
                        if ch_code in (9, 10, 13) or 32 <= ch_code <= 126:
                            chars.append(ch_code)
                        else:
                            ok = False
                            break
                    if ok:
                        logging.debug(f"PRINT: Detected pstr at {ai} len={L}")
                        for ch_code in chars:
                            dp.mem_write_word(dp.mmio_out, ch_code)
                        return
            # numeric printing
            x = dp.ACC & 0xFFFFFFFF
            if x & 0x80000000:
                x = -((~x + 1) & 0xFFFFFFFF)
            s = str(x)
            logging.debug(f"PRINT: numeric -> {s}")
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
                else:
                    logging.debug("READ: no data -> returning 0 (EOF or closed)")
                    return
            else:
                try:
                    chrepr = chr(dp.ACC)
                except Exception:
                    chrepr = repr(dp.ACC)
                logging.debug(f"READ: got {dp.ACC} ('{chrepr}')")
                dp.mem_write_word(dp.mmio_in, 0)
                return
        logging.debug(f"Unhandled opcode: {opcode}")

    def _dump_memory_to_file(self, path):
        dp = self.dp
        with open(path, "w", encoding="utf-8") as f:
            f.write("=== MEMORY DUMP ===\n")
            f.write(f"mem_cells: {dp.mem_cells}  mem_bytes: {len(dp.memory)}\n")
            f.write(f"data_bytes_len (code_start): {dp.code_start}  code_bytes_len: {dp.code_len}\n\n")

            # DATA section
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
                L = signed
                if 0 <= L and (i + 1 + L) * 4 <= dp.code_start and L < 100000:
                    chars = []
                    ok = True
                    for j in range(L):
                        offj = (i + 1 + j) * 4
                        if offj + 4 > dp.code_start:
                            ok = False
                            break
                        ch = struct.unpack("<I", dp.memory[offj : offj + 4])[0] & 0xFF
                        chars.append(chr(ch) if 32 <= ch < 127 else f"\\x{ch:02X}")
                    if ok:
                        s = "".join(chars)
                        txt += f"   (pstr len={L} '{s}')"
                f.write(txt + "\n")

            f.write("\n=== CODE (instructions) ===\n")
            pc = dp.code_start
            code_end = dp.code_start + dp.code_len
            while pc < code_end:
                try:
                    opcode, arg = decode_instr(dp.memory, pc)
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

            f.write("\n=== END DUMP ===\n")


# ---------- Public API ----------
def run_bytes(code_bytes, data_bytes, config):
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
        logging.debug(f"Failed to write memory_dump.txt: {e}")

    # if debug -> write out.bin / out.hex (from original code_bytes)
    try:
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            _write_debug_out_files(code_bytes)
    except Exception as e:
        logging.debug(f"Failed to write debug out files: {e}")

    return out, ticks, state


# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    from parser import tokenize, parse, Compiler

    ap = argparse.ArgumentParser()
    ap.add_argument("program", help="program.lisp or program.bin")
    ap.add_argument("--config", help="path to yaml config", default=None)
    ap.add_argument("--input-schedule", help="input schedule file (tick char per line)", default=None)
    ap.add_argument("--debug", action="store_true", help="enable debug logging to logfile")
    ap.add_argument("--logfile", default=LOGFILE, help="path to processor log")
    ap.add_argument("--console", action="store_true", help="also echo logs to console")
    args = ap.parse_args()

    # initialize logging according to CLI flags
    init_logging(logfile=args.logfile, debug=args.debug, console=args.console)

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print("Bad config:", e)
        sys.exit(2)

    # read input schedule if exists
    def parse_schedule_file(path):
        result = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                try:
                    tick = int(parts[0])
                except Exception:
                    raise ValueError(f"Bad schedule line (bad tick): {line!r}")
                token = parts[1] if len(parts) > 1 else " "
                try:
                    token_unescaped = bytes(token, "utf-8").decode("unicode_escape")
                except Exception:
                    token_unescaped = token
                if token_unescaped == "":
                    ch = " "
                else:
                    if len(token_unescaped) > 1:
                        logging.debug(f"Schedule token {token!r} decoded to {token_unescaped!r} (len>1); using first char")
                    ch = token_unescaped[0]
                result.append((tick, ch))
        return result

    sched = []
    if args.input_schedule and os.path.exists(args.input_schedule):
        sched = parse_schedule_file(args.input_schedule)
        logging.debug(f"CLI: parsed schedule from {args.input_schedule}: {sched!r}")

    if args.program.endswith(".lisp"):
        src = open(args.program, encoding="utf-8").read()
        toks = tokenize(src)
        ast = parse(toks)
        comp = Compiler(ast)
        comp.compile()
        code_bytes = bytes(comp.code)
        data_bytes = bytes(comp.data)
    else:
        code_bytes = open(args.program, "rb").read()
        data_path = args.program + ".data"
        data_bytes = open(data_path, "rb").read() if os.path.exists(data_path) else b""

    # Decide Datapath creation: if schedule explicitly provided, create Datapath with schedule and run
    if sched:
        vm_cfg = cfg.copy()
        mmio_in = vm_cfg["mmio_in"] if vm_cfg.get("mmio_in") is not None else 0xFFF0
        mmio_out = vm_cfg["mmio_out"] if vm_cfg.get("mmio_out") is not None else 0xFFF1
        spc = vm_cfg.get("string_pool_cells", None)
        if spc is None:
            spc = 1024
        lenient = vm_cfg.get("lenient_log", False)
        dp = Datapath(
            code_bytes,
            data_bytes,
            mem_cells=vm_cfg["mem_cells"],
            mmio_in=mmio_in,
            mmio_out=mmio_out,
            tick_limit=vm_cfg["tick_limit"],
            pause_tick=vm_cfg["pause_tick"],
            string_pool_cells=spc,
            lenient_log=lenient,
        )
        dp.schedule_input(sched)
        cu = ControlUnit(dp)
        out, ticks, state = cu.run()
        # debug outputs for CLI-sched case
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            try:
                _write_debug_out_files(code_bytes)
            except Exception as e:
                logging.debug(f"Failed to write debug out files (CLI sched): {e}")
    else:
        out, ticks, state = run_bytes(code_bytes, data_bytes, cfg)

    sys.stdout.write(out)
    sys.stdout.write("\n")
    sys.stdout.write("TIKS: " + str(ticks))
    sys.stdout.write("\n")
