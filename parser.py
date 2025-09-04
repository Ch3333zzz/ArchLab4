"""Module: produce AST from Lisp-like source and compile it to binary.

This module contains:
- tokenize(s) -> list of tokens
- parse(tokens) -> AST
- Compiler class that compiles AST into code/data bytearrays
"""

from __future__ import annotations

# ruff: noqa: A005
import argparse
import re
import struct
from pathlib import Path
from typing import Any, Callable

from isa import INSTR_SIZE, OpCode, encode_instr

try:
    from config import DEFAULTS

    DEFAULT_STRING_POOL_CELLS = int(DEFAULTS.get("string_pool_cells", 1024))
except Exception:
    DEFAULT_STRING_POOL_CELLS = 1024


TOKEN_RE = re.compile(
    r"""
    \s*                             # skip leading whitespace
    (,@|                            # unquote-splicing
     [\(\)'`]|                      # single-char tokens: ( ) ' ` ,
     "([^"\\]|\\.)*"|               # double-quoted string (with escapes)
     ;[^\n]*|                       # comment until end-of-line
     [^\s('(\`;)] +)                  # atom/symbol (no whitespace or special chars)
    """,
    re.VERBOSE,
)
FORBIDDEN_FOR_USER = {"push", "pop", "halt"}


def tokenize(s: str) -> list[str]:
    """Compile regexp, find matches and return list of tokens (skip comments)."""
    tokens: list[str] = []
    for m in TOKEN_RE.finditer(s):
        tok = m.group(1)
        if tok is None:
            continue
        if tok.startswith(";"):
            continue
        tokens.append(tok)
    return tokens


def _decode_string_token(tok: str) -> str:
    """Decode a double-quoted token into a Python string."""
    s = tok[1:-1]
    return s.encode("utf-8").decode("unicode_escape")


_INT_RE = re.compile(r"^[-+]?\d+$")


def parse(tokens: list[str]) -> list[Any]:
    """Parse token list into an AST.

    Implemented iteratively and delegating token handling to a helper to
    keep function complexity low.
    """
    ast: list[Any] = []
    stack: list[list[Any]] = [ast]

    def _handle_token(tok: str) -> None:
        """Handle a single token and mutate `stack`/`ast` appropriately."""
        if tok == "(":
            new_list: list[Any] = []
            stack[-1].append(new_list)
            stack.append(new_list)
            return
        if tok == ")":
            if len(stack) == 1:
                err = "unexpected ')'"
                raise SyntaxError(err)
            stack.pop()
            return
        if tok.startswith('"'):
            stack[-1].append(("str", _decode_string_token(tok)))
            return
        if _INT_RE.fullmatch(tok):
            stack[-1].append(int(tok))
            return
        # symbol/atom
        stack[-1].append(tok)

    for tok in tokens:
        _handle_token(tok)

    if len(stack) != 1:
        err = "missing ')'"
        raise SyntaxError(err)
    return ast


class Compiler:
    """Compiler: transforms AST into machine code bytes and data bytes."""

    def __init__(self, ast: list[Any], string_pool_cells: int = DEFAULT_STRING_POOL_CELLS):
        """Create a Compiler that compiles an AST to binary using the ISA."""
        self.ast = ast
        self.code = bytearray()
        self.data = bytearray()

        # reserve string pool at the start of data (words)
        self.string_pool_cells = int(string_pool_cells)
        if self.string_pool_cells < 0:
            self.string_pool_cells = 0

        # pre-allocate pool bytes (zeroed)
        self.data += b"\x00" * (self.string_pool_cells * 4)

        # next free slot within pool (word index)
        self.pool_next = 0

        self.pc = 0
        # encode_instr returns bytes/bytearray; keep bytearray here
        self.debug: list[tuple[int, bytearray, str]] = []
        self.labels: dict[str, int] = {}
        self.patch: list[tuple[int, str]] = []  # (pos, label)
        self.consts: dict[str, int] = {}  # map string -> addr (word index)
        self.globals: dict[str, int] = {}  # global variables (top-level setq etc)

        # current function context while compiling its body
        self.current_function: str | None = None
        # map name -> local slot, or None when not in function
        self.current_locals: dict[str, int] | None = None
        self.current_epilogue: str | None = None

        # op mapping
        self.binop_code = {"+": 1, "-": 2, "*": 3, "/": 4}
        self.cmp_code = {"=": 1, "!=": 2, "<": 3, "<=": 4, ">": 5, ">=": 6}

        self._anon_label_counter = 0  # generator of unique labels for closures

    def _new_label(self, base: str = "L") -> str:
        """Create unique label name."""
        self._anon_label_counter += 1
        return f"__{base}_{self._anon_label_counter}"

    def emit(self, opcode: OpCode, arg: int = 0) -> int:
        """Generate bytecode for instruction and append it to the code buffer."""
        b = encode_instr(opcode, arg)
        # ensure debug stores a bytearray (encode_instr may return bytes/bytearray)
        b_arr = bytearray(b)
        addr = self.pc
        self.code += b_arr
        self.debug.append((addr, b_arr, f"{opcode.name} {arg}"))
        self.pc += INSTR_SIZE
        return addr

    def add_pstr(self, s: str) -> int:
        """Allocate pstr in the reserved pool if possible, otherwise append.

        Stores and returns word-index address of the string.
        """
        if s in self.consts:
            return self.consts[s]
        length = len(s)
        # try allocate inside pool
        if self.pool_next + 1 + length <= self.string_pool_cells:
            addr = self.pool_next
            off = addr * 4
            # header
            self.data[off : off + 4] = struct.pack("<i", length)
            off += 4
            for ch in s:
                self.data[off : off + 4] = struct.pack("<i", ord(ch) & 0xFF)
                off += 4
            self.pool_next += 1 + length
            self.consts[s] = addr
            return addr
        # fallback: append at end of data (after pool and any globals)
        addr = len(self.data) // 4
        self.data += struct.pack("<i", length)
        for ch in s:
            self.data += struct.pack("<i", ord(ch) & 0xFF)
        self.consts[s] = addr
        return addr

    # --- helper to patch/store 32-bit arg preserving opcode word ---
    def _write_arg_at(self, pos: int, arg: int) -> None:
        """Write the 32-bit instruction argument into instruction starting at byte pos.

        New layout: instruction is 8 bytes; we store 4-byte little-endian arg at pos..pos+3
        and leave pos+4..pos+7 (where opcode sits) unchanged.
        """
        if INSTR_SIZE < 8:
            err = "INSTR_SIZE must be >= 8 for 32-bit instruction arg layout"
            raise RuntimeError(err)

        arg_bytes = (int(arg) & 0xFFFFFFFF).to_bytes(4, byteorder="little", signed=False)

        endpos = pos + INSTR_SIZE
        if len(self.code) < endpos:
            # be robust: extend with zeros if needed
            self.code += b"\x00" * (endpos - len(self.code))

        # write 4-byte arg into pos .. pos+3
        self.code[pos : pos + 4] = arg_bytes

        # update debug record if exists
        for i, (a, _b, m) in enumerate(self.debug):
            if a == pos:
                mnemonic_part = m.split()[0]
                self.debug[i] = (a, self.code[a : a + INSTR_SIZE], f"{mnemonic_part} {arg}")
                break

    def _patch_labels(self) -> None:
        """Patch label placeholders in emitted code."""
        for pos, label in self.patch:
            if label not in self.labels:
                raise RuntimeError("unknown label " + label)
            addr = self.labels[label]
            self._write_arg_at(pos, addr)

    def compile(self) -> None:
        """Compile top-level AST into code+data."""
        self._compile_main()
        self._compile_defuns()
        self._patch_labels()

    # --- compile functions ---
    def _compile_main(self) -> None:
        """Compile top-level non-defun forms as main and emit HALT."""
        for node in self.ast:
            if isinstance(node, list) and node and node[0] == "defun":
                continue
            self.compile_expr(node)
        self.emit(OpCode.HALT, 0)

    def _compile_defuns(self) -> None:
        """Compile all defun forms after main."""
        for node in self.ast:
            if isinstance(node, list) and node and node[0] == "defun":
                self.compile_defun(node)

    # --- local-name collection  ---
    def _extract_setq_targets(self, body: list[Any]) -> list[str]:
        """Return list of variable names that are targets of `setq` in body."""
        targets: list[str] = []
        stack: list[Any] = list(body)
        while stack:
            node = stack.pop()
            if isinstance(node, list) and node:
                if node[0] == "setq" and len(node) >= 2 and isinstance(node[1], str):
                    v = node[1]
                    if v not in targets:
                        targets.append(v)
                for sub in node:
                    stack.append(sub)
        return targets

    def _collect_local_names(self, args: Any, body: list[Any]) -> list[str]:
        """Collect local names: args first, then any setq targets found in body."""
        locals_list: list[str] = []
        if isinstance(args, list):
            for a in args:
                if a not in locals_list:
                    locals_list.append(a)
        # delegate traversal to helper
        targets = self._extract_setq_targets(body)
        for t in targets:
            if t not in locals_list:
                locals_list.append(t)
        return locals_list

    def compile_defun(self, node: list[Any]) -> None:
        """Compile a defun node: ['defun', name, args_list, body...]."""
        if len(node) < 3:
            raise RuntimeError("malformed defun: " + repr(node))
        _, name, args, *body = node

        # forbid reserved names for functions
        if isinstance(name, str) and name.lower() in FORBIDDEN_FOR_USER:
            err = f"function name '{name}' is reserved and not allowed"
            raise RuntimeError(err)

        addr = self.pc  # function address
        self.labels[name] = addr

        args_list = args if isinstance(args, list) else []
        local_names = self._collect_local_names(args_list, body)
        total_locals = len(local_names)
        num_args = len(args_list)

        # save current context and set new
        old_function: str | None = self.current_function
        old_locals: dict[str, int] | None = self.current_locals
        old_ep: str | None = self.current_epilogue

        self.current_function = name
        # use a newly created dict and a local alias for assignment to satisfy mypy
        locals_map: dict[str, int] = {}
        self.current_locals = locals_map
        for idx, n in enumerate(local_names):
            locals_map[n] = idx

        epilogue_label = self._new_label(f"epilogue_{name}")
        self.current_epilogue = epilogue_label

        enter_arg = (total_locals << 16) | (num_args & 0xFFFF)
        self.emit(OpCode.ENTER, enter_arg)

        # compile function body
        for expr in body:
            self.compile_expr(expr)

        # natural end -> jump to epilogue
        jpos = self.emit(OpCode.JMP, 0)
        self.patch.append((jpos, epilogue_label))

        self.labels[epilogue_label] = self.pc
        self.emit(OpCode.LEAVE, 0)
        self.emit(OpCode.RET, 0)

        # restore context
        self.current_function = old_function
        self.current_locals = old_locals
        self.current_epilogue = old_ep

    # --- small helper compile fragments ---
    def _compile_setq(self, expr: list[Any]) -> None:
        _, var, val = expr

        if isinstance(var, str) and var.lower() in FORBIDDEN_FOR_USER:
            err = f"variable name '{var}' is reserved and not allowed"
            raise RuntimeError(err)

        self.compile_expr(val)
        if self.current_locals is not None and isinstance(var, str) and var in self.current_locals:
            self.emit(OpCode.STORE_LOCAL, self.current_locals[var])
            return
        if isinstance(var, str):
            if var not in self.globals:
                slot = len(self.data) // 4
                self.data += struct.pack("<i", 0)
                self.globals[var] = slot
            self.emit(OpCode.STORE_MEM, self.globals[var])
            return
        raise RuntimeError("setq: unsupported target " + repr(var))

    def _compile_pop_target(self, var: str) -> None:
        if self.current_locals is not None and var in self.current_locals:
            slot = self.current_locals[var]
            self.emit(OpCode.STORE_LOCAL, slot)
            return
        if var not in self.globals:
            slot = len(self.data) // 4
            self.data += struct.pack("<i", 0)
            self.globals[var] = slot
        self.emit(OpCode.STORE_MEM, self.globals[var])

    def _compile_call(self, expr: list[Any]) -> None:
        fname = expr[1]
        args_nodes = expr[2:]
        for argnode in args_nodes:
            self.compile_expr(argnode)
            self.emit(OpCode.PUSH_IMM, 0)
        pos = self.emit(OpCode.CALL, 0)
        self.patch.append((pos, fname))

    def _compile_if(self, expr: list[Any]) -> None:
        _, cond, thenb, *rest = expr
        self.compile_expr(cond)
        be_pos = self.emit(OpCode.BEQZ, 0)
        self.compile_expr(thenb)
        jmp_pos = self.emit(OpCode.JMP, 0)
        else_addr = self.pc
        # write else_addr into BEQZ at be_pos (preserve opcode)
        self._write_arg_at(be_pos, else_addr)
        if rest:
            self.compile_expr(rest[0])
        end_addr = self.pc
        # write end_addr into JMP at jmp_pos
        self._write_arg_at(jmp_pos, end_addr)

    def _compile_while(self, expr: list[Any]) -> None:
        _, cond, *body = expr
        loop_start = self.pc
        self.compile_expr(cond)
        be_pos = self.emit(OpCode.BEQZ, 0)
        for b in body:
            self.compile_expr(b)
        self.emit(OpCode.JMP, loop_start)
        after = self.pc
        self._write_arg_at(be_pos, after)

    def _compile_binop(self, head: str, expr: list[Any]) -> None:
        if len(expr) == 2:
            self.compile_expr(expr[1])
            return
        self.compile_expr(expr[1])
        for argnode in expr[2:]:
            self.emit(OpCode.PUSH_IMM, 0)
            self.compile_expr(argnode)
            self.emit(OpCode.BINOP_POP, self.binop_code[head])

    def _compile_cmp(self, head: str, expr: list[Any]) -> None:
        if len(expr) != 3:
            self.emit(OpCode.LOAD_IMM, 0)
            return
        self.compile_expr(expr[1])
        self.emit(OpCode.PUSH_IMM, 0)
        self.compile_expr(expr[2])
        self.emit(OpCode.CMP_POP, self.cmp_code[head])

    def _compile_push(self, expr: list[Any]) -> None:
        if len(expr) == 1:
            self.emit(OpCode.PUSH_IMM, 0)
        else:
            self.compile_expr(expr[1])
            self.emit(OpCode.PUSH_IMM, 0)

    def _compile_pop(self, expr: list[Any]) -> None:
        if len(expr) == 1:
            self.emit(OpCode.POP, 0)
            return
        self.emit(OpCode.POP, 0)
        var = expr[1]
        if isinstance(var, str):
            self._compile_pop_target(var)
            return
        raise RuntimeError("pop: unsupported target " + repr(var))

    def _compile_setvec(self, expr: list[Any]) -> None:
        fname = expr[1]
        pos = self.emit(OpCode.SETVEC, 0)
        self.patch.append((pos, fname))

    def _compile_return(self, expr: list[Any]) -> None:
        if len(expr) > 1:
            self.compile_expr(expr[1])
        if self.current_epilogue is not None:
            pos = self.emit(OpCode.JMP, 0)
            self.patch.append((pos, self.current_epilogue))
        else:
            self.emit(OpCode.RET, 0)

    def _compile_out(self, expr: list[Any]) -> None:
        if len(expr) < 2:
            self.emit(OpCode.LOAD_IMM, 0)
            return
        self.compile_expr(expr[1])
        mmio_out_addr = 0xFFF1
        self.emit(OpCode.STORE_MEM, mmio_out_addr)

    # --- new array/heap compile helpers ---
    def _compile_alloc(self, expr: list[Any]) -> None:
        # (alloc n)  where n is either int literal or expression
        if len(expr) < 2:
            # no size -> allocate 0 -> ALLOC 0 will return 0
            self.emit(OpCode.ALLOC, 0)
            return
        size_expr = expr[1]
        if isinstance(size_expr, int):
            self.emit(OpCode.ALLOC, int(size_expr))
            return
        # dynamic: compute size -> result in ACC, then ALLOC 0
        self.compile_expr(size_expr)
        self.emit(OpCode.ALLOC, 0)

    def _compile_aset(self, expr: list[Any]) -> None:
        # (aset base idx val)
        if len(expr) != 4:
            err = "aset expects exactly 3 args: base idx value"
            raise RuntimeError(err)
        base, idx, val = expr[1], expr[2], expr[3]
        # compile base and push it
        self.compile_expr(base)
        self.emit(OpCode.PUSH_IMM, 0)
        # compile idx and push it
        self.compile_expr(idx)
        self.emit(OpCode.PUSH_IMM, 0)
        # compile value into ACC
        self.compile_expr(val)
        # dynamic form: pop idx & base, use ACC as value
        self.emit(OpCode.ASET, 0)

    def _compile_aget(self, expr: list[Any]) -> None:
        # (aget base idx) -> leaves value in ACC
        if len(expr) != 3:
            err = "aget expects exactly 2 args: base idx"
            raise RuntimeError(err)
        base, idx = expr[1], expr[2]
        # push base then push idx, call AGET 0 which will pop idx, base
        self.compile_expr(base)
        self.emit(OpCode.PUSH_IMM, 0)
        self.compile_expr(idx)
        self.emit(OpCode.PUSH_IMM, 0)
        self.emit(OpCode.AGET, 0)

    # --- symbol handling ---
    def _compile_symbol(self, expr: str) -> None:
        if self.current_locals and expr in self.current_locals:
            self.emit(OpCode.LOAD_LOCAL, self.current_locals[expr])
            return
        if expr in self.globals:
            self.emit(OpCode.LOAD_MEM, self.globals[expr])
            return
        # 'nil' and unknown symbols map to 0
        self.emit(OpCode.LOAD_IMM, 0)

    # --- small top-level handlers pulled out to methods to reduce complexity ---
    def _handle_print(self, e: list[Any]) -> None:
        self.compile_expr(e[1])
        self.emit(OpCode.PRINT, 0)

    def _handle_progn(self, e: list[Any]) -> None:
        for s in e[1:]:
            self.compile_expr(s)

    def _handle_read(self, e: list[Any]) -> None:
        self.emit(OpCode.READ, 0)

    def _handle_ei(self, e: list[Any]) -> None:
        self.emit(OpCode.EI, 0)

    def _handle_di(self, e: list[Any]) -> None:
        self.emit(OpCode.DI, 0)

    def _handle_halt(self, e: list[Any]) -> None:
        self.emit(OpCode.HALT, 0)

    # --- main compile_expr (delegates to helpers to reduce complexity) ---
    def compile_expr(self, expr: Any) -> None:  # noqa: C901
        """Compile an expression (recursive)."""
        # top-level dispatch kept intentionally tiny — helpers do the work
        if expr is None:
            return

        if isinstance(expr, int):
            self.emit(OpCode.LOAD_IMM, expr)
            return

        if isinstance(expr, tuple) and expr[0] == "str":
            addr = self.add_pstr(expr[1])
            self.emit(OpCode.LOAD_CONST, addr)
            return

        if isinstance(expr, str):
            self._compile_symbol(expr)
            return

        if not (isinstance(expr, list) and len(expr) > 0):
            return

        head = expr[0]

        if isinstance(head, str):
            if head.lower() in FORBIDDEN_FOR_USER:
                err = (
                    f"use of reserved low-level form '{head}' is not allowed in user code; "
                    "these ops are emitted by the compiler automatically."
                )
                raise RuntimeError(err)

        handlers: dict[str, Callable[[list[Any]], None]] = {
            "setq": self._compile_setq,
            "print": self._handle_print,
            "read": self._handle_read,
            "push": self._compile_push,
            "pop": self._compile_pop,
            "defun": lambda e: None,  # compiled separately
            "set-interrupt-vector": self._compile_setvec,
            "ei": self._handle_ei,
            "di": self._handle_di,
            "halt": self._handle_halt,
            "HALT": self._handle_halt,
            "return": self._compile_return,
            "ret": self._compile_return,
            "RET": self._compile_return,
            "call": self._compile_call,
            "if": self._compile_if,
            "while": self._compile_while,
            "progn": self._handle_progn,
            "out": self._compile_out,
            "alloc": self._compile_alloc,
            "aset": self._compile_aset,
            "aget": self._compile_aget,
        }

        # arithmetic and comparisons handled via helpers
        if head in self.binop_code:
            self._compile_binop(head, expr)
            return

        if head in self.cmp_code:
            self._compile_cmp(head, expr)
            return

        h = handlers.get(head)
        if h:
            h(expr)  # handler performs all necessary emits
            return

        # fallback: treat as call (f a b) -> (call f a b)
        parts = ["call", head, *expr[1:]]
        self.compile_expr(parts)
        return


# --- helper entrypoints for using this module programmatically ---


def compile_source_bytes(src_bytes: bytes, string_pool_cells: int | None = None) -> tuple[bytes, bytes, Compiler]:
    """Compile source bytes (UTF-8) and return (code_bytes, data_bytes, compiler).

    The returned `compiler` object allows inspection of `debug`, `labels`, etc.
    """
    src = src_bytes.decode("utf-8")
    toks = tokenize(src)
    ast = parse(toks)
    if string_pool_cells is None:
        string_pool_cells = DEFAULT_STRING_POOL_CELLS
    comp = Compiler(ast, string_pool_cells=string_pool_cells)
    comp.compile()
    return bytes(comp.code), bytes(comp.data), comp


def compile_file(
    input_path: str | Path,
    out_bin: str | Path | None = None,
    out_data: str | Path | None = None,
    string_pool_cells: int | None = None,
    debug: bool = False,
) -> tuple[str, str]:
    """Compile a source file and write output files.

    Returns a tuple (bin_path, data_path).
    If out_bin/out_data are not provided they are derived from input_path
    ("<stem>.bin" and "<stem>.bin.data").
    """
    p = Path(input_path)
    if not p.exists():
        err = f"Source file not found: {input_path}"
        raise FileNotFoundError(err)

    src_bytes = p.read_bytes()
    code_bytes, data_bytes, comp = compile_source_bytes(src_bytes, string_pool_cells=string_pool_cells)

    if out_bin is None:
        out_bin_path = p.with_suffix(".bin")
    else:
        out_bin_path = Path(out_bin)
    if out_data is None:
        out_data_path = Path(str(out_bin_path) + ".data")
    else:
        out_data_path = Path(out_data)

    out_bin_path.write_bytes(code_bytes)
    out_data_path.write_bytes(data_bytes)

    if debug:
        # also write debug hex listing similar to processor._write_out_hex
        try:
            hex_lines: list[str] = []
            pc = 0
            while pc + INSTR_SIZE <= len(code_bytes):
                opcode_word = code_bytes[pc : pc + INSTR_SIZE]
                hexbytes = opcode_word.hex().upper()
                # try to decode (best-effort)
                from isa import decode_instr, mnemonic

                try:
                    op, arg = decode_instr(code_bytes, pc)
                    mnem = mnemonic(op, arg)
                except Exception:
                    mnem = "<decode error>"
                hex_lines.append(f"{pc} - {hexbytes} - {mnem}")
                pc += INSTR_SIZE
            Path(str(out_bin_path) + ".hex").write_text("\n".join(hex_lines), encoding="utf-8")
        except Exception:
            pass

    return str(out_bin_path), str(out_data_path)


# --- CLI ---
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compile Lisp-like source to VM binary")
    ap.add_argument("input", help="source file (e.g. program.lisp)")
    ap.add_argument("-o", "--out", help="output binary file (default: <input>.bin)")
    ap.add_argument("--data", help="output data file (default: <out>.bin.data)")
    ap.add_argument(
        "--string-pool-cells",
        type=int,
        help=f"number of words reserved for string pool (default: {DEFAULT_STRING_POOL_CELLS})",
    )
    ap.add_argument("--debug", action="store_true", help="write additional debug hex file (<out>.hex)")
    args = ap.parse_args()

    out_bin, out_data = compile_file(
        args.input,
        out_bin=args.out,
        out_data=args.data,
        string_pool_cells=args.string_pool_cells,
        debug=args.debug,
    )
    print(out_bin)
