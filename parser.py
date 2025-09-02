"""Module: produce AST from Lisp-like source and compile it to binary.

This module contains:
- tokenize(s) -> list of tokens
- parse(tokens) -> AST
- Compiler class that compiles AST into code/data bytearrays
"""

from __future__ import annotations

# ruff: noqa: A005
import re
import struct
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
     [^\s('"`;)]+)                  # atom/symbol (no whitespace or special chars)
    """,
    re.VERBOSE,
)
FORBIDDEN_FOR_USER = {"push", "pop", "ei", "di", "halt"}

def tokenize(s: str) -> list[str]:
    """Compile regexp, find matches and return list of tokens (skip comments)."""
    tokens: list[str] = []
    for m in TOKEN_RE.finditer(s):
        tok = m.group(1)
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
        self.code += b_arr
        addr = self.pc
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

    def _patch_labels(self) -> None:
        """Patch label placeholders in emitted code."""
        for pos, label in self.patch:
            if label not in self.labels:
                raise RuntimeError("unknown label " + label)
            addr = self.labels[label]
            self.code[pos + 1 : pos + 5] = struct.pack("<i", addr)
            for i, (a, _b, m) in enumerate(self.debug):
                if a == pos:
                    self.debug[i] = (a, self.code[a : a + INSTR_SIZE], f"{m.split()[0]} {addr}")
                    break

    def compile(self) -> None:
        """Compile top-level AST into code+data."""
        self._compile_main()
        self._compile_defuns()
        self._patch_labels()

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
        self.code[be_pos + 1 : be_pos + 5] = struct.pack("<i", else_addr)
        if rest:
            self.compile_expr(rest[0])
        end_addr = self.pc
        self.code[jmp_pos + 1 : jmp_pos + 5] = struct.pack("<i", end_addr)

    def _compile_while(self, expr: list[Any]) -> None:
        _, cond, *body = expr
        loop_start = self.pc
        self.compile_expr(cond)
        be_pos = self.emit(OpCode.BEQZ, 0)
        for b in body:
            self.compile_expr(b)
        self.emit(OpCode.JMP, loop_start)
        after = self.pc
        self.code[be_pos + 1 : be_pos + 5] = struct.pack("<i", after)

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
    def compile_expr(self, expr: Any) -> None:
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
                    err = f"these ops are emitted by the compiler automatically."
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
