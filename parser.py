import re
import struct
from isa import OpCode, encode_instr, INSTR_SIZE

# берем значение размера пула строк из конфигурации DEFAULTS, если доступно
try:
    from config import DEFAULTS
    DEFAULT_STRING_POOL_CELLS = int(DEFAULTS.get("string_pool_cells", 1024))
except Exception:
    DEFAULT_STRING_POOL_CELLS = 1024


TOKEN_RE = re.compile(r'''\s*(,@|[(')`]|"([^"\\]|\\.)*"|;[^\n]*|[^\s('"`;)]+)''') # RegExp to make tokens

def tokenize(s):
    """
    Compiling RegExp, finding all mathes and spliting by space-symbol
    and returning the list of tokens
    """
    tokens = []
    for m in TOKEN_RE.finditer(s):
        tok = m.group(1)
        if tok.startswith(';'):
            continue
        tokens.append(tok)
    return tokens

def parse(tokens):
    """
    This method is creating AST
    """
    def parse_expr(i):
        if i >= len(tokens):
            raise SyntaxError("unexpected EOF")
        tok = tokens[i]
        if tok == '(': #is it a s-exp?
            lst = []
            i += 1
            while i < len(tokens) and tokens[i] != ')':
                expr, i = parse_expr(i)
                lst.append(expr)
            if i >= len(tokens):
                raise SyntaxError("missing ')'")
            return lst, i+1
        elif tok.startswith('"'): #is it a string?
            s = tok[1:-1]
            s = s.encode('utf-8').decode('unicode_escape')
            return ('str', s), i+1
        else: 
            if re.fullmatch(r'[-+]?\d+', tok): #is it integer?
                return int(tok), i+1
            return tok, i+1 #if all cases failed, its Symbol
    ast = []
    i = 0
    while i < len(tokens):
        expr, i = parse_expr(i)
        ast.append(expr)
    return ast

class Compiler:
    def __init__(self, ast, string_pool_cells: int = DEFAULT_STRING_POOL_CELLS):
        self.ast = ast
        self.code = bytearray()
        self.data = bytearray()

        # reserve string pool at the start of data (words)
        self.string_pool_cells = int(string_pool_cells)
        if self.string_pool_cells < 0:
            self.string_pool_cells = 0
        self.data += b'\x00' * (self.string_pool_cells * 4) # pre-allocate pool bytes (zeroed)
        self.pool_next = 0 # next free slot within pool (word index)

        self.pc = 0
        self.debug = []
        self.labels = {}
        self.patch = []  # (pos, label)
        self.consts = {}  # map string -> addr (word index)
        self.globals = {}  # global variables (top-level setq etc)

        # current function context while compiling its body
        self.current_function = None
        self.current_locals = None
        self.current_epilogue = None

        # op mapping
        self.binop_code = { '+':1, '-':2, '*':3, '/':4 }
        self.cmp_code = { '=':1, '!=':2, '<':3, '<=':4, '>':5, '>=':6 }

        self._anon_label_counter = 0 # generator of unique lables for clouses

    def _new_label(self, base="L"): # creates unique label name
        self._anon_label_counter += 1
        return f"__{base}_{self._anon_label_counter}"

    def emit(self, opcode: OpCode, arg: int = 0): #generates the instruction and puts it to buffer
        b = encode_instr(opcode, arg)
        self.code += b
        addr = self.pc
        self.debug.append((addr, b, f"{opcode.name} {arg}"))
        self.pc += INSTR_SIZE
        return addr

    def add_pstr(self, s: str):
        """Allocate pstr in the reserved pool if possible, otherwise append at end."""
        if s in self.consts:
            return self.consts[s]
        L = len(s)
        # try allocate inside pool
        if self.pool_next + 1 + L <= self.string_pool_cells:
            addr = self.pool_next
            # write header and chars *into* preallocated data area
            off = addr * 4
            # header
            self.data[off:off+4] = struct.pack("<i", L)
            off += 4
            for ch in s:
                self.data[off:off+4] = struct.pack("<i", ord(ch) & 0xFF)
                off += 4
            self.pool_next += 1 + L
            self.consts[s] = addr
            return addr
        else:
            # fallback: append at end of data (after pool and any globals)
            addr = len(self.data) // 4
            self.data += struct.pack("<i", L)
            for ch in s:
                self.data += struct.pack("<i", ord(ch) & 0xFF)
            self.consts[s] = addr
            return addr

    def compile(self):
        # compile top-level non-defun forms as main
        for node in self.ast:
            if isinstance(node, list) and len(node) > 0 and node[0] == 'defun':
                continue
            self.compile_expr(node)
        self.emit(OpCode.HALT, 0)
        # compile defuns after main
        for node in self.ast:
            if isinstance(node, list) and len(node) > 0 and node[0] == 'defun':
                self.compile_defun(node)
        # patch labels
        for pos, label in self.patch:
            if label not in self.labels:
                raise RuntimeError("unknown label " + label)
            addr = self.labels[label]
            self.code[pos+1:pos+5] = struct.pack("<i", addr)
            for i, (a, b, m) in enumerate(self.debug):
                if a == pos:
                    self.debug[i] = (a, self.code[a:a+INSTR_SIZE], f"{m.split()[0]} {addr}")
                    break

    # collect local names: args first, then any setq targets found in body
    def _collect_local_names(self, args, body):
        locals_list = []
        if isinstance(args, list):
            for a in args:
                if a not in locals_list:
                    locals_list.append(a)
        def walk(node): #recursive traversal over ast nodes
            if isinstance(node, list) and node:
                if node[0] == 'setq' and len(node) >= 2 and isinstance(node[1], str):
                    v = node[1]
                    if v not in locals_list:
                        locals_list.append(v)
                for sub in node:
                    walk(sub)
        for n in body:
            walk(n)
        return locals_list

    def compile_defun(self, node):
        # node format: ['defun', name, args_list, body...]
        if len(node) < 3:
            raise RuntimeError("malformed defun: " + repr(node))
        _, name, args, *body = node
        addr = self.pc #funct address
        self.labels[name] = addr #putting it in labels

        args_list = args if isinstance(args, list) else []
        local_names = self._collect_local_names(args_list, body)
        total_locals = len(local_names)
        num_args = len(args_list)

        # save current context and set new if there one more function call
        old_function = self.current_function
        old_locals = self.current_locals
        old_ep = self.current_epilogue

        self.current_function = name
        self.current_locals = {}
        for idx, n in enumerate(local_names):
            self.current_locals[n] = idx

        epilogue_label = self._new_label(f"epilogue_{name}")
        self.current_epilogue = epilogue_label

        # PROLOGUE: ENTER(total_locals, num_args)
        # (Prologue is the code that is executed when entering a function and prepares its stack frame)
        enter_arg = (total_locals << 16) | (num_args & 0xFFFF)
        self.emit(OpCode.ENTER, enter_arg)

        # compile function body
        for expr in body:
            self.compile_expr(expr)

        # natural end -> jump to epilogue
        jpos = self.emit(OpCode.JMP, 0)
        self.patch.append((jpos, epilogue_label))

        # EPILOGUE: LEAVE; RET
        # Epilogue is the code that closes the function:
        self.labels[epilogue_label] = self.pc
        self.emit(OpCode.LEAVE, 0)
        self.emit(OpCode.RET, 0)

        # restore context
        self.current_function = old_function
        self.current_locals = old_locals
        self.current_epilogue = old_ep

    def compile_expr(self, expr):
        if expr is None:
            return
        # integer literal
        if isinstance(expr, int):
            self.emit(OpCode.LOAD_IMM, expr)
            return
        # string literal
        if isinstance(expr, tuple) and expr[0] == 'str':
            addr = self.add_pstr(expr[1])
            self.emit(OpCode.LOAD_CONST, addr)
            return
        # symbol (variable)
        if isinstance(expr, str):
            if self.current_locals and expr in self.current_locals:
                self.emit(OpCode.LOAD_LOCAL, self.current_locals[expr])
                return
            if expr in self.globals:
                self.emit(OpCode.LOAD_MEM, self.globals[expr])
            else:
                if expr == 'nil':
                    self.emit(OpCode.LOAD_IMM, 0)
                else:
                    self.emit(OpCode.LOAD_IMM, 0)
            return

        # list / form
        if isinstance(expr, list) and len(expr) > 0:
            head = expr[0]

            # setq
            if head == 'setq':
                _, var, val = expr
                self.compile_expr(val)
                if self.current_locals is not None and isinstance(var, str) and var in self.current_locals:
                    self.emit(OpCode.STORE_LOCAL, self.current_locals[var])
                    return
                if isinstance(var, str):
                    if var not in self.globals:
                        # allocate a new global slot AFTER current data (pool already reserved)
                        slot = len(self.data) // 4
                        self.data += struct.pack("<i", 0)
                        self.globals[var] = slot
                    self.emit(OpCode.STORE_MEM, self.globals[var])
                    return
                else:
                    raise RuntimeError("setq: unsupported target " + repr(var))

            if head == 'print':
                self.compile_expr(expr[1])
                self.emit(OpCode.PRINT, 0)
                return

            if head == 'read':
                self.emit(OpCode.READ, 0)
                return

            if head == 'push':
                if len(expr) == 1:
                    self.emit(OpCode.PUSH_IMM, 0)
                else:
                    self.compile_expr(expr[1])
                    self.emit(OpCode.PUSH_IMM, 0)
                return

            if head == 'pop':
                if len(expr) == 1:
                    self.emit(OpCode.POP, 0)
                    return
                else:
                    self.emit(OpCode.POP, 0)
                    var = expr[1]
                    if isinstance(var, str):
                        if self.current_locals is not None and var in self.current_locals:
                            slot = self.current_locals[var]
                            self.emit(OpCode.STORE_LOCAL, slot)
                            return
                        else:
                            if var not in self.globals:
                                slot = len(self.data) // 4
                                self.data += struct.pack("<i", 0)
                                self.globals[var] = slot
                            self.emit(OpCode.STORE_MEM, self.globals[var])
                            return
                    else:
                        raise RuntimeError("pop: unsupported target " + repr(var))

            if head == 'defun': #compiled separately
                return

            if head == 'set-interrupt-vector':
                fname = expr[1]
                pos = self.emit(OpCode.SETVEC, 0)
                self.patch.append((pos, fname))
                return

            if head == 'ei':
                self.emit(OpCode.EI, 0)
                return

            if head == 'di':
                self.emit(OpCode.DI, 0)
                return

            if head == 'halt' or head == 'HALT':
                self.emit(OpCode.HALT, 0)
                return

            if head in ('return', 'ret', 'RET'):
                if len(expr) > 1:
                    self.compile_expr(expr[1])
                if self.current_epilogue is not None:
                    pos = self.emit(OpCode.JMP, 0)
                    self.patch.append((pos, self.current_epilogue))
                else:
                    self.emit(OpCode.RET, 0)
                return

            if head == 'call':
                fname = expr[1]
                args_nodes = expr[2:]

                # push args in order: evaluate each and PUSH_IMM so top of stack is last arg
                for argnode in args_nodes:
                    self.compile_expr(argnode)
                    self.emit(OpCode.PUSH_IMM, 0)

                pos = self.emit(OpCode.CALL, 0)
                self.patch.append((pos, fname))
                return

            if head == 'if':
                _, cond, thenb, *rest = expr
                self.compile_expr(cond)
                be_pos = self.emit(OpCode.BEQZ, 0)
                self.compile_expr(thenb)
                jmp_pos = self.emit(OpCode.JMP, 0)
                else_addr = self.pc
                self.code[be_pos+1:be_pos+5] = struct.pack("<i", else_addr)
                if rest:
                    self.compile_expr(rest[0])
                end_addr = self.pc
                self.code[jmp_pos+1:jmp_pos+5] = struct.pack("<i", end_addr)
                return

            if head == 'while':
                _, cond, *body = expr
                loop_start = self.pc
                self.compile_expr(cond)
                be_pos = self.emit(OpCode.BEQZ, 0)
                for b in body:
                    self.compile_expr(b)
                self.emit(OpCode.JMP, loop_start)
                after = self.pc
                self.code[be_pos+1:be_pos+5] = struct.pack("<i", after)
                return

            if head == 'progn':
                for sub in expr[1:]:
                    self.compile_expr(sub)
                return

            if head == 'out':
                if len(expr) < 2:
                    self.emit(OpCode.LOAD_IMM, 0)
                    return
                self.compile_expr(expr[1])
                MMIO_OUT_ADDR = 0xFFF1
                self.emit(OpCode.STORE_MEM, MMIO_OUT_ADDR)
                return

            # arithmetic n-ary: use BINOP_POP
            if head in self.binop_code:
                if len(expr) == 2:
                    self.compile_expr(expr[1])
                    return
                self.compile_expr(expr[1])
                for argnode in expr[2:]:
                    self.emit(OpCode.PUSH_IMM, 0)
                    self.compile_expr(argnode)
                    self.emit(OpCode.BINOP_POP, self.binop_code[head])
                return

            # comparisons
            if head in self.cmp_code:
                if len(expr) != 3:
                    self.emit(OpCode.LOAD_IMM, 0)
                    return
                self.compile_expr(expr[1])
                self.emit(OpCode.PUSH_IMM, 0)
                self.compile_expr(expr[2])
                self.emit(OpCode.CMP_POP, self.cmp_code[head])
                return

            # fallback: treat as call (f a b) -> (call f a b)
            parts = ['call', head] + expr[1:]
            self.compile_expr(parts)
            return
