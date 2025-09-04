"""ISA: instruction encodings and helpers."""

import struct
from enum import IntEnum


class OpCode(IntEnum):
    """Keeps opcodes from all operations."""

    NOP = 0
    HALT = 1

    LOAD_IMM = 10  # ACC = imm
    ADD_IMM = 11  # ACC += imm
    SUB_IMM = 12  # ACC -= imm
    MUL_IMM = 13  # ACC *= imm
    DIV_IMM = 14  # ACC /= imm

    LOAD_CONST = 20  # ACC = data_word_addr
    LOAD_MEM = 21  # ACC = MEM[arg]
    STORE_MEM = 22  # MEM[arg] = ACC
    STORE_IND = 23  # MEM[addr_popped_from_stack] = ACC
    LOAD_IND = 24  # ACC = MEM[addr_popped_from_stack]

    JMP = 30
    BEQZ = 31
    BNEZ = 32
    CALL = 40
    RET = 41

    SETVEC = 50
    EI = 51
    DI = 52
    IRET = 53
    SOFTINT = 54  # software/internal interrupt

    PUSH_IMM = 60  # push ACC to stack
    POP = 61  # pop -> ACC

    PRINT = 70
    READ = 71

    LOAD_LOCAL = 80  # ACC = frame_local_or_arg[arg]
    STORE_LOCAL = 81  # frame_local_or_arg[arg] = ACC
    ENTER = 82  # arg: (total_locals << 16) | num_args
    LEAVE = 83  # pop frame

    # --- new array/heap ops ---
    ALLOC = 84  # allocate arg words (if arg==0, take size from ACC)
    ASET = 85  # array set
    AGET = 86  # array get

    BINOP_POP = 90  # arg: 1=ADD,2=SUB,3=MUL,4=DIV  (pop v; ACC = v <op> ACC)
    CMP_POP = 91  # arg: 1==,2!=,3<,4<=,5>,6>=


# Instruction size is now 8 bytes (two machine words). Machine word stays 32 bits.
# Layout (64-bit little-endian):
#   low 4 bytes  : signed 32-bit arg (little-endian)
#   high 4 bytes : opcode (low 8 bits used), other bits reserved/zero
INSTR_SIZE = 8  # 8 bytes per instruction (two 32-bit words)


def encode_instr(opcode: OpCode, arg: int = 0) -> bytes:
    """Encode instruction into 8 bytes.

    Format (little-endian 64-bit):
      word64 = (opcode << 32) | (arg & 0xFFFFFFFF)

    Arg is stored as 32-bit value (decode_instr will sign-extend to signed int).
    """
    op = int(opcode) & 0xFF
    try:
        a = int(arg)
    except Exception:
        a = 0
    a32 = a & 0xFFFFFFFF
    word64 = (op << 32) | a32
    return struct.pack("<Q", word64)


def decode_instr(blob: bytes, offset: int) -> tuple[OpCode, int]:
    """Decode instruction from bytes at offset.

    Returns (OpCode, signed_arg).
    Raises EOFError if not enough bytes.
    """
    b = blob[offset : offset + INSTR_SIZE]
    if len(b) < INSTR_SIZE:
        err = "End of program"
        raise EOFError(err)
    (word64,) = struct.unpack("<Q", b)
    op = (word64 >> 32) & 0xFF
    arg_raw = word64 & 0xFFFFFFFF
    # sign-extend 32-bit
    if arg_raw & 0x80000000:
        arg = arg_raw - (1 << 32)
    else:
        arg = arg_raw
    return OpCode(op), int(arg)


def mnemonic(opcode: OpCode, arg: int) -> str:
    """Get operation mnemonic."""
    if opcode in (
        OpCode.LOAD_IMM,
        OpCode.LOAD_CONST,
        OpCode.LOAD_MEM,
        OpCode.STORE_MEM,
        OpCode.JMP,
        OpCode.BEQZ,
        OpCode.BNEZ,
        OpCode.CALL,
        OpCode.SETVEC,
        OpCode.PUSH_IMM,
        OpCode.BINOP_POP,
        OpCode.CMP_POP,
        OpCode.STORE_IND,
        OpCode.LOAD_IND,
        OpCode.LOAD_LOCAL,
        OpCode.STORE_LOCAL,
        OpCode.ENTER,
        OpCode.LEAVE,
        OpCode.SOFTINT,
        OpCode.ALLOC,
        OpCode.ASET,
        OpCode.AGET,
    ):
        return f"{opcode.name} {arg}"
    return opcode.name
