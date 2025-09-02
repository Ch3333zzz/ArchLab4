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

    BINOP_POP = 90  # arg: 1=ADD,2=SUB,3=MUL,4=DIV  (pop v; ACC = v <op> ACC)
    CMP_POP = 91  # arg: 1==,2!=,3<,4<=,5>,6>=   (pop v; ACC = cond(v, ACC) ? 1 : 0)

    LOAD_LOCAL = 80  # ACC = frame_local_or_arg[arg]
    STORE_LOCAL = 81  # frame_local_or_arg[arg] = ACC
    ENTER = 82  # arg: (total_locals << 16) | num_args
    LEAVE = 83  # pop frame


INSTR_SIZE = 5  # 1 byte opcode + 4 byte signed little-endian arg


def encode_instr(opcode: OpCode, arg: int = 0) -> bytes:
    """Encode instruction (opcode + 32-bit little-endian signed arg)."""
    val = int(arg) & 0xFFFFFFFF
    if val & 0x80000000:
        signed = val - (1 << 32)
    else:
        signed = val
    return bytes([int(opcode)]) + struct.pack("<i", signed)


def decode_instr(blob: bytes, offset: int) -> tuple[OpCode, int]:
    """Decode instruction from bytes."""
    b = blob[offset : offset + INSTR_SIZE]
    if len(b) < INSTR_SIZE:
        err = "End of program"
        raise EOFError(err)
    opcode = OpCode(b[0])
    arg = struct.unpack("<i", b[1:5])[0]
    return opcode, arg


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
    ):
        return f"{opcode.name} {arg}"
    return opcode.name
