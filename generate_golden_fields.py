#!/usr/bin/env python3
# tools/generate_golden_fields.py
"""
Сгенерировать out_code (!!binary) и out_code_hex для golden YAML.
Usage: python generate_golden_fields.py path/to/golden.yaml
"""

import sys
import io
import yaml
import os
from parser import tokenize, parse, Compiler
from isa import decode_instr, INSTR_SIZE, mnemonic

def compile_source_to_code_bytes(src_code):
    toks = tokenize(src_code)
    ast = parse(toks)
    comp = Compiler(ast)
    comp.compile()
    return bytes(comp.code), bytes(comp.data)

def build_code_hex(code_bytes):
    lines = []
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
            lines.append(f"{pc} - {hexbytes} - {mnem}")
        except Exception as e:
            # dump remainder as hex if decode fails
            rest = code_bytes[pc:].hex().upper()
            lines.append(f"{pc} - {rest} - <decode error: {e}>")
            break
        pc += INSTR_SIZE
    return "\n".join(lines)

def main(path):
    if not os.path.exists(path):
        print("File not found:", path)
        sys.exit(2)

    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    # find source (support colleague style in_source too)
    src_spec = doc.get("source")
    if not src_spec and "in_source" in doc:
        src_spec = {"language": "lisp", "code": doc["in_source"]}

    if not src_spec:
        print("No 'source' or 'in_source' found in YAML — nothing to compile")
        sys.exit(2)

    lang = src_spec.get("language", "lisp")
    if lang != "lisp":
        print("Unsupported language for auto-generation:", lang)
        sys.exit(2)

    src_code = src_spec.get("code", "")
    code_bytes, data_bytes = compile_source_to_code_bytes(src_code)

    # insert out_code (bytes) — PyYAML will emit !!binary
    out = doc.get("expect") or doc.get("out") or doc
    # choose where to place: prefer expect then out then top-level 'expect'
    if "expect" in doc:
        target = doc["expect"]
    elif "out" in doc:
        target = doc["out"]
    else:
        # create expect section if none
        doc["expect"] = {}
        target = doc["expect"]

    target["out_code"] = code_bytes  # bytes -> yaml !!binary
    target["out_code_hex"] = build_code_hex(code_bytes)

    # optionally add out_data if you want data bytes included:
    if data_bytes:
        target["out_data"] = data_bytes

    # write back YAML (use block style where possible)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"Updated {path} with out_code (!!binary) and out_code_hex (text).")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: generate_golden_fields.py path/to/golden.yaml")
        sys.exit(1)
    main(sys.argv[1])
