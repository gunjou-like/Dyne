import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_wasm")
    parser.add_argument("output_header")
    parser.add_argument("--var_name", default="wasm_blob")
    args = parser.parse_args()

    with open(args.input_wasm, "rb") as f:
        data = f.read()

    print(f"Converting {len(data)} bytes to C header...")

    with open(args.output_header, "w") as f:
        f.write(f"#ifndef {args.var_name.upper()}_H\n")
        f.write(f"#define {args.var_name.upper()}_H\n\n")
        f.write(f"const unsigned char {args.var_name}[] = {{\n")
        
        for i, byte in enumerate(data):
            f.write(f"0x{byte:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n")
        
        f.write("\n};\n\n")
        f.write(f"const unsigned int {args.var_name}_len = {len(data)};\n")
        f.write("#endif\n")

    print(f"✅ Saved to {args.output_header}")

if __name__ == "__main__":
    main()