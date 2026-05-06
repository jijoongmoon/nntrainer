#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Repack a 4-bit-in-uint8 LUT into nibble-packed (2-per-byte) form.

The CausalLM EmbeddingLayer LUT runtime (and the Gemma4 PLE loader) read
their binaries as nibble-packed bytes — two 4-bit values per uint8 with
the convention:

    out[2*k]     = byte[k] & 0x0F        (low nibble  → even index)
    out[2*k + 1] = (byte[k] >> 4) & 0x0F (high nibble → odd  index)

Some quantizer pipelines emit the same logical 4-bit values one-per-byte
(uint8 storage, only the low nibble is used). This script repacks such a
file into the nibble-packed layout so the runtime can consume it without
any per-file format flag.

Per-row size halves; the manifest's `size` (per-row element count) and
`quant-param.scale`/`offset` stay unchanged. Only `lut-path` should
point to the repacked file (and the `datatype` field stays "ufixed8" —
in this project's convention that name spans both the unpacked and the
packed forms).

Usage:
    python pack_4bit_lut.py <input> [output]
    python pack_4bit_lut.py --check <packed_file>
    python pack_4bit_lut.py --batch <input_dir> --out <output_dir>

Examples:
    # repack one file
    python pack_4bit_lut.py embed_lut.unpacked.bin embed_lut.bin

    # repack PLE in-place-style (different path)
    python pack_4bit_lut.py gemma_4_E2B_ple_quantized.unpacked.bin \
                            gemma_4_E2B_ple_quantized.bin

    # quickly verify a packed file roundtrips
    python pack_4bit_lut.py --check embed_lut.bin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


CHUNK = 1 << 22  # 4 MiB; balances RAM use and syscall count.


def pack_stream(in_path: Path, out_path: Path, *, strict: bool = True) -> int:
    """Stream `in_path` into `out_path` packing two consecutive bytes per byte.

    Returns the number of output bytes written. Streams in CHUNK-sized
    blocks so multi-GB files (e.g. PLE LUTs at vocab × 8960 bytes)
    do not blow up RAM. Handles a chunk boundary that splits a pair by
    carrying the leftover byte into the next iteration.
    """
    out_bytes = 0
    carry: int | None = None  # leftover low nibble awaiting its high nibble

    with in_path.open("rb") as fin, out_path.open("wb") as fout:
        while True:
            buf = fin.read(CHUNK)
            if not buf:
                break

            if carry is not None:
                # Pair the carried byte with the first byte of this chunk.
                first = buf[0]
                if strict and (carry > 0x0F or first > 0x0F):
                    raise ValueError(
                        f"Out-of-range nibble at boundary: "
                        f"carry={carry:#x}, first={first:#x}"
                    )
                fout.write(bytes([((first & 0x0F) << 4) | (carry & 0x0F)]))
                out_bytes += 1
                buf = buf[1:]
                carry = None

            n = len(buf)
            if n == 0:
                continue

            if strict:
                # Cheap range check: max byte must be ≤ 15.
                m = max(buf)
                if m > 0x0F:
                    bad_idx = next(i for i, b in enumerate(buf) if b > 0x0F)
                    raise ValueError(
                        f"Byte at offset {fin.tell() - n + bad_idx} "
                        f"is {buf[bad_idx]:#x}, expected ≤ 0x0F"
                    )

            # Pack pairs.
            pair_count = n // 2
            packed = bytearray(pair_count)
            # Pure-Python tight loop. memoryview gets us indexed reads
            # without per-iteration slice copies.
            mv = memoryview(buf)
            for i in range(pair_count):
                lo = mv[2 * i] & 0x0F
                hi = mv[2 * i + 1] & 0x0F
                packed[i] = (hi << 4) | lo
            fout.write(bytes(packed))
            out_bytes += pair_count

            if n & 1:
                carry = buf[-1]

        # Trailing odd byte: pair with a zero high nibble.
        if carry is not None:
            if strict and carry > 0x0F:
                raise ValueError(f"Final byte {carry:#x} out of range")
            fout.write(bytes([(carry & 0x0F)]))
            out_bytes += 1

    return out_bytes


def check_file(path: Path) -> int:
    """Sanity check on a packed file: reports size and nibble histogram top.

    Doesn't verify correctness end-to-end (we'd need the original), but
    catches the common mistake of pointing at an unpacked file by accident
    — most bytes will then be ≤ 15, so high nibbles are mostly zero.
    """
    sz = path.stat().st_size
    print(f"[check] {path}: {sz:,} bytes ({sz / 2**20:.1f} MiB)")

    counts = [0] * 16
    high_zero = 0
    seen = 0
    with path.open("rb") as f:
        while True:
            buf = f.read(CHUNK)
            if not buf:
                break
            seen += len(buf)
            for b in buf:
                counts[b & 0x0F] += 1
                counts[(b >> 4) & 0x0F] += 1
                if (b >> 4) == 0:
                    high_zero += 1
            # Histogram on a few MB is enough.
            if seen >= 8 * (1 << 20):
                break

    pct_high_zero = high_zero / seen * 100 if seen else 0
    print(
        f"[check] sampled {seen:,} bytes; nibble histogram: "
        + ", ".join(f"{i:x}={counts[i]}" for i in range(16))
    )
    print(f"[check] high-nibble-zero rate: {pct_high_zero:.1f}%")
    if pct_high_zero > 90:
        print(
            "[warn] >90% of bytes have a zero high nibble — this file looks "
            "UNPACKED (one 4-bit value per byte). Re-run pack_4bit_lut.py.",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Repack a 4-bit-in-uint8 LUT into nibble-packed (2-per-byte) "
            "form so the CausalLM runtime can consume it directly."
        )
    )
    p.add_argument("input", type=Path, help="Input file path (or directory if --batch)")
    p.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output file path (default: <input>.packed)",
    )
    p.add_argument(
        "--no-strict",
        action="store_true",
        help="Skip the [0,15] range check and silently mask to low nibble.",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Don't pack — inspect the file and warn if it looks unpacked.",
    )
    p.add_argument(
        "--batch",
        action="store_true",
        help="Treat `input` as a directory; repack every *.bin into --out dir.",
    )
    p.add_argument(
        "--out",
        type=Path,
        help="Output directory for --batch mode.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    args = p.parse_args()

    if args.check:
        return check_file(args.input)

    if args.batch:
        if args.out is None:
            print("error: --batch requires --out <dir>", file=sys.stderr)
            return 1
        in_dir: Path = args.input
        out_dir: Path = args.out
        if not in_dir.is_dir():
            print(f"error: {in_dir} is not a directory", file=sys.stderr)
            return 1
        out_dir.mkdir(parents=True, exist_ok=True)
        rc = 0
        for src in sorted(in_dir.glob("*.bin")):
            dst = out_dir / src.name
            if dst.exists() and not args.force:
                print(f"[skip] {dst} exists (use --force to overwrite)")
                continue
            try:
                in_sz = src.stat().st_size
                out_sz = pack_stream(src, dst, strict=not args.no_strict)
                print(f"[ok]   {src.name}: {in_sz:,} -> {out_sz:,} bytes")
            except ValueError as e:
                print(f"[fail] {src.name}: {e}", file=sys.stderr)
                rc = 2
        return rc

    in_path: Path = args.input
    out_path: Path = args.output or in_path.with_suffix(in_path.suffix + ".packed")

    if not in_path.exists():
        print(f"error: input not found: {in_path}", file=sys.stderr)
        return 1
    if out_path.exists() and not args.force:
        print(
            f"error: refusing to overwrite {out_path} (use --force)",
            file=sys.stderr,
        )
        return 1

    try:
        in_sz = in_path.stat().st_size
        out_sz = pack_stream(in_path, out_path, strict=not args.no_strict)
        print(
            f"packed {in_path} ({in_sz:,} bytes) -> "
            f"{out_path} ({out_sz:,} bytes)"
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
