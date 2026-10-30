"""The workbench harness: checks a kernel against its reference and times it against its floor and a yardstick.

Written by `crit kernels new` beside kernel.py and kernel.json; `crit kernels check|bench|profile` run it in the engine's
own Python (torch and the GPU are the engine's). kernel.py supplies:

    inputs(shape, columns, seed) -> dict     the operation's inputs for one shape and column (token) count
    reference(inputs) -> tensor              the exact result, computed in float64
    candidate(inputs) -> tensor              the kernel under work
    yardstick(inputs) -> tensor or None      the best existing provider for the same arithmetic (a library or Hub kernel)
    work(shape, columns) -> {"bytes", "flops"}  what one call must move and compute, for the floor

usage: python harness.py check|bench|run [--shape INDEX] [--iters N] [--device DEVICE_JSON] [--out FILE]
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

import torch

HERE = Path(__file__).resolve().parent


def load():
    spec = importlib.util.spec_from_file_location("kernel", HERE / "kernel.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["kernel"] = module
    spec.loader.exec_module(module)
    return module, json.loads((HERE / "kernel.json").read_text(encoding="utf-8"))


def relative_error(found, expected):
    expected = expected.double()
    scale = expected.abs().max().item() or 1.0
    return (found.double() - expected).abs().max().item() / scale


def check(kernel, spec, shapes):
    rows, failed = [], 0
    for index, shape in shapes:
        for columns in spec["columns"]:
            data = kernel.inputs(shape, columns, seed=index * 1000 + columns)
            found = kernel.candidate(data)
            error = relative_error(found, kernel.reference(data))
            row = {"shape": shape, "columns": columns, "relative_error": error, "passed": error <= spec["tolerance"]}
            if spec.get("column_invariant") and columns > 1:
                # Each column computed alone gives the same bits as inside the multi-column call.
                alone = [kernel.candidate(kernel.select(data, c)) for c in range(columns)]
                row["column_invariant"] = all(torch.equal(a.reshape(-1), found.select(0, c).reshape(-1))
                                              for c, a in enumerate(alone))
                row["passed"] = row["passed"] and row["column_invariant"]
            # Determinism: the same inputs again give the same bits (catches atomics, races and reads past a buffer).
            row["repeatable"] = all(torch.equal(kernel.candidate(data), found) for _ in range(2))
            row["passed"] = row["passed"] and row["repeatable"]
            failed += not row["passed"]
            rows.append(row)
            print(f"{'ok  ' if row['passed'] else 'FAIL'} shape {shape} columns {columns}: relative error {error:.2e}, "
                  f"repeatable {row['repeatable']}"
                  + ("" if "column_invariant" not in row else f", column invariant {row['column_invariant']}"))
    return {"kind": "kernels_check", "kernel": spec["name"], "tolerance": spec["tolerance"], "rows": rows,
            "passed": failed == 0}


def copies_for(work):
    """Input copies to cycle through so the weights come from memory, not the L2: their bytes over three L2s."""
    l2 = torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size
    return max(1, min(64, math.ceil(3 * l2 / max(1, work["bytes"]))))


def time_call(fn, copies, iters):
    # Warm up for a quarter second, so the clocks have left their idle state before anything is timed.
    import time
    started, i = time.perf_counter(), 0
    while time.perf_counter() - started < 0.25 or i < 3:
        fn(copies[i % len(copies)])
        torch.cuda.synchronize()
        i += 1
    times = []
    for i in range(iters):
        data = copies[i % len(copies)]
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn(data)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(times)


def floor_us(work, device, rate):
    """The call's floor: its bytes at the device's bandwidth or its flops at the given TFLOPS rate, the larger."""
    if not device:
        return None
    parts = [work["bytes"] / (device["bandwidth_gbs"] * 1e3)]
    if rate:
        parts.append(work["flops"] / (rate * 1e6))
    return max(parts)


def bench(kernel, spec, shapes, iters, device):
    rate = None
    if device:
        rates = {**(device.get("tflops") or {}), **(device.get("mma_tflops") or {})}
        rate = rates.get(spec.get("rate") or "")
    rows = []
    for index, shape in shapes:
        for columns in spec["columns"]:
            work = kernel.work(shape, columns)
            copies = [kernel.inputs(shape, columns, seed=index * 1000 + c) for c in range(copies_for(work))]
            row = {"shape": shape, "columns": columns, "bytes": work["bytes"], "flops": work["flops"],
                   "copies": len(copies), "candidate_us": time_call(kernel.candidate, copies, iters)}
            if kernel.yardstick(copies[0]) is not None:
                row["yardstick_us"] = time_call(kernel.yardstick, copies, iters)
            row["floor_us"] = floor_us(work, device, rate)
            for side in ("candidate", "yardstick"):
                us = row.get(side + "_us")
                if us:
                    row[side + "_gbs"] = work["bytes"] / (us * 1e3)
                    row[side + "_tflops"] = work["flops"] / (us * 1e6)
                    if row["floor_us"]:
                        row[side + "_fraction_of_floor"] = row["floor_us"] / us
            rows.append(row)
            print(f"shape {shape} columns {columns}: candidate {row['candidate_us']:.1f} us"
                  + (f", yardstick ({spec['yardstick']}) {row['yardstick_us']:.1f} us" if "yardstick_us" in row else "")
                  + (f", floor {row['floor_us']:.1f} us" if row["floor_us"] else ""))
    ratio = [r["yardstick_us"] / r["candidate_us"] for r in rows if r.get("yardstick_us")]
    return {"kind": "kernels_bench", "kernel": spec["name"], "yardstick": spec.get("yardstick"),
            "device": (device or {}).get("device"), "iters": iters, "rows": rows,
            "speed_over_yardstick": math.exp(sum(map(math.log, ratio)) / len(ratio)) if ratio else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "bench", "run"))
    parser.add_argument("--shape", type=int, help="index of one shape in kernel.json (default: all)")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--device", type=Path, help="a `crit eval probe` result, for the floor")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    kernel, spec = load()
    shapes = [(i, s["shape"]) for i, s in enumerate(spec["shapes"]) if args.shape is None or i == args.shape]
    if args.command == "run":   # one call per shape and column count, for a profiler
        for index, shape in shapes:
            for columns in spec["columns"]:
                kernel.candidate(kernel.inputs(shape, columns, seed=index))
        torch.cuda.synchronize()
        return 0
    device = json.loads(args.device.read_text(encoding="utf-8")) if args.device else None
    result = check(kernel, spec, shapes) if args.command == "check" else bench(kernel, spec, shapes, args.iters, device)
    # Provenance: what was measured, so evidence can be matched to the package that ships it.
    import hashlib
    result["gpu"] = torch.cuda.get_device_name()
    result["sources"] = {p.relative_to(HERE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(HERE.rglob("*")) if p.is_file() and p.suffix in {".cu", ".cuh", ".h", ".cpp", ".py"}
                         and "build" not in p.relative_to(HERE).parts}
    result["toolchain"] = {"torch": torch.__version__, "cuda": torch.version.cuda}
    if args.out:
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
