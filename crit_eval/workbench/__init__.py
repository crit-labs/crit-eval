"""Measurement of a kernel on crit's kernel workbench (a folder `crit kernels new` scaffolds: kernel.json, kernel.py,
harness.py, src/).

`crit eval workbench check FOLDER` compares the kernel with its FP64 reference on every shape and column count, bit
for bit across repeats and, when declared, across column counts; `bench` times it and its yardstick on inputs cycled
past the L2, after the clocks have warmed, against the floor from a `crit eval probe` result, and writes evidence a
plan's `yardstick` cites; `profile` reports the compiled kernels' registers, shared memory and spills (cuobjdump),
and Nsight Compute's sections with `--ncu`. The folder's harness runs in the engine's own Python (`--python`): torch
and the GPU are the engine's.
"""
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

HARNESS = Path(__file__).resolve().parent / "harness.py"


def run_harness(folder, command, python=None, shape=None, iters=None, device=None, out=None):
    """Run the folder's harness (check, bench or run) in `python`; returns (exit code, the JSON it wrote or None)."""
    folder = Path(folder).absolute()
    if not (folder / "harness.py").is_file():
        raise SystemExit(f"{folder} is no workbench folder (crit kernels new)")
    out = Path(out).absolute() if out else (folder / f"{command}.json" if command != "run" else None)
    argv = [python or sys.executable, str(folder / "harness.py"), command]
    if shape is not None:
        argv += ["--shape", str(shape)]
    if iters is not None:
        argv += ["--iters", str(iters)]
    if device is not None:
        argv += ["--device", str(Path(device).absolute())]
    if out is not None:
        argv += ["--out", str(out)]
    code = subprocess.run(argv, cwd=folder).returncode
    return code, (json.loads(out.read_text(encoding="utf-8")) if out is not None and out.is_file() else None)


RESOURCE = re.compile(r"Function (\S+):\s*REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)")


def resource_usage(text):
    """cuobjdump -res-usage lines: {function: {registers, stack, shared, local (spill bytes)}}."""
    return {m[1]: {"registers": int(m[2]), "stack": int(m[3]), "shared": int(m[4]), "local": int(m[5])}
            for m in RESOURCE.finditer(text)}


def profile(folder, python=None, ncu=None, shape=0):
    """Registers, shared memory and spills of every kernel the folder built, and Nsight Compute's SpeedOfLight,
    Occupancy and WarpStateStats sections over one run of the harness when `ncu` is given."""
    folder = Path(folder).absolute()
    libraries = sorted((folder / "build").glob("*.so")) + sorted((folder / "build").glob("*.pyd"))
    if not libraries:
        raise SystemExit(f"{folder}/build holds no compiled kernel: run crit eval workbench check first")
    nvcc = shutil.which("nvcc")
    cuobjdump = shutil.which("cuobjdump") or (str(Path(nvcc).with_name("cuobjdump")) if nvcc else None)
    if not cuobjdump:
        raise SystemExit("cuobjdump not found: put the CUDA toolkit's bin on PATH")
    result = {"kind": "kernels_profile", "kernel": folder.name, "functions": {}}
    for library in libraries:
        dumped = subprocess.run([cuobjdump, "-res-usage", str(library)], capture_output=True, text=True)
        result["functions"].update(resource_usage(dumped.stdout))
    if ncu:
        argv = [ncu, "--section", "SpeedOfLight", "--section", "Occupancy", "--section", "WarpStateStats",
                python or sys.executable, str(folder / "harness.py"), "run", "--shape", str(shape)]
        done = subprocess.run(argv, cwd=folder, capture_output=True, text=True)
        (folder / "profile-ncu.txt").write_text(done.stdout + done.stderr, encoding="utf-8")
        result["ncu"] = "profile-ncu.txt" if done.returncode == 0 else f"ncu failed ({done.returncode}): see profile-ncu.txt"
    (folder / "profile.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
