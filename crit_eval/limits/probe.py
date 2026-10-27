"""Measure the device ceilings crit-eval compares engines with: `crit eval probe --out device.json`.

Builds probe/device_probe.cu with nvcc (CUDA toolkit with cuBLASLt), runs it several times and
keeps the median of each value: achievable copy bandwidth (GB/s, bytes read plus written) for
the decode pass floor, dense cuBLASLt GEMM TFLOPS per weight format (bf16, fp8, nvfp4) and raw
tensor-core (mma.sync) TFLOPS per input and accumulator format (bf16, fp16, fp16_acc, fp8, nvfp4,
fp4_fp8) for the prefill floor, which takes the better of the two per format
(crit_eval.limits.ceiling.best_rates). It also keeps the least free device memory any run saw, for the max context,
the host's physical and available memory, and host-to-device rates per transfer size (pinned
copy, pageable copy, mapped read) for engines that keep weights in host memory. Pass the file to `crit eval run --device
device.json`. Probe on the machine and in the environment the engine runs in, with the same
other GPU load as the evaluation.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .. import EVAL_DIR, portable_path, write_json
from ..reports import metrics

SOURCE = EVAL_DIR / "probe" / "device_probe.cu"


def find_nvcc(explicit=None):
    candidates = [explicit] if explicit else []
    candidates.append(shutil.which("nvcc"))
    for root in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda"):
        if root:
            candidates.append(str(Path(root) / "bin" / "nvcc"))
    candidates += sorted((str(p) for p in Path("/usr/local").glob("cuda-*/bin/nvcc")), reverse=True)
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise SystemExit("nvcc not found: install the CUDA toolkit or pass --nvcc PATH")


def host_compiler():
    """nvcc arguments naming the newest MSVC x64 compiler on Windows when cl.exe is not on PATH (outside a
    developer prompt); nvcc sets up the compiler's environment from -ccbin itself. Elsewhere none."""
    if os.name != "nt" or shutil.which("cl"):
        return []
    vswhere = (Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) /
               "Microsoft Visual Studio" / "Installer" / "vswhere.exe")
    if not vswhere.is_file():
        return []
    done = subprocess.run([str(vswhere), "-products", "*", "-requires",
                           "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
                          capture_output=True, text=True)
    compilers = [cl for line in done.stdout.splitlines() if line.strip()
                 for cl in (Path(line.strip()) / "VC" / "Tools" / "MSVC").glob("*/bin/Hostx64/x64/cl.exe")]
    version = lambda cl: tuple(int(part) if part.isdigit() else 0 for part in cl.parents[3].name.split("."))
    return ["-ccbin", str(max(compilers, key=version).parent)] if compilers else []


def architecture():
    """nvcc's target for the device the probe measures: its compute capability, with the architecture-specific
    suffix from 9.0 on (sm_90a, sm_100a, sm_120a), whose raw FP4 tensor-core instructions the probe times;
    -arch=native when nvidia-smi cannot name it."""
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader", "-i", "0"],
                              capture_output=True, text=True, timeout=30)
        major, minor = done.stdout.strip().split(".")
        cc = int(major) * 10 + int(minor)
    except (OSError, ValueError, subprocess.SubprocessError):
        return ["-arch=native"]
    suffix = "a" if cc >= 90 else ""
    return ["-gencode", f"arch=compute_{cc}{suffix},code=sm_{cc}{suffix}"]


def build(nvcc, directory):
    binary = Path(directory) / ("device_probe.exe" if os.name == "nt" else "device_probe")
    # The host measurements use this CPU's vector width (AVX2 on Windows, every extension elsewhere).
    vector = ["-Xcompiler", "/arch:AVX2"] if os.name == "nt" else ["-Xcompiler", "-march=native"]
    command = [nvcc, *host_compiler(), "-O2", "-std=c++17", *vector, "-o", str(binary), str(SOURCE), "-lcublasLt"]
    # The device's own architecture; an nvcc that does not know it builds for the native one.
    for target in (architecture(), ["-arch=native"]):
        done = subprocess.run(command[:1] + target + command[1:], capture_output=True, text=True)
        if done.returncode == 0:
            return binary
    # nvcc reports a missing host compiler on stdout
    raise SystemExit("building the device probe failed:\n" + (done.stdout + done.stderr)[-4000:])


def host_to_device(samples, key="host_to_device_gbs"):
    """Median transfer GB/s (`key`: host_to_device_gbs or device_to_host_gbs) per kind and transfer size over the
    samples' nonzero (measured) values; a size no sample measured, and a kind with no measured size, are left out."""
    curves = {}
    for kind in sorted({k for s in samples for k in s.get(key, {})}):
        sizes = sorted({n for s in samples for n in s.get(key, {}).get(kind, {})}, key=int)
        curve = {}
        for size in sizes:
            measured = [s[key][kind][size] for s in samples if s.get(key, {}).get(kind, {}).get(size)]
            if measured:
                curve[size] = metrics.median(measured)
        if curve:
            curves[kind] = curve
    return curves


def medians(samples, key):
    """Median of each measured (nonzero) value of the samples' `key` dict; None when none measured it."""
    names = sorted({name for s in samples for name in s.get(key) or {}})
    found = {name: metrics.median(values) for name in names
             for values in [[s[key][name] for s in samples if (s.get(key) or {}).get(name)]] if values}
    return found or None


def combine(samples):
    """Median of each measured value over repeated probe runs."""
    first = samples[0]
    formats = sorted({f for s in samples for f in s.get("tflops", {})})
    return {
        "device": first.get("device"), "compute_capability": first.get("compute_capability"),
        "cuda_runtime": first.get("cuda_runtime"),
        "bandwidth_gbs": metrics.median([s["bandwidth_gbs"] for s in samples if s.get("bandwidth_gbs")]),
        "tflops": {f: metrics.median([s["tflops"][f] for s in samples if f in s.get("tflops", {})]) for f in formats},
        # Raw tensor-core TFLOPS per input and accumulator format (mma.sync, operands in registers).
        "mma_tflops": medians(samples, "mma_tflops"),
        # Host to device GB/s per kind (pinned, pageable, mapped_read) and transfer size in bytes, and back.
        "host_to_device_gbs": host_to_device(samples),
        "device_to_host_gbs": host_to_device(samples, "device_to_host_gbs"),
        # The host: memory read, write and copy GB/s (all threads, and read on one) and CPU FP32 GFLOPS.
        "host_threads": first.get("host_threads"), "cpu_isa": first.get("cpu_isa"),
        "host_memory_gbs": medians(samples, "host_memory_gbs"),
        "cpu_gflops": medians(samples, "cpu_gflops"),
        "memory_total_bytes": max((s["memory_total_bytes"] for s in samples if s.get("memory_total_bytes")), default=None),
        # The least free memory any run saw: the max context must fit the device as it is shared.
        "memory_free_bytes": min((s["memory_free_bytes"] for s in samples if s.get("memory_free_bytes")), default=None),
    }


def host_memory():
    """(physical, available) host memory in bytes, or (None, None) where it cannot be read."""
    try:
        if sys.platform == "win32":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("available", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong),
                            ("page_available", ctypes.c_ulonglong), ("virtual_total", ctypes.c_ulonglong),
                            ("virtual_available", ctypes.c_ulonglong), ("extended", ctypes.c_ulonglong)]
            status = Status()
            status.length = ctypes.sizeof(Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.total, status.available
            return None, None
        fields = {}
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                name, _, value = line.partition(":")
                fields[name] = int(value.split()[0]) * 1024
        return fields.get("MemTotal"), fields.get("MemAvailable")
    except (OSError, ValueError, AttributeError):
        return None, None


def probe(out, nvcc=None, repeats=3, log=print):
    nvcc = find_nvcc(nvcc)
    with tempfile.TemporaryDirectory() as directory:
        binary = build(nvcc, directory)
        samples = []
        for i in range(repeats):
            done = subprocess.run([str(binary)], capture_output=True, text=True)
            if done.returncode != 0:
                raise SystemExit("device probe failed:\n" + (done.stdout + done.stderr)[-4000:])
            sample = json.loads(done.stdout)
            samples.append(sample)
            log(f"probe {i + 1}/{repeats}: {sample['bandwidth_gbs']:.0f} GB/s, "
                + ", ".join(f"{k} {v:.0f} TFLOPS" for k, v in sample.get("tflops", {}).items())
                + "; raw MMA " + (", ".join(f"{k} {v:.0f}" for k, v in (sample.get("mma_tflops") or {}).items()) or "none")
                + "; host memory read {read:.0f} GB/s, CPU {fp32:.0f} GFLOPS FP32".format(
                    read=(sample.get("host_memory_gbs") or {}).get("read", 0),
                    fp32=(sample.get("cpu_gflops") or {}).get("fp32", 0)))
    result = combine(samples)
    result["host_memory_total_bytes"], result["host_memory_available_bytes"] = host_memory()
    result.update(samples=samples, nvcc=portable_path(nvcc),
                  probe_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  measured_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(out, result)
    return result
