"""Measure the device ceilings crit-eval compares engines with: `crit eval probe --out device.json`.

Builds probe/device_probe.cu with nvcc (CUDA toolkit with cuBLASLt), or on ROCm probe/device_probe_hip.cpp with
hipcc (hipBLASLt; --backend hip, chosen by itself when hipcc is found and nvcc is not), runs it several times and
keeps the median of each value: achievable copy bandwidth (GB/s, bytes read plus written) for
the decode pass floor, dense cuBLASLt GEMM TFLOPS per weight format (bf16, fp8, nvfp4) and raw
tensor-core (mma.sync) TFLOPS per input and accumulator format (bf16, fp16, fp16_acc, fp8, nvfp4,
fp4_fp8) for the prefill floor, which takes the better of the two per format
(crit_eval.limits.ceiling.best_rates). It also keeps the least free device memory any run saw, for the max context,
the host's physical and available memory, and host-to-device rates per transfer size (pinned
copy, pageable copy, mapped read) for engines that keep weights in host memory. Pass the file to `crit eval run --device
device.json`. Probe on the machine and in the environment the engine runs in, with the same
other GPU load as the evaluation.

Every result names its backend (cuda or hip), architecture (sm_120a, gfx1201) and runtime version, and lists the rate
keys the device has no matrix units for (unsupported_formats), which the prefill floor prices at the BF16 rate
(crit_eval.limits.ceiling.best_rates).
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
from . import ceiling

SOURCE = EVAL_DIR / "probe" / "device_probe.cu"
# The probe source of each backend.
SOURCES = {"cuda": SOURCE, "hip": EVAL_DIR / "probe" / "device_probe_hip.cpp"}


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


def find_hipcc(explicit=None):
    candidates = [explicit] if explicit else []
    candidates.append(shutil.which("hipcc"))
    for root in (os.environ.get("ROCM_PATH"), os.environ.get("HIP_PATH"), "/opt/rocm"):
        if root:
            candidates.append(str(Path(root) / "bin" / ("hipcc.exe" if os.name == "nt" else "hipcc")))
    candidates += sorted((str(p) for p in Path("/opt").glob("rocm-*/bin/hipcc")), reverse=True)
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise SystemExit("hipcc not found: install ROCm or pass --hipcc PATH")


def choose_backend(backend=None, nvcc=None, hipcc=None):
    """The backend to probe: the one asked for; else cuda when an nvcc is given or found, hip when only hipcc is."""
    if backend:
        if backend not in SOURCES:
            raise SystemExit(f"--backend is one of {', '.join(SOURCES)}")
        return backend
    if nvcc or shutil.which("nvcc") or not (hipcc or shutil.which("hipcc")):
        return "cuda"
    return "hip"


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


def build_hip(hipcc, directory):
    binary = Path(directory) / ("device_probe_hip.exe" if os.name == "nt" else "device_probe_hip")
    vector = ["/arch:AVX2"] if os.name == "nt" else ["-march=native"]
    # --offload-arch=native: the GPU the probe measures.
    command = [hipcc, "-O2", "-std=c++17", "--offload-arch=native", *vector, "-o", str(binary),
               str(SOURCES["hip"]), "-lhipblaslt"]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit("building the HIP device probe failed:\n" + (done.stdout + done.stderr)[-4000:])
    return binary


def identity(sample):
    """(backend, arch, runtime_version) of one probe sample: a HIP sample names them; a CUDA sample's arch is its
    compute capability as nvcc names the architecture-specific target from 9.0 on (sm_90a, sm_120a)."""
    if sample.get("backend"):
        return sample["backend"], sample.get("arch"), sample.get("runtime_version")
    try:
        major, minor = (int(part) for part in str(sample.get("compute_capability")).split("."))
        arch = f"sm_{major * 10 + minor}" + ("a" if major >= 9 else "")
    except ValueError:
        arch = None
    return "cuda", arch, sample.get("cuda_runtime")


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
    backend, arch, runtime = identity(first)
    return {
        "backend": backend, "arch": arch, "runtime_version": runtime, "device": first.get("device"),
        **({"compute_capability": first.get("compute_capability"), "cuda_runtime": first.get("cuda_runtime")}
           if backend == "cuda" else {"fp8_encoding": first.get("fp8_encoding")}),
        # Rate keys without matrix units on this device; a CUDA probe's come from its compute capability.
        "unsupported_formats": first.get("unsupported_formats",
                                         ceiling.unsupported_formats({"compute_capability": first.get("compute_capability")})),
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
        # The CPU expert lane: crit's integer block rows at 1, 3 and 8 tokens an expert on one thread per core, and at
        # 3 beside expert-sized link copies (both rates over the same window): what a CPU tier computing misses reaches.
        "host_expert_gbs": medians(samples, "host_expert_gbs"), "host_expert_isa": first.get("host_expert_isa"),
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


def run_binary(binary, *arguments):
    done = subprocess.run([str(binary), *map(str, arguments)], capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit("device probe failed:\n" + (done.stdout + done.stderr)[-4000:])
    return json.loads(done.stdout)


def device_indices(binary, devices):
    """The devices to probe: [0] for None, every one the runtime sees for "all", else the given indices."""
    if devices is None:
        return [0]
    count = run_binary(binary, "--count")["devices"]
    found = list(range(count)) if devices == "all" else [int(d) for d in devices]
    if not found or any(not 0 <= d < count for d in found) or len(set(found)) != len(found):
        raise SystemExit(f"probe devices {devices}: the runtime sees {count} device(s), 0..{count - 1}")
    return found


def probe(out, nvcc=None, repeats=3, log=print, backend=None, hipcc=None, devices=None):
    """Probe the device (device 0) and write its figures to `out`. With `devices` ("all" or indices, CUDA), every one
    of them: the file keeps the first device's figures at its top, as a single device's file does, and adds "devices"
    (each one's figures, with its index) and "links" (each ordered pair's copy rate in GB/s and whether peer access
    is on), which multi-GPU limits price (crit_eval.limits.devices)."""
    backend = choose_backend(backend, nvcc, hipcc)
    if devices is not None and backend != "cuda":
        raise SystemExit("probing several devices needs the CUDA probe")
    compiler = find_nvcc(nvcc) if backend == "cuda" else find_hipcc(hipcc)
    source = SOURCES[backend]
    with tempfile.TemporaryDirectory() as directory:
        binary = build(compiler, directory) if backend == "cuda" else build_hip(compiler, directory)
        members = []
        for device in device_indices(binary, devices):
            samples = probe_device(binary, device if devices is not None else None, repeats, log)
            members.append((device, samples))
        links = [run_binary(binary, "--peer", a, b) for a, _ in members for b, _ in members if a != b] \
            if len(members) > 1 else []
        for link in links:
            log(f"link {link['from']} -> {link['to']}: {link['gbs']:.1f} GB/s"
                + (" (peer access)" if link["peer_access"] else ""))
    result = combine(members[0][1])
    result["host_memory_total_bytes"], result["host_memory_available_bytes"] = host_memory()
    result.update(samples=members[0][1], **{"nvcc" if backend == "cuda" else "hipcc": portable_path(compiler)},
                  probe_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  measured_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if devices is not None:
        result["devices"] = [{"index": device, **combine(samples), "samples": samples} for device, samples in members]
        result["links"] = links
    write_json(out, result)
    return result


def probe_device(binary, device, repeats, log):
    """One device's samples: the probe run `repeats` times (on `device` when given, else the default device 0)."""
    samples = []
    for i in range(repeats):
        sample = run_binary(binary, *(("--device", device) if device is not None else ()))
        samples.append(sample)
        log((f"device {device} " if device is not None else "") + f"probe {i + 1}/{repeats}: {sample['bandwidth_gbs']:.0f} GB/s, "
                + ", ".join(f"{k} {v:.0f} TFLOPS" for k, v in sample.get("tflops", {}).items())
                + "; raw MMA " + (", ".join(f"{k} {v:.0f}" for k, v in (sample.get("mma_tflops") or {}).items()) or "none")
                + "; host memory read {read:.0f} GB/s, CPU {fp32:.0f} GFLOPS FP32".format(
                    read=(sample.get("host_memory_gbs") or {}).get("read", 0),
                    fp32=(sample.get("cpu_gflops") or {}).get("fp32", 0)))
    return samples


# What a probe from today measures that older probes did not, and what each prices.
MEASURED = {"mma_tflops": "raw tensor-core rates (fp16_acc, fp4_fp8 prompt activations)",
            "host_to_device_gbs": "host-to-device rates (weights kept in host memory)",
            "host_memory_gbs": "host memory rates", "cpu_gflops": "CPU rates",
            "host_expert_gbs": "the CPU expert lane (integer block rows, alone and beside link copies)"}


def outdated(device):
    """Why a `crit eval probe` result is older than this probe: the measurements it lacks (MEASURED) and a probe
    source other than this checkout's. [] for a current one, or no device file. Eval prices nothing it lacks, so a
    floor that needs it reads as missing (an fp16_acc prefill ceiling) until the device is probed again."""
    if not device:
        return []
    backend = ceiling.device_backend(device)
    # A HIP probe measures no raw matrix-instruction rates (probe/device_probe_hip.cpp).
    reasons = [f"no {what}" for key, what in MEASURED.items() if not device.get(key)
               and not (backend == "hip" and key == "mma_tflops")]
    source = SOURCES.get(backend, SOURCE)
    if device.get("probe_sha256") != hashlib.sha256(source.read_bytes()).hexdigest():
        reasons.append("measured by another probe source" if device.get("probe_sha256") else "no probe_sha256")
    return reasons
