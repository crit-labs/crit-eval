"""Energy per token in the speed cases (reported, not judged): the GPU's draw while the engine serves them.

Two engines at the same tok/s can draw very different power. While a round runs, a Meter reads the engine's GPUs
through NVML (the card's management library, which nvidia-smi reads; through ctypes) on a side thread every
INTERVAL_SECONDS: the energy counter (nvmlDeviceGetTotalEnergyConsumption, millijoules since the driver loaded) where
every GPU's driver has it, and the power draw (nvmlDeviceGetPowerUsage) for the peak, and for the energy too where
there is no counter. Before the speed suite's first request it reads the device at rest with the engine loaded for
IDLE_SECONDS (`idle_watts`).

Each timed request of a speed case is bracketed on the host's clock (metered): from the call to its first token is
the prefill, from the first token to the last the decode. A case's figures are over its measured repetitions: joules
per prompt token (the prefills' joules over their prompt tokens) and per output token (the decodes' joules over the
tokens after the first, as decode tok/s counts them), the average watts over the requests and the peak reading.
Without NVML, a GPU it cannot find or one whose driver reports neither figure, the round's energy is unmeasured and
says why; so are a case's requests that a resumed round's journal answered, whose times are the earlier round's.
"""
import ctypes
import os
from pathlib import Path
import threading
import time

from ..reports import metrics

INTERVAL_SECONDS = 0.05   # between readings: the counter and the power reading move every few tens of ms
IDLE_SECONDS = 2.0        # the device at rest with the engine loaded, read before the suite's first request
WINDOW = "energy_window"  # a timed sample's host-clock bracket (start, first token, last token), taken off by case()
LIBRARIES = ("nvml.dll", "libnvidia-ml.so.1", "/usr/lib/wsl/lib/libnvidia-ml.so.1",
             str(Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "NVIDIA Corporation/NVSMI/nvml.dll"))

_active = None   # the meter of the round in progress (Meter.start)


class Unavailable(Exception):
    """Why the GPU's energy cannot be read: the round's energy is unmeasured with it."""


class Nvml:
    """The few NVML calls a Meter makes. load() finds the library; a test gives its own object with the same methods
    (handle, power_watts, energy_joules, close)."""

    def __init__(self, lib):
        self.lib = lib

    @classmethod
    def load(cls):
        for name in LIBRARIES:
            try:
                lib = ctypes.CDLL(name)
            except OSError:
                continue
            found = lib.nvmlInit_v2()
            if found != 0:
                raise Unavailable(f"NVML did not start (nvmlInit error {found})")
            return cls(lib)
        raise Unavailable("no NVML library (nvml.dll or libnvidia-ml.so.1) on this host")

    def handle(self, key):
        """The device of nvidia-smi's index `key`, or of its UUID when `key` is a string ("GPU-...")."""
        handle = ctypes.c_void_p()
        found = (self.lib.nvmlDeviceGetHandleByUUID(ctypes.c_char_p(key.encode()), ctypes.byref(handle))
                 if isinstance(key, str) else
                 self.lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(int(key)), ctypes.byref(handle)))
        if found != 0:
            raise Unavailable(f"NVML has no GPU {key} (error {found})")
        return handle

    def power_watts(self, handle):
        milliwatts = ctypes.c_uint()
        return milliwatts.value / 1000 if self.lib.nvmlDeviceGetPowerUsage(handle, ctypes.byref(milliwatts)) == 0 \
            else None

    def energy_joules(self, handle):
        try:
            reader = self.lib.nvmlDeviceGetTotalEnergyConsumption
        except AttributeError:   # a library older than the counter
            return None
        millijoules = ctypes.c_ulonglong()
        return millijoules.value / 1000 if reader(handle, ctypes.byref(millijoules)) == 0 else None

    def close(self):
        self.lib.nvmlShutdown()


def engine_devices(indices=None, environ=None):
    """The GPUs the engine runs on, as NVML finds them: a device set's indices, else the one CUDA_VISIBLE_DEVICES names
    first (by number or UUID), else GPU 0, as the run's device memory reads them (crit_eval.run.host.device_memory).
    Raises Unavailable when CUDA_VISIBLE_DEVICES hides every GPU."""
    if indices:
        return list(indices)
    visible = (os.environ if environ is None else environ).get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return [0]
    first = visible.split(",")[0].strip()
    if not first or first == "-1":
        raise Unavailable("CUDA_VISIBLE_DEVICES hides every GPU")
    return [int(first)] if first.isdigit() else [first] if first.startswith(("GPU-", "MIG-")) else [0]


class Meter:
    """The GPUs' energy over a round (see the module). start() opens NVML and starts the side thread, or records why
    it cannot; stop() ends it. A Meter that could not start reads nothing and leaves every figure unmeasured.
    nvml: the NVML object (Nvml.load() when None); interval 0 reads only when asked (read()), for tests."""

    def __init__(self, indices=None, nvml=None, interval=None, idle_seconds=None, clock=time.perf_counter,
                 sleep=time.sleep, environ=None):
        self.interval = INTERVAL_SECONDS if interval is None else interval
        self.idle_seconds = IDLE_SECONDS if idle_seconds is None else idle_seconds
        self.clock, self.sleep = clock, sleep
        self.nvml, self.handles, self.devices, self.reason, self.source = nvml, [], None, None, None
        self.readings = []   # (host time, watts summed over the GPUs or None, joules summed or None)
        self.idle = None
        self.lock, self.done, self.thread = threading.Lock(), threading.Event(), None
        try:
            self.devices = engine_devices(indices, environ)
        except Unavailable as error:
            self.reason = str(error)

    def start(self):
        global _active
        if self.reason is None:
            try:
                self._open()
            except Unavailable as error:
                self.reason = str(error)
            except (OSError, AttributeError) as error:   # a library that will not answer, or too old
                self.reason = f"NVML failed: {error}"
        _active = self
        if self.reason is None:
            self.read()
            if self.interval:
                self.thread = threading.Thread(target=self._run, name="crit-eval-energy", daemon=True)
                self.thread.start()
        return self

    def _open(self):
        if self.nvml is None:
            self.nvml = Nvml.load()
        self.handles = [self.nvml.handle(key) for key in self.devices]
        power = [self.nvml.power_watts(h) for h in self.handles]
        energy = [self.nvml.energy_joules(h) for h in self.handles]
        if all(e is not None for e in energy):
            self.source = "energy_counter"
        elif all(p is not None for p in power):
            self.source = "power"
        else:
            missing = [str(k) for k, p, e in zip(self.devices, power, energy) if p is None and e is None]
            raise Unavailable(f"the driver reports neither power nor energy for GPU {', '.join(missing)}" if missing
                              else "the GPUs' drivers report no figure all of them have (power or energy)")

    def stop(self):
        global _active
        self.done.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        if self.nvml is not None:
            try:
                self.nvml.close()
            except (OSError, AttributeError):
                pass
        if _active is self:
            _active = None

    def _run(self):
        while not self.done.wait(self.interval):
            self.read()

    def read(self):
        """One reading of every GPU, summed; a figure one of them does not give is None for the reading."""
        if self.reason is not None:
            return
        counter = self.source == "energy_counter"
        try:
            # The reading's time brackets the calls of the figure the joules integrate (the counter, else the power)
            # and no other: a stall between the two figures' calls (the host busy elsewhere) would date the counter
            # by the power's calls, and over a 2 s idle window 5 ms of it read 250 W as 248.8.
            if counter:
                before = self.clock()
                energy = [self.nvml.energy_joules(h) for h in self.handles]
                after = self.clock()
                power = [self.nvml.power_watts(h) for h in self.handles]
            else:
                before = self.clock()
                power = [self.nvml.power_watts(h) for h in self.handles]
                after = self.clock()
                energy = [None]
        except OSError:   # a reading never fails the run
            return
        now = (before + after) / 2   # the calls' own time, between the two
        row = (now, None if None in power else sum(power), None if None in energy else sum(energy))
        with self.lock:
            self.readings.append(row)

    def measure_idle(self):
        """The device at rest before the suite's first request, once a round: idle_watts over idle_seconds."""
        if self.reason is not None or self.idle is not None:
            return
        started = self.clock()
        self.sleep(self.idle_seconds)
        ended = self.clock()
        self.read()   # a reading at or after the end
        joules = self.joules(started, ended)
        with self.lock:
            count = sum(1 for t, _, _ in self.readings if started <= t <= ended)
        self.idle = {"watts": round(joules / (ended - started), 2) if joules is not None and ended > started else None,
                     "seconds": round(ended - started, 3), "readings": count}

    def _bracket(self, a, b, field):
        """The readings with `field` from the last at or before a to the first at or after b, or None when they do not
        cover [a, b]."""
        with self.lock:
            rows = [r for r in self.readings if r[field] is not None]
        before = [i for i, r in enumerate(rows) if r[0] <= a]
        after = [i for i, r in enumerate(rows) if r[0] >= b]
        if not before or not after:
            return None
        return rows[before[-1]:after[0] + 1]

    def joules(self, a, b):
        """The GPUs' energy from host time a to b: the counter interpolated at both ends, or with power alone the
        readings' draw integrated (linear between readings). None where the readings do not cover the interval."""
        if b < a:
            return None
        field = 2 if self.source == "energy_counter" else 1
        rows = self._bracket(a, b, field)
        if rows is None:
            return None
        if field == 2:
            return _at(rows, b) - _at(rows, a)
        points = [(a, _at(rows, a, 1))] + [(t, w) for t, w, _ in rows if a < t < b] + [(b, _at(rows, b, 1))]
        return sum((t2 - t1) * (w1 + w2) / 2 for (t1, w1), (t2, w2) in zip(points, points[1:]))

    def peak(self, a, b):
        """The highest power reading from the one before a to the one after b, or None."""
        rows = self._bracket(a, b, 1)
        return max(w for _, w, _ in rows) if rows else None

    def record(self):
        """The round's `energy`: whether it was measured and why not, how, on which GPUs, and the idle draw."""
        if self.reason is not None:
            return {"status": "unmeasured", "reason": self.reason}
        return {"status": "measured", "source": self.source, "devices": list(self.devices),
                "interval_ms": round(self.interval * 1000) if self.interval else None,
                "idle_watts": (self.idle or {}).get("watts"), "idle_seconds": (self.idle or {}).get("seconds"),
                "idle_readings": (self.idle or {}).get("readings")}

    def case(self, samples, prompt_tokens):
        """A speed case's `energy` over its measured samples (each bracketed by metered), which loses their brackets."""
        windows = [(s, s.pop(WINDOW, None)) for s in samples]
        if self.reason is not None:
            return {"status": "unmeasured", "reason": self.reason}
        measured = [(s, w) for s, w in windows if not s.get("warmup")]
        if not measured:
            return {"status": "unmeasured", "reason": "no measured request"}
        if any(w is None for _, w in measured):
            return {"status": "unmeasured", "reason": "a resumed round's journal answered the case's requests"}
        self.read()   # a reading after the last request's end
        prefill = [self.joules(w[0], w[1]) for _, w in measured]
        decode = [self.joules(w[1], w[2]) for _, w in measured]
        peaks = [self.peak(w[0], w[2]) for _, w in measured]
        last = max(w[2] for _, w in measured)
        with self.lock:   # what later cases need is after this one
            self.readings = [r for r in self.readings if r[0] >= last] or self.readings[-1:]
        if None in prefill or None in decode:
            return {"status": "unmeasured", "reason": "the readings do not cover the case's requests"}
        prompts = sum(int((s.get("served") or {}).get("prompt_tokens") or prompt_tokens) for s, _ in measured)
        outputs = sum(max(0, int(s["output_tokens"]) - 1) for s, _ in measured)
        seconds = sum(w[2] - w[0] for _, w in measured)
        joules = sum(prefill) + sum(decode)
        return {"status": "measured", "requests": len(measured), "seconds": round(seconds, 4),
                "prefill_joules": round(sum(prefill), 4), "decode_joules": round(sum(decode), 4),
                "joules_per_prompt_token": _round(sum(prefill) / prompts) if prompts else None,
                "joules_per_output_token": _round(sum(decode) / outputs) if outputs else None,
                "average_watts": round(joules / seconds, 2) if seconds > 0 else None,
                "peak_watts": None if None in peaks else round(max(peaks), 2)}


def _at(rows, t, field=2):
    """`field` of the readings interpolated linearly at host time t (rows cover t)."""
    for (t1, *v1), (t2, *v2) in zip(rows, rows[1:]):
        if t1 <= t <= t2:
            return v1[field - 1] if t2 == t1 else v1[field - 1] + (v2[field - 1] - v1[field - 1]) * (t - t1) / (t2 - t1)
    return rows[0][field]


def _round(value):
    """Four significant figures: joules per token range from millijoules (a prompt token) to joules."""
    return float(f"{value:.4g}") if value else value


def active():
    return _active


def measure_idle():
    """The active meter's idle draw, before the suite's first request (once a round)."""
    if _active is not None:
        _active.measure_idle()


def metered(call, *args, **kwargs):
    """A timed request (crit_eval.run.timing's timed_*), its sample bracketed on the host's clock for the active meter:
    the call, its first token and its last, from the sample's own times. A request a resumed round's journal answered
    has no bracket."""
    if _active is None or _active.reason is not None:
        return call(*args, **kwargs)
    from .journal import active as journal
    written = journal()
    replays = written.replays if written is not None else 0
    started = _active.clock()
    out = call(*args, **kwargs)
    sample = out[1]
    if (written.replays if written is not None else 0) == replays:
        sample[WINDOW] = (started, started + sample["ttft_ms"] / 1000, started + sample["request_ms"] / 1000)
    return out


def case(samples, prompt_tokens):
    """The active meter's figures for a speed case (Meter.case), or None in a run without one; brackets are taken off
    the samples either way."""
    if _active is None:
        for sample in samples:
            sample.pop(WINDOW, None)
        return None
    return _active.case(samples, prompt_tokens)


def summary(cases, prefill_cases):
    """A speed suite's `energy` over its cases: the geometric mean of joules per output token over every measured case
    and of joules per prompt token over the prefill cases (protocol speed.prefill_cases), each None unless all of them
    have one. None when no case carries energy (a run without a meter)."""
    rows = {cid: c.get("energy") for cid, c in cases.items() if c.get("status") == "measured" and "energy" in c}
    if not rows:
        return None
    measured = {cid: r for cid, r in rows.items() if r and r.get("status") == "measured"}
    out = [r.get("joules_per_output_token") for r in measured.values()]
    prompt = [(measured.get(cid) or {}).get("joules_per_prompt_token") for cid in prefill_cases if cid in rows]
    peaks = [r["peak_watts"] for r in measured.values() if r.get("peak_watts") is not None]
    return {"joules_per_output_token": _round(metrics.geometric_mean(out))
            if out and len(measured) == len(rows) and all(out) else None,
            "joules_per_prompt_token": _round(metrics.geometric_mean(prompt)) if prompt and all(prompt) else None,
            "peak_watts": max(peaks) if peaks else None,
            "unmeasured": {cid: r.get("reason") for cid, r in rows.items() if not r or r.get("status") != "measured"}}


def joules_text(row):
    """A case's energy for its log line, '' without one."""
    if not row or row.get("status") != "measured" or row.get("joules_per_output_token") is None:
        return ""
    return f", {row['joules_per_output_token']:.3g} J/output token at {row['average_watts'] or 0:.0f} W"

