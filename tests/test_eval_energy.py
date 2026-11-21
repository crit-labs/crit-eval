"""Energy per token in the speed cases (crit_eval.run.energy; reported, not judged), on a mocked NVML: no GPU, and the
host's own NVML is never asked."""
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from crit_eval.reports import status  # noqa: E402
from crit_eval.run import energy, runner  # noqa: E402
from test_eval import MOCK, fixture  # noqa: E402


class FakeNvml:
    """NVML's calls on fake GPUs: each draws `watts(t)` at host time t, its counter the integral (a constant draw is
    a straight line); power or counter None where its driver lacks them."""

    def __init__(self, watts=lambda t: 300.0, counter=True, power=True, devices=(0,), clock=time.perf_counter):
        self.watts, self.counter, self.power, self.devices, self.clock = watts, counter, power, devices, clock
        self.joules = {}
        self.closed = False

    def handle(self, key):
        if key not in self.devices:
            raise energy.Unavailable(f"NVML has no GPU {key} (error 13)")
        return key

    def power_watts(self, handle):
        return self.watts(self.clock()) if self.power else None

    def energy_joules(self, handle):
        if not self.counter:
            return None
        return self.watts(0.0) * self.clock()   # a constant draw since time 0

    def close(self):
        self.closed = True


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def bracket(clock, meter, prompt_seconds, decode_seconds, step=0.01):
    """One request on the fake clock: readings every `step` through its prefill and decode, its bracket returned."""
    start = clock.now
    for seconds in (prompt_seconds, decode_seconds):
        end = clock.now + seconds
        while clock.now + step < end:
            clock.now += step
            meter.read()
        clock.now = end
    return start, start + prompt_seconds, start + prompt_seconds + decode_seconds


class MeterTests(unittest.TestCase):
    def meter(self, nvml, clock, **options):
        return energy.Meter([0], nvml=nvml, interval=0, idle_seconds=2.0, clock=clock, sleep=clock.sleep,
                            **options).start()

    def tearDown(self):
        if energy.active() is not None:
            energy.active().stop()

    def test_the_energy_counter_gives_joules_per_prompt_and_output_token(self):
        clock = Clock()
        meter = self.meter(FakeNvml(watts=lambda t: 400.0, clock=clock), clock)
        meter.measure_idle()
        self.assertEqual(meter.idle["watts"], 400.0)
        samples = []
        for warmup in (True, False, False):
            window = bracket(clock, meter, 0.5, 2.0)
            samples.append({"warmup": warmup, "output_tokens": 101, energy.WINDOW: window})
        row = meter.case(samples, prompt_tokens=1000)
        self.assertEqual(row["status"], "measured")
        self.assertEqual(row["requests"], 2)   # the warmup's draw is not the case's
        self.assertAlmostEqual(row["prefill_joules"], 2 * 0.5 * 400)
        self.assertAlmostEqual(row["joules_per_prompt_token"], 0.2)
        # The decode's tokens after the first, as decode tok/s counts them: 2 s at 400 W over 100 tokens.
        self.assertAlmostEqual(row["joules_per_output_token"], 8.0)
        self.assertAlmostEqual(row["average_watts"], 400.0)
        self.assertEqual(row["peak_watts"], 400.0)
        self.assertFalse(any(energy.WINDOW in s for s in samples))   # brackets never reach the result
        self.assertEqual(meter.record()["source"], "energy_counter")
        self.assertEqual(meter.record()["idle_watts"], 400.0)

    def test_without_a_counter_the_power_readings_are_integrated(self):
        clock = Clock()
        # 100 W at rest, 500 W while the device works (from t=12.5 on, after the idle reading at 12.0).
        nvml = FakeNvml(watts=lambda t: 500.0 if t >= 12.5 else 100.0, counter=False, clock=clock)
        meter = self.meter(nvml, clock)
        meter.measure_idle()
        self.assertEqual(meter.record()["source"], "power")
        self.assertAlmostEqual(meter.idle["watts"], 100.0)
        clock.now = 12.5
        meter.read()
        window = bracket(clock, meter, 1.0, 1.0)
        row = meter.case([{"warmup": False, "output_tokens": 11, energy.WINDOW: window}], 50)
        self.assertAlmostEqual(row["joules_per_output_token"], 50.0, delta=1e-6)   # 1 s at 500 W over 10 tokens
        self.assertAlmostEqual(row["joules_per_prompt_token"], 10.0, delta=1e-6)
        self.assertEqual(row["peak_watts"], 500.0)

    def test_a_stall_between_a_readings_two_figures_leaves_the_counter_its_own_time(self):
        # The host busy elsewhere for 20 ms during the idle window's closing reading, between its power call and its
        # counter call: the counter's joules keep the time of their own call, so the draw at rest reads as it is
        # (dated by both calls together it read 251.24 W).
        clock = Clock()

        class Stalling(FakeNvml):
            stall = 0.0

            def power_watts(self, handle):
                clock.now += self.stall
                self.stall = 0.0
                return super().power_watts(handle)
        nvml = Stalling(watts=lambda t: 250.0, clock=clock)
        meter = self.meter(nvml, clock)
        nvml.stall = 0.02
        meter.measure_idle()
        self.assertAlmostEqual(meter.idle["watts"], 250.0, places=6)

    def test_a_served_reply_counts_the_servers_prompt_tokens(self):
        clock = Clock()
        meter = self.meter(FakeNvml(watts=lambda t: 200.0, clock=clock), clock)
        window = bracket(clock, meter, 1.0, 1.0)
        row = meter.case([{"warmup": False, "output_tokens": 3, "served": {"prompt_tokens": 400},
                           energy.WINDOW: window}], 100)
        self.assertAlmostEqual(row["joules_per_prompt_token"], 0.5)

    def test_without_nvml_everything_is_unmeasured_and_says_why(self):
        with patch.object(energy.Nvml, "load", side_effect=energy.Unavailable("no NVML library on this host")):
            meter = energy.Meter([0], interval=0).start()
        meter.measure_idle()   # nothing to wait for
        self.assertIsNone(meter.idle)
        self.assertEqual(meter.record(), {"status": "unmeasured", "reason": "no NVML library on this host"})
        sample = {"warmup": False, "output_tokens": 3, energy.WINDOW: (0, 1, 2)}
        self.assertEqual(meter.case([sample], 10)["status"], "unmeasured")
        self.assertNotIn(energy.WINDOW, sample)
        meter.stop()
        self.assertIsNone(energy.active())

    def test_hidden_gpus_and_drivers_without_figures_leave_it_unmeasured(self):
        nvml = FakeNvml()
        hidden = energy.Meter(nvml=nvml, interval=0, environ={"CUDA_VISIBLE_DEVICES": ""}).start()
        self.assertEqual(hidden.record()["reason"], "CUDA_VISIBLE_DEVICES hides every GPU")
        hidden.stop()
        blind = energy.Meter([0], nvml=FakeNvml(counter=False, power=False), interval=0).start()
        self.assertEqual(blind.record()["reason"], "the driver reports neither power nor energy for GPU 0")
        blind.stop()
        missing = energy.Meter([3], nvml=FakeNvml(), interval=0).start()
        self.assertEqual(missing.record()["reason"], "NVML has no GPU 3 (error 13)")
        missing.stop()

    def test_the_engines_gpus(self):
        self.assertEqual(energy.engine_devices([0, 1], {}), [0, 1])   # a device set
        self.assertEqual(energy.engine_devices(None, {}), [0])
        self.assertEqual(energy.engine_devices(None, {"CUDA_VISIBLE_DEVICES": "2,0"}), [2])
        self.assertEqual(energy.engine_devices(None, {"CUDA_VISIBLE_DEVICES": "GPU-ab12"}), ["GPU-ab12"])
        with self.assertRaises(energy.Unavailable):
            energy.engine_devices(None, {"CUDA_VISIBLE_DEVICES": "-1"})

    def test_requests_a_resumed_journal_answered_are_unmeasured(self):
        clock = Clock()
        meter = self.meter(FakeNvml(clock=clock), clock)
        row = meter.case([{"warmup": False, "output_tokens": 3}], 10)   # no bracket: the journal answered it
        self.assertEqual(row, {"status": "unmeasured",
                               "reason": "a resumed round's journal answered the case's requests"})

    def test_several_gpus_add_up(self):
        clock = Clock()
        meter = energy.Meter([0, 1], nvml=FakeNvml(watts=lambda t: 150.0, devices=(0, 1), clock=clock),
                             interval=0, clock=clock, sleep=clock.sleep).start()
        window = bracket(clock, meter, 1.0, 1.0)
        row = meter.case([{"warmup": False, "output_tokens": 2, energy.WINDOW: window}], 300)
        self.assertAlmostEqual(row["average_watts"], 300.0)
        self.assertEqual(row["peak_watts"], 300.0)

    def test_a_suites_summary(self):
        cases = {"a": {"status": "measured", "energy": {"status": "measured", "joules_per_output_token": 2.0,
                                                         "joules_per_prompt_token": 0.01, "peak_watts": 400.0}},
                 "b": {"status": "measured", "energy": {"status": "measured", "joules_per_output_token": 8.0,
                                                         "joules_per_prompt_token": 0.04, "peak_watts": 450.0}},
                 "c": {"status": "skipped"}}
        found = energy.summary(cases, ["a", "b"])
        self.assertEqual((found["joules_per_output_token"], found["joules_per_prompt_token"], found["peak_watts"]),
                         (4.0, 0.02, 450.0))
        cases["b"]["energy"] = {"status": "unmeasured", "reason": "why"}
        found = energy.summary(cases, ["a", "b"])
        self.assertIsNone(found["joules_per_output_token"])
        self.assertEqual(found["unmeasured"], {"b": "why"})
        self.assertIsNone(energy.summary({"a": {"status": "measured"}}, []))   # a run without a meter


class RoundTests(unittest.TestCase):
    """A whole round of the mock engine, its GPU drawing a constant 250 W on a mocked NVML."""

    def run_round(self, directory, environ):
        data = Path(directory)
        fixture(data)
        with patch.object(energy.Nvml, "load", return_value=FakeNvml(watts=lambda t: 250.0)) as load, \
                patch.object(energy, "IDLE_SECONDS", 0.05), patch.object(energy, "INTERVAL_SECONDS", 0.005), \
                patch.dict(energy.os.environ, environ):
            result = runner.run(MOCK, "unused-model", data, data / "result.json", {"max_context": 64},
                                stage="full", warmups=0, repetitions=2, log=lambda message: None,
                                parts=("speed",))
        return result, load

    def test_the_eval_file_records_joules_per_token_per_case_and_the_idle_draw(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _ = self.run_round(directory, {"CUDA_VISIBLE_DEVICES": "0"})
            stored = json.loads((Path(directory) / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(stored["energy"]["status"], "measured")
        self.assertEqual(stored["energy"]["devices"], [0])
        self.assertAlmostEqual(stored["energy"]["idle_watts"], 250.0, delta=0.5)
        self.assertIsNone(energy.active())   # the side thread ends with the engine
        case = stored["speed"]["cases"]["short"]
        row = case["energy"]
        self.assertEqual(row["status"], "measured")
        self.assertAlmostEqual(row["average_watts"], 250.0, delta=0.5)
        # The counter is a straight line: the decode's joules per token are its watts over its tok/s.
        measured = [s for s in case["samples"] if not s["warmup"]]
        decode_seconds = sum((s["request_ms"] - s["ttft_ms"]) / 1000 for s in measured)
        self.assertAlmostEqual(row["joules_per_output_token"],
                               250.0 * decode_seconds / sum(s["output_tokens"] - 1 for s in measured),
                               delta=0.01 * row["joules_per_output_token"])
        self.assertFalse(any(energy.WINDOW in s for s in case["samples"]))
        self.assertEqual(stored["speed"]["energy"]["joules_per_output_token"], row["joules_per_output_token"])
        self.assertEqual(stored["status"]["kpis"]["energy"]["status"], "measured")
        self.assertIn("  energy   sampled ", status.render(stored["status"]))
        # Reported, not judged: no gate reads it.
        self.assertFalse(any("energy" in g["name"] for g in stored["status"]["gates"]))

    def test_a_round_whose_gpus_are_hidden_records_why_and_asks_no_nvml(self):
        with tempfile.TemporaryDirectory() as directory:
            result, load = self.run_round(directory, {"CUDA_VISIBLE_DEVICES": ""})
        load.assert_not_called()
        self.assertEqual(result["energy"], {"status": "unmeasured", "reason": "CUDA_VISIBLE_DEVICES hides every GPU"})
        self.assertEqual(result["speed"]["cases"]["short"]["energy"]["status"], "unmeasured")
        self.assertIn("energy   unmeasured: CUDA_VISIBLE_DEVICES hides every GPU",
                      status.render(result["status"]))


if __name__ == "__main__":
    unittest.main()
