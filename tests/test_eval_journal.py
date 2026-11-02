"""An Eval round's journal (crit_eval.run.journal): requests recorded as they complete, answered again at their
recorded times when a round resumes, engines loaded only for the requests left (fake engine, no GPU)."""
from pathlib import Path
import json
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from crit_eval.run import journal  # noqa: E402
from crit_eval.run.timing import timed_generate  # noqa: E402  (timed on the round's clock)

IDENTITY = {"protocol": "test", "data": "d", "adapter_sha256": "a"}


class Engine:
    """Streams prompt[-1] + 1, + 2, ... one token every 2 ms; score returns the prompt's length."""
    opened = 0

    def __init__(self):
        Engine.opened += 1
        self.calls = 0
        self.closed = False

    def info(self):
        return {"name": "fake", "configuration": {}, "max_context": 4096}

    def generate(self, prompt, n, on_tokens):
        self.calls += 1
        tokens = []
        for i in range(n):
            time.sleep(0.002)
            tokens.append(prompt[-1] + 1 + i)
            on_tokens([tokens[-1]])
        return {"tokens": tokens}

    def score(self, prompt, continuation, top_k):
        self.calls += 1
        return {"rows": [[len(prompt), top_k]] * len(continuation)}

    def close(self):
        self.closed = True


def opener(context):
    engine = Engine()
    return engine, engine.info()


def round_trip(path, requests, resume=None, stop_after=None, end=None, mark_every=None):
    """Open one engine, mark a part per `mark_every` requests, run `requests` [(prompt, n)]: the samples, the
    engine and the journal. stop_after: raise KeyboardInterrupt before that request."""
    samples = []
    with journal.Journal(path, IDENTITY, resume=resume, log=lambda m: None) as written:
        engine, info = written.open(opener, {"context_tokens": 4096})
        try:
            for i, (prompt, n) in enumerate(requests):
                if mark_every and i % mark_every == 0:
                    written.mark(f"part-{i // mark_every}")
                if stop_after is not None and i == stop_after:
                    raise KeyboardInterrupt
                samples.append(timed_generate(engine, prompt, n))
                engine.score(prompt, [1, 2], 5)
        except KeyboardInterrupt:
            pass
        if end:
            written.end(end)
        engine.close()
    return samples, engine, written


def _cut(path, to):
    """A copy of the journal without its end line, as a killed round leaves it."""
    lines = path.read_text(encoding="utf-8").splitlines()
    to.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    return to


REQUESTS = [([1, 2, 3], 4), ([7], 6), ([10, 11], 3), ([20], 5)]


class JournalTests(unittest.TestCase):
    def setUp(self):
        Engine.opened = 0

    def test_a_finished_round_resumes_without_loading_its_engine_and_reads_the_same_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            first, _, _ = round_trip(Path(folder) / "a.journal.jsonl", REQUESTS)
            self.assertEqual(Engine.opened, 1)
            again, engine, written = round_trip(Path(folder) / "b.journal.jsonl", REQUESTS,
                                                resume=Path(folder) / "a.journal.jsonl")
        self.assertEqual(Engine.opened, 1)          # every request answered from the journal
        self.assertIsNone(engine.real)
        self.assertEqual([tokens for tokens, _ in again], [tokens for tokens, _ in first])
        self.assertEqual([sample for _, sample in again], [sample for _, sample in first])   # the clock's readings
        self.assertEqual(written.counts["replayed_calls"], 8)
        self.assertEqual(written.counts["live_calls"], 0)
        self.assertEqual(written.summary()["resumed"]["outcome"], "completed")

    def test_an_interrupted_round_loads_its_engine_for_the_requests_left_only(self):
        with tempfile.TemporaryDirectory() as folder:
            round_trip(Path(folder) / "a.journal.jsonl", REQUESTS, stop_after=2)
            lines = journal.read(Path(folder) / "a.journal.jsonl")
            self.assertEqual(lines[-1]["outcome"], "completed")   # the loop caught it; the block ended normally
            again, engine, written = round_trip(Path(folder) / "b.journal.jsonl", REQUESTS,
                                                resume=Path(folder) / "a.journal.jsonl")
            resumed = journal.read(Path(folder) / "b.journal.jsonl")
            self.assertEqual(Engine.opened, 2)
            self.assertEqual(engine.real.calls, 4)      # requests 3 and 4, a generate and a score each
            self.assertEqual(written.counts["replayed_calls"], 4)
            self.assertEqual([e["event"] for e in resumed].count("load"), 1)
            self.assertEqual(len(again), 4)
            # The resumed journal holds every request, so it can be resumed in turn without any engine.
            third, engine, written = round_trip(Path(folder) / "c.journal.jsonl", REQUESTS,
                                                resume=Path(folder) / "b.journal.jsonl")
        self.assertEqual(Engine.opened, 2)
        self.assertIsNone(engine.real)
        self.assertEqual(written.counts["replayed_calls"], 8)
        self.assertEqual([tokens for tokens, _ in third], [tokens for tokens, _ in again])

    def test_a_journal_cut_by_a_crash_resumes_from_its_last_whole_line(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "a.journal.jsonl"
            round_trip(path, REQUESTS)
            lines = path.read_text(encoding="utf-8").splitlines()
            # No end line, and the last request cut in the middle of its line.
            path.write_text("\n".join(lines[:-2]) + "\n" + lines[-2][:40], encoding="utf-8")
            _, engine, written = round_trip(Path(folder) / "b.journal.jsonl", REQUESTS, resume=path)
        self.assertEqual(written.resumed["outcome"], "unfinished")
        self.assertEqual(written.counts["replayed_calls"], 7)
        self.assertEqual(engine.real.calls, 1)

    def test_another_request_than_the_journal_holds_measures_live_from_there(self):
        with tempfile.TemporaryDirectory() as folder:
            round_trip(Path(folder) / "a.journal.jsonl", REQUESTS)
            changed = REQUESTS[:1] + [([8], 6)] + REQUESTS[2:]
            _, engine, written = round_trip(Path(folder) / "b.journal.jsonl", changed,
                                            resume=Path(folder) / "a.journal.jsonl")
        self.assertEqual(written.counts["replayed_calls"], 2)
        self.assertEqual(engine.real.calls, 6)
        self.assertEqual(written.diverged["after_replayed_calls"], 2)

    def test_a_disturbed_round_measures_the_part_it_stopped_in_again(self):
        with tempfile.TemporaryDirectory() as folder:
            round_trip(Path(folder) / "a.journal.jsonl", REQUESTS, end="disturbed", mark_every=2)
            _, engine, written = round_trip(Path(folder) / "b.journal.jsonl", REQUESTS,
                                            resume=Path(folder) / "a.journal.jsonl", mark_every=2)
        self.assertEqual(written.counts["replayed_calls"], 4)    # part 0's two requests, each with its score
        self.assertEqual(engine.real.calls, 4)

    def test_a_journal_of_another_round_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            round_trip(Path(folder) / "a.journal.jsonl", REQUESTS)
            with self.assertRaises(SystemExit) as refused:
                journal.Journal(Path(folder) / "b.journal.jsonl", {**IDENTITY, "data": "other"},
                                resume=Path(folder) / "a.journal.jsonl")
        self.assertIn("data", str(refused.exception))

    def test_a_result_json_cannot_keep_is_measured_again_on_resume(self):
        class Odd(Engine):
            def score(self, prompt, continuation, top_k):
                self.calls += 1
                return {"rows": object()} if prompt == [10, 11] else super().score(prompt, continuation, top_k)

        def odd(context):
            engine = Odd()
            return engine, engine.info()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "a.journal.jsonl"
            with journal.Journal(path, IDENTITY, log=lambda m: None) as written:
                engine, _ = written.open(odd, {})
                for prompt, n in REQUESTS:
                    engine.generate(prompt, n, lambda tokens: None)
                    engine.score(prompt, [1], 5)
            self.assertEqual(written.counts["unrecorded_calls"], 1)
            kept = journal.replayable(journal.read(path))
        self.assertEqual(sum(row["event"] == "call" for row in kept), 5)

    def test_a_request_that_raised_raises_again_on_replay_and_replay_goes_on(self):
        class Refusing(Engine):
            def loads(self):
                self.calls += 1
                raise ConnectionRefusedError("no /health")

        def refusing(context):
            engine = Refusing()
            return engine, engine.info()

        def measure(path, resume=None):
            raised = []
            with journal.Journal(path, IDENTITY, resume=resume, log=lambda m: None) as written:
                engine, _ = written.open(refusing, {})
                try:
                    engine.loads()
                except OSError as error:
                    raised.append((type(error), str(error)))
                for prompt, n in REQUESTS:
                    engine.generate(prompt, n, lambda tokens: None)
            return raised, engine, written
        with tempfile.TemporaryDirectory() as folder:
            first, _, _ = measure(Path(folder) / "a.journal.jsonl")
            again, engine, written = measure(Path(folder) / "b.journal.jsonl", resume=Path(folder) / "a.journal.jsonl")
        self.assertEqual(first, [(ConnectionRefusedError, "no /health")])
        self.assertEqual(again, first)
        self.assertIsNone(engine.real)
        self.assertIsNone(written.diverged)
        self.assertEqual(written.counts["replayed_calls"], 5)

    def test_a_served_engine_the_journal_answers_keeps_the_deadline_it_observed(self):
        import random
        from crit_eval import EVAL_DIR, load_json
        from crit_eval.run.deadline import RequestDeadline
        rule = load_json(EVAL_DIR / "protocol.json")["requests"]["timeout"]

        class Served(Engine):
            def __init__(self):
                super().__init__()
                self.deadline = RequestDeadline(rule)

            def info(self):
                return {**super().info(), "served": True}

            def generate(self, prompt, n, on_tokens):
                out = super().generate(prompt, n, on_tokens)
                # The engine times a request its own way, never quite as the token arrivals do.
                self.deadline.observe(len(prompt), random.uniform(0.01, 0.02), n, random.uniform(0.01, 0.02))
                return out

        def served(context):
            engine = Served()
            return engine, engine.info()

        def measure(path, resume=None):
            with journal.Journal(path, IDENTITY, resume=resume, log=lambda m: None) as written:
                engine, _ = written.open(served, {})
                for prompt, n in REQUESTS:
                    engine.generate(prompt, n, lambda tokens: None)
                return engine, vars(engine.deadline)
        with tempfile.TemporaryDirectory() as folder:
            first, observed = measure(Path(folder) / "a.journal.jsonl")
            again, replayed = measure(Path(folder) / "b.journal.jsonl", resume=Path(folder) / "a.journal.jsonl")
        self.assertIsNone(again.real)
        self.assertEqual(replayed, observed)

    def test_a_round_interrupted_in_its_curve_resumes_with_the_speed_it_measured(self):
        from unittest.mock import patch
        from crit_eval.run import runner
        from test_eval import FACTS, MOCK, fixture
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)
            fixture(data)
            options = dict(config={"facts_from": FACTS}, log=lambda m: None, context_tokens=65536)

            def interrupted(*args, **kwargs):
                raise KeyboardInterrupt
            with patch.object(runner, "run_context", interrupted), self.assertRaises(KeyboardInterrupt):
                runner.run(MOCK, "unused-model", data, data / "first.json", **options)
            first = json.loads((data / "first.json").read_text())
            self.assertEqual(first["aborted"]["reason"], "interrupted")
            self.assertEqual(first["journal"]["path"], "first.journal.jsonl")
            self.assertEqual(journal.read(data / "first.journal.jsonl")[-1]["outcome"], "interrupted")
            resumed = runner.run(MOCK, "unused-model", data, data / "second.json",
                                 resume=data / "first.journal.jsonl", **options)
            described = journal.describe(data / "second.journal.jsonl")
            self.assertIn("resumes " + str(data / "first.journal.jsonl") + " (interrupted", described)
            self.assertIn("ended: completed", described)
            self.assertIn("no end", journal.describe(_cut(data / "second.journal.jsonl", data / "cut.journal.jsonl")))
            with self.assertRaises(SystemExit):   # another round's options: its journal answers none of this one
                runner.run(MOCK, "unused-model", data, data / "third.json", resume=data / "first.journal.jsonl",
                           **{**options, "context_tokens": 32768})
        self.assertNotIn("aborted", resumed)
        self.assertEqual(resumed["journal"]["resumed"]["path"], "first.journal.jsonl")
        self.assertEqual(resumed["journal"]["resumed"]["outcome"], "interrupted")
        self.assertIsNone(resumed["journal"]["diverged"])
        self.assertGreater(resumed["journal"]["replayed_calls"], 0)
        # The speed cases are the interrupted round's measurements, read back to the bit: the round's clock gives
        # back the readings it took.
        # Host load is unmeasured over requests the journal answered.
        def measured(cases):
            return {k: {key: v for key, v in case.items() if key != "host_other_cpu_cores"} for k, case in cases.items()}
        self.assertEqual(measured(resumed["speed"]["cases"]), measured(first["speed"]["cases"]))
        self.assertEqual({c["host_other_cpu_cores"] for c in resumed["speed"]["cases"].values()}, {None})
        self.assertTrue(resumed["context"]["points"])
        self.assertEqual(resumed["stages"]["reached"], first["stages"]["reached"])

    def test_path_for_a_result(self):
        self.assertEqual(journal.path_for("eval/attempt-0003.json").name, "attempt-0003.journal.jsonl")


if __name__ == "__main__":
    unittest.main()
