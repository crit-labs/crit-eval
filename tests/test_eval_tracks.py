"""Closed and open tracks (protocol tracks): what Eval checks of the closed track on a result, the card declaring the
rest."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from crit_eval.reports import status  # noqa: E402
from crit_eval.run import chat, runner, tracks  # noqa: E402
from test_eval import MOCK, fixture, run_full  # noqa: E402

PRESERVE = {"requirements": {"quality_mode": "preserve"}}
GB = 10**9


class ClosedTrackTests(unittest.TestCase):
    def state(self, result, info, own=2 * GB):
        found = tracks.closed(result, info, own, 0.05)
        return found["qualifies"], {name: row["state"] for name, row in found["checks"].items()}

    def test_a_plain_engine_in_preserve_loaded_once_qualifies(self):
        self.assertEqual(self.state(PRESERVE, {}), (True, {"quality_mode": "pass", "one_load": "pass",
                                                           "own_drafter": "pass"}))
        self.assertEqual(self.state({"requirements": {"quality_mode": "as_quantized"}}, {})[0], False)
        served = {**PRESERVE, "engine_loads": {"first": 1, "last": 1}}
        self.assertEqual(self.state(served, {"served": True})[0], True)
        reloaded = {**PRESERVE, "engine_loads": {"first": 1, "last": 3}}
        self.assertEqual(self.state(reloaded, {"served": True}), (False, {"quality_mode": "pass", "one_load": "fail",
                                                                          "own_drafter": "pass"}))

    def test_a_drafter_reads_no_more_than_the_checkpoints_own_draft_layers(self):
        drafting = {"speculative": True, "draft_bytes_per_step": 2 * GB}
        self.assertEqual(self.state(PRESERVE, drafting)[0], True)
        self.assertEqual(self.state(PRESERVE, {**drafting, "draft_bytes_per_step": GB})[0], True, "a re-encoded head")
        self.assertEqual(self.state(PRESERVE, {**drafting, "draft_bytes_per_step": int(2.2 * GB)})[0], False)
        self.assertEqual(self.state(PRESERVE, drafting, own=None)[0], False, "no draft layers of its own")
        self.assertEqual(self.state(PRESERVE, {"speculative": True})[0], None, "no declared bytes to compare")

    def test_a_served_engine_reports_its_load_count(self):
        engine = chat.ChatEngine.__new__(chat.ChatEngine)
        engine.deadline = SimpleNamespace(seconds=lambda _: 30)
        engine.server = SimpleNamespace(get=lambda path, timeout: {"engine": {"engine_loads": 2}})
        self.assertEqual(tracks.engine_loads(engine), 2)
        engine.server = SimpleNamespace(get=lambda path, timeout: {"status": "ok"})
        self.assertIsNone(tracks.engine_loads(engine))

        def unreachable(path, timeout):
            raise OSError("no /health")
        engine.server = SimpleNamespace(get=unreachable)
        self.assertIsNone(tracks.engine_loads(engine))
        self.assertIsNone(tracks.engine_loads(object()))

    def test_a_run_records_the_closed_tracks_checks_and_status_names_them(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            fixture(data)
            result = run_full(MOCK, "unused-model", data, data / "result.json", {"max_context": 128}, parts=("speed",),
                              warmups=0, repetitions=1, log=lambda message: None)
        closed = result["track"]["closed"]
        self.assertEqual(set(closed["checks"]), set(tracks.CHECKS))
        self.assertEqual(closed["checks"]["quality_mode"]["state"], "pass")
        self.assertEqual(closed["checks"]["one_load"]["state"], "pass")
        protocol = runner.load_json(runner.EVAL_DIR / "protocol.json")
        text = status.render(status.build(result, protocol, None))
        self.assertIn("track    closed track", text)


if __name__ == "__main__":
    unittest.main()
