"""One request per case (protocol requests.cases): every request a round builds is its case's definition with only
the overrides its part declares, a part cannot change a field it does not declare, and the parts whose requests
crit-eval-v0.9.0 did not change send exactly what crit-eval-v0.8.3 sent."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("", "tests"):
    sys.path.insert(0, str(ROOT / _folder))
from crit_eval.quality import prepare  # noqa: E402
from crit_eval.run import requests  # noqa: E402
import requests_stream  # noqa: E402

PROTOCOL = json.loads((ROOT / "crit_eval/protocol.json").read_text(encoding="utf-8"))
PINNED = json.loads((ROOT / "tests/fixtures/requests_crit-eval-v0.8.3.json").read_text(encoding="utf-8"))


def cases_of(prepared, protocol=PROTOCOL):
    """Every case of the prepared data by its key, and a lookup that builds a curve or retrieval point's case."""
    found = {requests.case_key(c): c for kind in ("speed", "heldout", "followup") for c in prepared.get(kind) or []}

    def case(key):
        if key.startswith("curve@"):
            return requests.curve_case(prepared["retrieval"], protocol["context"], int(key.split("@")[1]))
        if key.startswith("retrieval@"):
            return requests.retrieval_case(prepared["retrieval"], protocol["context"]["retrieval"],
                                           int(key.split("@")[1]))
        return found[key]
    return case


def check(test, seen, prepared, protocol=PROTOCOL):
    """Each request is its case's definition but for the fields it names as overrides, each one its part declares
    and each different from the definition; returns the parts seen."""
    case = cases_of(prepared, protocol)
    declared = protocol["requests"]["overrides"]
    for request in seen:
        with test.subTest(part=request["part"], case=request["case"]):
            definition = requests.definition(case(request["case"]))
            test.assertIn(request["part"], declared)
            for field in requests.FIELDS:
                if field in request["overrides"]:
                    test.assertIn(field, declared[request["part"]], f"{field} undeclared")
                    test.assertNotEqual(request[field], definition[field])
                else:
                    test.assertEqual(request[field], definition[field], field)
            # A sampling override is the prepared sampling at a seed; a prompt override never lengthens the prompt.
            if "sampling" in request["overrides"]:
                test.assertEqual(request["sampling"],
                                 prepare.sampled_request(prepared["sampling"], request["sampling"]["seed"]))
            if "prompt_ids" in request["overrides"] and request["part"] != "kernels":
                test.assertLessEqual(len(request["prompt_ids"]), len(definition["prompt_ids"]))
    return {request["part"] for request in seen}


class CaseRequestTests(unittest.TestCase):
    CASE = {"id": "chat", "prompt_ids": [1, 2, 3], "output_tokens": 8, "messages": [], "tools": None}

    def test_a_case_request_is_its_definition_with_the_parts_declared_overrides(self):
        plain = requests.case_request(self.CASE, "speed", PROTOCOL)
        self.assertEqual({k: plain[k] for k in requests.FIELDS}, {k: requests.definition(self.CASE)[k]
                                                                  for k in requests.FIELDS})
        self.assertEqual((plain["case"], plain["part"], plain["overrides"]), ("chat", "speed", []))
        drawn = requests.case_request(self.CASE, "sampled_speed", PROTOCOL,
                                      sampling=prepare.sampled_request({"temperature": 1.0}, 1))
        self.assertEqual(drawn["overrides"], ["sampling"])
        # A field at its definition's value is no override, and no difference to compare across.
        self.assertEqual(requests.case_request(self.CASE, "history", PROTOCOL, output_tokens=8)["overrides"], [])
        self.assertTrue(requests.same_request(plain, requests.case_request(self.CASE, "concurrency", PROTOCOL)))
        self.assertFalse(requests.same_request(plain, drawn))
        self.assertEqual(requests.request_sha256(plain),
                         requests.request_sha256(requests.case_request(self.CASE, "concurrency", PROTOCOL)))

    def test_a_part_cannot_change_a_field_it_does_not_declare(self):
        # The concurrency part asks each case its own length: a level of another one is refused, not built.
        with self.assertRaisesRegex(ValueError, "does not declare"):
            requests.case_request(self.CASE, "concurrency", PROTOCOL, output_tokens=128)
        with self.assertRaisesRegex(ValueError, "does not declare"):
            requests.case_request(self.CASE, "history", PROTOCOL, prompt_cache=True)
        with self.assertRaisesRegex(ValueError, "not one protocol"):
            requests.case_request(self.CASE, "somewhere", PROTOCOL)
        with self.assertRaisesRegex(ValueError, "no request field"):
            requests.case_request(self.CASE, "speed", PROTOCOL, stop=[1])
        # Data prepared before the one definition holds no follow-up turn's length.
        with self.assertRaises(SystemExit):
            requests.case_request({"turn": 1, "prompt_ids": [1]}, "followup", PROTOCOL)

    def test_every_part_declared_in_the_protocol_names_only_request_fields(self):
        for part, fields in PROTOCOL["requests"]["overrides"].items():
            self.assertLessEqual(set(fields), set(requests.FIELDS) - {"case"}, part)


class RoundRequestTests(unittest.TestCase):
    def test_every_request_of_a_mock_round_is_its_cases_definition_and_its_parts_overrides(self):
        with requests.observe() as seen:
            requests_stream.stream()
        parts = check(self, seen, requests_stream.fixture())
        # The sampling check takes the sampled suite's warmup replies, asked with its log-probabilities.
        self.assertLessEqual({"speed", "sampled_speed", "heldout", "exactness", "history", "followup", "context",
                              "retrieval"}, parts)

    def test_the_parts_that_ask_a_case_outside_a_round_build_its_request_too(self):
        # The sampling check's own replies (when the sampled suite's warmups carry none) and the pre-check's speed
        # samples and request history, on the mock engine.
        from crit_eval.run import adapters, history, precheck, speed
        prepared = requests_stream.fixture()
        engine, info = adapters.open_engine(requests_stream.Adapter([]), "unused-model", {}, True, 4096)
        with requests.observe() as seen:
            found = speed.run_sampling_check(engine, info, prepared, PROTOCOL, lambda m: None)
            precheck.speed_samples(engine, prepared["speed"], samples=1)
            history.history_outputs(engine, prepared["speed"], "precheck")
        self.assertEqual(found["status"], "measured")
        self.assertEqual(check(self, seen, prepared), {"sampling_check", "precheck"})

    def test_every_request_of_a_served_round_is_its_cases_definition_and_its_parts_overrides(self):
        from test_eval_server import write_data
        with requests.observe() as seen:
            requests_stream.served_stream()
        with tempfile.TemporaryDirectory() as directory:
            write_data(Path(directory))
            prepared = json.loads((Path(directory) / "prepared.json").read_text())
        parts = check(self, seen, prepared)
        self.assertLessEqual({"speed", "sampled_speed", "heldout", "history", "followup", "context"}, parts)

    def test_the_parts_whose_requests_did_not_change_send_what_v0_8_3_sent(self):
        # The distinct requests of a whole round, on an engine in process and behind a server: crit-eval 8882699's
        # (protocol crit-eval-v0.8.3) and this one's are the same, so every part but concurrency scores as it did.
        calls, _ = requests_stream.stream()
        found = sorted({json.dumps([c["engine"], c["method"], c["prompt_tokens"], c["output"], c["digest"]])
                        for c in calls})
        self.assertEqual([json.loads(c) for c in found], PINNED["calls"])
        bodies, _ = requests_stream.served_stream()
        self.assertEqual(sorted(set(bodies)), PINNED["served"])


if __name__ == "__main__":
    unittest.main()
