"""Prompt-length jitter (protocol jitter): the held-out prompts and the long-context curve's points below the
requirement at lengths a run's seed draws, recorded in the result, the speed suite and the requirement exact."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from crit_eval.quality import prepare  # noqa: E402
from crit_eval.run import chat, jitter, journal, long_context, runner  # noqa: E402
from test_eval import MOCK, fixture, run_full  # noqa: E402

RULE = {"fraction": [0.8, 1.0]}
HEAD, TAIL = [100, 101], [102, 103]


def held(pid, content):
    return {"id": pid, "language": pid, "kind": "prose", "prompt_ids": HEAD + content + TAIL, "content_ids": content,
            "messages": [{"role": "user", "content": pid}], "output_tokens": 8}


class DrawTests(unittest.TestCase):
    def test_a_seed_draws_each_length_from_its_own_name_within_the_fraction(self):
        seed = {"seed": 7, **RULE}
        draws = [jitter.drawn(7, "heldout", pid, 100, RULE) for pid in ("zh", "ja", "ko", "de", "es")]
        self.assertTrue(all(80 <= d <= 100 for d in draws), draws)
        self.assertGreater(len(set(draws)), 1, "prompts draw apart")
        self.assertEqual(draws, [jitter.drawn(7, "heldout", pid, 100, RULE) for pid in ("zh", "ja", "ko", "de", "es")])
        self.assertNotEqual(draws, [jitter.drawn(8, "heldout", pid, 100, RULE) for pid in ("zh", "ja", "ko", "de", "es")])
        self.assertEqual(jitter.drawn(7, "heldout", "zh", 100, RULE),
                         math.ceil(100 * jitter.fraction(7, "heldout", "zh", RULE)))
        self.assertEqual(jitter.drawn(7, "heldout", "zh", 3, RULE, least=3), 3)
        # The requirement is filled exactly; a point below it fills its drawn share; no jitter, the point.
        self.assertEqual(jitter.curve_point(65536, 65536, seed), 65536)
        below = jitter.curve_point(32768, 65536, seed)
        self.assertTrue(0.8 * 32768 <= below <= 32768 and below != 32768, below)
        self.assertEqual(jitter.curve_point(32768, 65536, None), 32768)

    def test_a_held_out_prompt_cuts_its_text_inside_the_template(self):
        prepared = {"heldout_frame": {"head_ids": HEAD, "tail_ids": TAIL, "messages": []}}
        prompt = held("zh", list(range(20)))
        for seed in range(40):
            ids, drawn = jitter.heldout_prompt(prepared, prompt, {"seed": seed, **RULE})
            self.assertEqual((ids[:2], ids[-2:]), (HEAD, TAIL))
            self.assertEqual(ids[2:-2], list(range(len(ids) - 4)))
            self.assertEqual(drawn, {"nominal_tokens": 24, "prompt_tokens": len(ids)})
            self.assertEqual(len(ids), max(5, math.ceil(24 * jitter.fraction(seed, "heldout", "zh", RULE))))
        self.assertEqual(jitter.heldout_prompt(prepared, prompt, None), (prompt["prompt_ids"], None))
        whole = {k: v for k, v in prompt.items() if k != "content_ids"}
        ids, drawn = jitter.heldout_prompt(prepared, whole, {"seed": 1, **RULE})
        self.assertEqual(ids, prompt["prompt_ids"])
        self.assertIn("measured whole", drawn["reason"])


class Text:
    """Characters as token ids, a template of <u> ... </u><a>: enough for prepare's held-out frame."""
    special_ids = ()

    def __init__(self):
        self.tokenizer = self

    def encode(self, text, add_special_tokens=False, split_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def render(self, messages, tools=None, add_generation_prompt=True, enable_thinking=False):
        return "".join(f"<u>{m['content']}</u>" for m in messages) + ("<a>" if add_generation_prompt else "")

    def encode_chat(self, messages, tools=None, add_generation_prompt=True, enable_thinking=False):
        return self.encode(self.render(messages, tools, add_generation_prompt))


class PreparedTests(unittest.TestCase):
    def test_prepare_freezes_the_frame_and_each_prompts_text_and_a_server_gets_the_cut_text(self):
        frame, rows = prepare.heldout_prompts(Text(), [{"id": "de", "language": "German", "kind": "prose",
                                                       "content": "Erkläre die Gezeiten ausführlich."}], 64)
        self.assertEqual(Text().decode(frame["head_ids"]), "<u>")
        self.assertEqual(Text().decode(frame["tail_ids"]), "</u><a>")
        row = rows[0]
        self.assertEqual(frame["head_ids"] + row["content_ids"] + frame["tail_ids"], row["prompt_ids"])
        prepared = {"speed": [], "heldout": rows, "heldout_frame": frame}
        ids = next(ids for ids, _ in (jitter.heldout_prompt(prepared, row, {"seed": seed, **RULE}) for seed in range(50))
                   if len(ids) < len(row["prompt_ids"]))   # a seed that cuts the text
        server = chat.Server("http://127.0.0.1:1/v1", prepared)
        engine = chat.ChatEngine.__new__(chat.ChatEngine)
        engine.server, engine.text = server, Text()
        messages, tools = engine.messages(ids)
        self.assertEqual(messages, [{"role": "user", "content": Text().decode(ids[3:-7])}])
        self.assertTrue("Erkläre die Gezeiten ausführlich.".startswith(messages[0]["content"]))
        self.assertEqual(engine.messages(row["prompt_ids"])[0], row["messages"], "the whole prompt keeps its messages")


class RunTests(unittest.TestCase):
    def run_once(self, data, out, **options):
        return run_full(MOCK, "unused-model", data, out, {"max_context": 4096}, parts=("speed", "heldout", "context"),
                        warmups=0, repetitions=1, context_tokens=2048, log=lambda message: None, **options)

    def prepared(self, data):
        prepared = fixture(data)
        prepared["heldout_frame"] = {"head_ids": HEAD, "tail_ids": TAIL, "messages": []}
        prepared["heldout"] = [held("zh", list(range(10, 50))), held("de", list(range(60, 90)))]
        (data / "prepared.json").write_text(json.dumps(prepared))

    def test_a_run_records_its_seed_and_lengths_and_the_seed_measures_them_again(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            self.prepared(data)
            load = runner.load_json

            def curve_from_512(path):   # curve points below the fixture's 2,048-token requirement
                value = load(path)
                if Path(path).name == "protocol.json":
                    value["context"]["curve_from"] = 512
                return value
            with patch.object(runner, "load_json", curve_from_512):
                result = self.run_once(data, data / "a.json")
                again = self.run_once(data, data / "b.json", jitter_seed=result["jitter"]["seed"])
                other = self.run_once(data, data / "c.json", jitter_seed=result["jitter"]["seed"] + 1)
            drawn = result["jitter"]
            self.assertEqual(drawn["fraction"], [0.8, 1.0])
            self.assertEqual(set(drawn["heldout"]), {"zh", "de"})
            for pid, row in result["heldout"]["prompts"].items():
                self.assertEqual(row["prompt_tokens"], drawn["heldout"][pid])
                self.assertTrue(0.8 * row["nominal_tokens"] <= row["prompt_tokens"] <= row["nominal_tokens"])
            points = result["context"]["points"]
            self.assertEqual(sorted(points, key=int), ["512", "1024", "2048"])
            self.assertEqual(points["2048"]["filled_context"], 2048, "the requirement is filled exactly")
            for key in ("512", "1024"):
                filled = points[key]["filled_context"]
                self.assertTrue(0.8 * int(key) <= filled <= int(key))
                self.assertEqual(points[key]["prompt_tokens"] + points[key]["output_tokens"], filled)
                self.assertEqual(drawn["curve"][key], filled)
            self.assertEqual(again["jitter"], drawn)
            # Another seed draws other lengths: compared over every draw, since two short held-out prompts alone (80-100%
            # of 30 and 40 tokens) draw the same lengths under another seed about one time in sixty.
            self.assertNotEqual((other["jitter"]["heldout"], other["jitter"]["curve"]), (drawn["heldout"], drawn["curve"]))
            # The speed suite keeps its lengths.
            self.assertEqual(result["speed"]["cases"]["long"]["prompt_tokens"], 60)

    def test_a_resumed_round_draws_the_lengths_its_journal_drew(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            self.prepared(data)
            first = self.run_once(data, data / "a.json")
            resumed = self.run_once(data, data / "b.json", resume=journal.path_for(data / "a.json"))
            self.assertEqual(resumed["jitter"], first["jitter"])
            with self.assertRaises(SystemExit):   # another seed is another round
                self.run_once(data, data / "c.json", resume=journal.path_for(data / "a.json"),
                              jitter_seed=first["jitter"]["seed"] + 1)


if __name__ == "__main__":
    unittest.main()
