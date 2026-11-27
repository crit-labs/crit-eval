"""The requests a whole mock round sends its engines, call by call (tests/test_eval_requests.py).

`stream()` runs every part of a full round (crit eval run --stage full) on the mock adapter, over a fixed prepared
fixture and jitter seed, through an adapter object whose engines record each request method they are called with:
the engine it went to, the method, the prompt's length, the output length and a digest of every argument but the
callback. The same file runs on any crit-eval, so the stream of one version is compared with another's
(`python tests/requests_stream.py OUT.json` on the checkout to record).
"""
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUESTS = ("generate", "generate_plain", "generate_sampled", "replay", "score", "reply", "cohort")
FACTS = 2000   # retrieval fact ids start here; the mock's facts_from knob repeats them
REQUIRED = 65536


def fixture():
    """Prepared data with every kind of case: speed cases, a held-out prompt, follow-up turns, a quality text and
    the retrieval archive (each with the fields every crit-eval from 0.8 reads)."""
    from crit_eval import load_json
    from crit_eval.quality import prepare
    protocol = load_json(ROOT / "crit_eval/protocol.json")
    turns = [list(range(100, 100 + n)) for n in (40, 60)]
    return {
        "protocol": protocol["protocol"], "suite_sha256": prepare.suite_identity(), "model": {"sha256": "model"},
        "stop_ids": [999], "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 8},
        "sampling_source": "generation_config", "valid_vocab": 1000,
        "speed": [{"id": "short", "kind": "prose", "prompt_ids": [1, 2, 3], "output_tokens": 12},
                  {"id": "long", "kind": "long-context", "prompt_ids": list(range(60)), "output_tokens": 12}],
        "heldout": [{"id": "zh", "language": "Chinese", "kind": "prose", "prompt_ids": [7, 8, 9], "output_tokens": 12}],
        "followup": [{"turn": t, "prompt_ids": ids, "output_tokens": protocol["followup"]["output_tokens"]}
                     for t, ids in enumerate(turns, 1)],
        "quality": [{"id": "c", "set": "conversations", "prompt_ids": [5, 6], "continuation_ids": [7, 8, 9],
                     "messages": [{"role": "user", "content": "c"}]}],
        "retrieval": {"head_ids": [1, 2], "tail_ids": [3, 4], "curve_tail_ids": [5, 6], "filler_ids": list(range(10, 50)),
                      "facts": [{"id": "start", "answer": "ab", "ids": [FACTS, FACTS + 1]},
                                {"id": "middle", "answer": "cd", "ids": [FACTS + 2, FACTS + 3]},
                                {"id": "end", "answer": "ef", "ids": [FACTS + 4, FACTS + 5]}],
                      "token_text": [None] * FACTS + list("abcdef")},
    }


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


class Recorded:
    """An engine whose request methods each append (engine, method, prompt tokens, output tokens, digest) to
    `calls`; everything else is the engine's own."""

    def __init__(self, engine, index, calls):
        self._engine, self._index, self._calls = engine, index, calls

    def __getattr__(self, name):
        found = getattr(self._engine, name)
        if name not in REQUESTS or not callable(found):
            return found

        def call(*args, **kwargs):
            kept = [a for a in args if not callable(a)]
            prompt = kept[0] if kept else None
            self._calls.append({"engine": self._index, "method": name,
                                "prompt_tokens": len(prompt) if isinstance(prompt, list) else None,
                                "output": kept[1] if len(kept) > 1 and isinstance(kept[1], int) else None,
                                "digest": digest([kept, {k: v for k, v in kwargs.items() if not callable(v)}])})
            return found(*args, **kwargs)
        return call


class Adapter:
    """The mock adapter, its engines recorded in opening order."""

    def __init__(self, calls):
        spec = importlib.util.spec_from_file_location("mock_adapter", ROOT / "crit_eval/examples/mock_adapter.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.calls, self.opened = calls, 0
        self.__file__ = str(ROOT / "crit_eval/examples/mock_adapter.py")

    def create_eval_engine(self, context):
        engine = Recorded(self.module.create_eval_engine(context), self.opened, self.calls)
        self.opened += 1
        return engine


def stream(parts=None):
    """(the calls of a full mock round, its result)."""
    from crit_eval.run import runner
    calls = []
    with tempfile.TemporaryDirectory() as directory:
        data = Path(directory)
        (data / "prepared.json").write_text(json.dumps(fixture()))
        options = {"parts": parts} if parts else {}
        result = runner.run(Adapter(calls), "unused-model", data, data / "result.json", {"facts_from": FACTS},
                            log=lambda message: None, context_tokens=REQUIRED, stage="full", jitter_seed=7,
                            journaled=False, **options)
    return calls, result


def served_stream():
    """(the request bodies a full round sends tests/test_eval_server.py's fake chat server, which serves no requests
    at once, so the concurrency part does not run; its result). The fake server's prepared data and reference
    scores, so quality, its honesty check and the follow-up turns run too."""
    from http.server import ThreadingHTTPServer
    import threading
    from unittest.mock import patch
    sys.path.insert(0, str(ROOT / "tests"))
    from test_eval_server import FakeChat, Text, write_data, write_store
    from crit_eval.run import chat, runner
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeChat)
    server.requests, server.headers, server.models = [], [], ["fake"]
    server.usage, server.streams_usage, server.logprobs = {}, True, {}
    server.stream_logprobs, server.one_piece, server.stream_error, server.telemetry = False, False, None, None
    server.real_usage, server.scores, server.score_offset, server.scored = True, True, 0.0, []
    server.card, server.packed = {}, None
    threading.Thread(target=server.serve_forever, daemon=True).start()
    by_bytes = {chr(i).encode(): i for i in range(1, 0xD800) if i != 0xFFFD}
    try:
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(chat, "load_text", lambda model, chat_template=None: Text()), \
                patch.object(chat, "ids_by_bytes", lambda tokenizer: by_bytes):
            data = Path(directory)
            write_data(data)
            write_store(data)
            result = runner.run(None, "unused-model", data, data / "result.json", log=lambda message: None,
                                context_tokens=4096, url=f"http://127.0.0.1:{server.server_address[1]}/v1/",
                                stage="full", jitter_seed=7, journaled=False)
    finally:
        server.shutdown()
        server.server_close()
    return [digest(body) for body in server.requests], result


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    found, _ = stream()
    bodies, _ = served_stream()
    Path(sys.argv[1]).write_text(json.dumps({"calls": found, "served": bodies}, indent=0) + "\n")
    print(f"{len(found)} calls, {len(bodies)} served requests")
