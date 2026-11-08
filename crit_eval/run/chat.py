"""crit-eval's transport for an engine behind an OpenAI-compatible server: `crit eval run --url URL`.

A chat API takes messages and returns text. Every prompt crit-eval sends is chat messages that prepare froze next to its
token ids, and the server applies its own chat template, which must be the one prepare rendered with (prepared.json's
chat_template: the model's own, crit's for a family whose checkpoints carry none, or a --chat-template file's); with
quality measured, a prompt the server renders to another length fails at once. A long-context or retrieval prompt is
its frame's messages with the document between the frame's head and tail decoded into them. Prompt and output
lengths are the server's usage counts; the reply's text, encoded with the model's tokenizer, stands in for its token
ids in the repeat, history and retrieval checks. No standard field turns end tokens off, so a reply that ends before
the case's output length is measured on the tokens it has (protocol speed.served).

Quality texts go to crit serve's score call as token ids (chat-score-v1, POST /v1/crit/score). The honesty
check's replies come with logprobs and top_logprobs (protocol quality.honesty); each reported token is mapped to
the model's id by its bytes, so the same tokens can be scored again after exactly the prompt the server rendered:
its usage must count the prompt ids the model's chat template renders.

The server must stream usage (stream_options.include_usage) and prefill every timed request: a reply whose usage
reports cached prompt tokens fails, since a prompt cache would make prefill and time to first token meaningless.
Requests judged on their text alone (retrieval) may be served from a prompt cache (prompt_cache), and their
cached tokens are recorded. A key for
the server goes in the CRIT_SERVER_KEY environment variable.

Each request may take as long as the engine's own earlier requests say it should (crit_eval.run.deadline, protocol
requests.timeout); one that takes longer raises Hung.
"""
import array
import base64
from contextlib import contextmanager
import gc
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

from .. import EVAL_DIR, load_json
from .deadline import Hung, RequestDeadline
from .outcomes import validate_completion
from ..quality.prepare import GREEDY_REQUEST, prepared_template, token_ids, with_document

# The engine facts crit serve publishes for crit-eval (crit_core.serving.ENGINE_FACTS), and the capability that lets a
# request decode plainly. prompt_cache false: the engine keeps no prompt cache (protocol followup).
FACTS = ("speculative", "plain", "draft_width", "bytes_per_pass", "host_bytes_per_pass", "draft_bytes_per_step",
         "draft_ms", "kv_bytes_per_token", "prompt_activations", "prompt_cache")
PLAIN_CAPABILITY = "chat-plain-v1"


class Server:
    """An engine behind a chat API at `url`, opened as an adapter opens one (create_eval_engine). model_name: the
    model to request; without it, the one /models lists."""

    def __init__(self, url, prepared, model_name=None):
        self.url, self.model_name = url.rstrip("/"), model_name
        self.timeout_rule = load_json(EVAL_DIR / "protocol.json")["requests"]["timeout"]
        # The server is reached directly: a proxy would add its own latency to every token.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        key = os.environ.get("CRIT_SERVER_KEY")
        self.headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
        cases = prepared["speed"] + prepared.get("heldout", []) + prepared.get("quality", []) + prepared.get("followup", [])
        self.prompts = {tuple(case["prompt_ids"]): (case["messages"], case.get("tools")) for case in cases}
        # The --chat-template the data was prepared with, if any: the server must render the same one.
        self.chat_template = prepared_template(prepared)
        retrieval = prepared.get("retrieval")
        self.frames = [(retrieval["head_ids"], retrieval["tail_ids"], retrieval["messages"], None)] if retrieval else []
        if retrieval and retrieval.get("curve_tail_ids"):   # the long-context curve's request after the same archive
            self.frames.append((retrieval["head_ids"], retrieval["curve_tail_ids"], retrieval["curve_messages"], None))
        held = prepared.get("heldout_frame")
        if held:   # a held-out prompt cut to its drawn length (protocol jitter): its user text alone in the template
            self.frames.append((held["head_ids"], held["tail_ids"], held["messages"], None))
        # The longest tail first: frames that share a head and end alike are told apart by their whole tail.
        self.frames.sort(key=lambda frame: -len(frame[1]))

    def get(self, path, timeout):
        request = urllib.request.Request(self.url + path, headers=self.headers)
        with _bounded(timeout, path), self.opener.open(request, timeout=timeout) as response:
            return json.load(response)

    def health(self, timeout):
        """The server's /health, which sits at its root beside the API (crit serve answers /health, not /v1/health):
        the API URL without a trailing /v1."""
        root = self.url[:-len("/v1")] if self.url.endswith("/v1") else self.url
        request = urllib.request.Request(root + "/health", headers=self.headers)
        with _bounded(timeout, "/health"), self.opener.open(request, timeout=timeout) as response:
            return json.load(response)

    def post(self, path, body, timeout):
        return self.post_timed(path, body, timeout)[0]

    def post_timed(self, path, body, timeout):
        """(the response, {"seconds": until it was read, "response_bytes", "parse_seconds"})."""
        request = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), headers=self.headers)
        started = time.monotonic()
        with _bounded(timeout, path), self.opener.open(request, timeout=timeout) as response:
            raw = response.read()
        read = time.monotonic()
        value = json.loads(raw)
        return value, {"seconds": read - started, "response_bytes": len(raw), "parse_seconds": time.monotonic() - read}

    def events(self, path, body, timeout):
        """The streamed events of a request; `timeout` bounds the wait for each. Lines without data, such as the
        keep-alive comments crit serve sends while the engine is quiet, are no answer."""
        request = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), headers=self.headers)
        with _bounded(timeout, path), self.opener.open(request, timeout=timeout) as response:
            waiting = time.monotonic()
            for line in response:
                line = line.strip()
                if not line.startswith(b"data:"):
                    if time.monotonic() - waiting > timeout:
                        raise TimeoutError
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    return
                yield json.loads(data)
                waiting = time.monotonic()

    def create_eval_engine(self, context):
        return ChatEngine(self, context)


class _bounded:
    """Turns a request's socket timeout into Hung."""

    def __init__(self, timeout, path):
        self.timeout, self.path = timeout, path

    def __enter__(self):
        return self

    def __exit__(self, kind, error, _traceback):
        timed_out = isinstance(error, (socket.timeout, TimeoutError)) or (
            isinstance(error, urllib.error.URLError) and isinstance(error.reason, (socket.timeout, TimeoutError)))
        if timed_out:
            raise Hung(f"{self.path} gave no answer within {self.timeout:,.0f} s") from error
        return False


def load_text(model, chat_template=None):
    """The model's tokenizer and chat template, loaded as prepare loads them: with `chat_template`, the source of the
    --chat-template the data was prepared with (prepare.prepared_template), for a model that carries none."""
    from crit_core.model import open_model
    from crit_core.text import load_model_text
    return load_model_text(open_model(model).files, clock=None, chat_template=chat_template)


def ids_by_bytes(tokenizer):
    """{bytes: token id} for mapping the tokens a server reports to the model's ids (bytes from
    Tokenizer.token_bytes). Where ids share their bytes, an ordinary piece wins over a SentencePiece byte piece and
    both over an added token. A token that reads as U+FFFD maps to nothing: servers show a token that ends inside a
    character that way, so its bytes are lost."""
    pieces = tokenizer.token_bytes()

    def rank(token):
        piece = tokenizer.id_to_token(token)
        return (token in tokenizer.added_tokens, len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">"))
    out = {}
    for token in sorted(range(len(pieces)), key=rank):
        if pieces[token] and pieces[token] != "\ufffd".encode():
            out.setdefault(pieces[token], token)
    return out


def cached_tokens(usage):
    """The prompt tokens a reply's usage reports as served from a prompt cache."""
    return int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)


def prefilled(server, usage):
    """Fail a reply whose usage reports prompt tokens served from a cache."""
    cached = cached_tokens(usage)
    if cached:
        raise RuntimeError(f"{server.url} served {cached} prompt tokens from its cache: turn its prompt cache off")


def rendered(url, served, prepared):
    """The failure of a server that rendered a prompt to another token count than the model's chat template."""
    return (f"{url} rendered the prompt as {served} tokens where the model's chat template renders {prepared}: serve "
            "the chat template the data was prepared with (prepared.json chat_template), thinking off")


def unpacked(part):
    """(tokens, logprobs, top tokens, top logprobs, alternatives) of score rows in crit serve's packed format: base64 of
    little-endian int32 tokens and float32 log-probabilities, the alternatives row by row."""
    def numbers(kind, key):
        data = array.array(kind, base64.b64decode(part[key]))
        if sys.byteorder != "little":
            data.byteswap()
        return data.tolist()
    tokens, logprobs = numbers("i", "tokens"), numbers("f", "logprobs")
    tops, values, k = numbers("i", "top_tokens"), numbers("f", "top_logprobs"), int(part["alternatives"])
    if not len(tokens) == len(logprobs) == int(part["count"]) or len(tops) != len(values) or len(tops) != k * len(tokens):
        raise ValueError("packed score rows do not add up")
    return tokens, logprobs, tops, values, k


@contextmanager
def paused_collector():
    """The cyclic garbage collector paused: building a long text's rows makes millions of small lists, each
    allocation burst of which would otherwise rescan all of them, freeing none."""
    collecting = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if collecting:
            gc.enable()


def reply_text(delta):
    """What one streamed delta adds to the reply: every text field but the role, and tool-call names and arguments."""
    parts = [value for key, value in delta.items() if key != "role" and isinstance(value, str)]
    for call in delta.get("tool_calls") or []:
        function = call.get("function") or {}
        parts += [function.get("name") or "", function.get("arguments") or ""]
    return "".join(parts)


class ChatEngine:
    def __init__(self, server, context):
        self.server = server
        if not context.get("context_tokens"):
            raise SystemExit("an engine behind a server needs --context: the context it was started with")
        self.max_context = int(context["context_tokens"])
        self.deadline = RequestDeadline(server.timeout_rule)
        cards = server.get("/models", self.deadline.seconds(0)).get("data") or []
        listed = [m.get("id") for m in cards]
        if server.model_name and server.model_name not in listed:
            raise SystemExit(f"{server.url}/models does not list {server.model_name}: it lists {', '.join(listed) or 'none'}")
        if not server.model_name and len(listed) != 1:
            raise SystemExit(f"{server.url}/models lists {', '.join(listed) or 'no model'}: name one with --served-model")
        self.model = server.model_name or listed[0]
        card = next((m for m in cards if m.get("id") == self.model), {})
        # crit serve lists the sampling fields a package applies; another server lists none, and gets every field.
        supported = card.get("supported_sampling_parameters")
        self.sampling_fields = set(supported) if isinstance(supported, list) else None
        # The loaded engine's facts crit serve publishes (engine_facts): drafting, plain decoding, bytes per pass.
        self.facts = {key: value for key, value in (card.get("engine_facts") or {}).items() if key in FACTS}
        self.plain = bool(self.facts.get("speculative") and self.facts.get("plain")
                          and PLAIN_CAPABILITY in (card.get("capabilities") or []))
        if self.plain:
            self.generate_plain = self._generate_plain
        self.text = load_text(context["model"], server.chat_template)
        self.by_bytes = None  # ids_by_bytes, built for the first quality reply
        self.packed = None   # whether the score call answers in the packed format (None: not asked yet)
        # With quality measured (crit_eval.run.runner), every prompt sent as its frozen messages must render to the ids
        # prepare froze, as the honesty check requires: a server on another template fails at its first request.
        self.exact_prompts = False

    def applied(self, sampling):
        """The fields of `sampling` the server applies (all of them for a server that lists none); temperature
        always goes."""
        return {key: value for key, value in sampling.items()
                if self.sampling_fields is None or key in self.sampling_fields or key == "temperature"}

    def info(self):
        return {"name": self.model, "configuration": {"url": self.server.url, "model": self.model},
                "max_context": self.max_context, **self.facts, "speculative": bool(self.facts.get("speculative")),
                "plain": self.plain, "served": True}

    def messages(self, prompt):
        """The messages and tools a prompt renders from: a speed or held-out prompt's own, or a frame's with the
        document between its head and tail decoded into them."""
        found = self.server.prompts.get(tuple(prompt))
        if found is not None:
            return found
        for head, tail, messages, tools in self.server.frames:  # the curve, retrieval and shared-prefix frames
            end = len(prompt) - len(tail)
            if end > len(head) and prompt[:len(head)] == head and prompt[end:] == tail:
                return with_document(messages, self.text.tokenizer.decode(prompt[len(head):end])), tools
        raise RuntimeError("the prompt is not one the prepared data sends to a server")

    def generate(self, prompt, max_new_tokens, on_tokens, prompt_cache=False, sampling=None, logprobs=None,
                 plain=False):
        """A greedy reply, or with sampling a sampled one (temperature, top_p, top_k, min_p and seed in the request
        body, protocol sampled_speed); with logprobs n each reply token's log-probability and n alternatives are
        returned as "logprobs". With prompt_cache it may be served from the server's prompt cache (a request judged
        on its text alone), and served records the cached prompt tokens."""
        prompt = list(prompt)
        messages, tools = self.messages(prompt)
        body = {"model": self.model, "messages": messages, "max_tokens": max_new_tokens,
                "stream": True, "stream_options": {"include_usage": True}, **({"tools": tools} if tools else {})}
        # Every sampling field the server applies, greedy requests' neutral penalties included: one it leaves out
        # would keep the engine's own default.
        body.update(self.applied(sampling or GREEDY_REQUEST))
        if logprobs:
            body.update(logprobs=True, top_logprobs=int(logprobs))
        if plain:
            body["crit"] = {"decode": "plain"}   # one token per target pass (chat-plain-v1)
        pieces, usage, finish, telemetry, entries, calls = [], None, None, None, [], {}
        started = time.monotonic()
        first = last = None
        for event in self.server.events("/chat/completions", body, self.deadline.seconds(len(prompt), max_new_tokens)):
            telemetry = event.get("crit_telemetry") or telemetry
            if event.get("error"):
                # A server that fails mid-stream sends its error as an event; say so instead of "no usage".
                error = event["error"]
                raise RuntimeError(f"{self.server.url} streamed an error: "
                                   f"{error.get('message') if isinstance(error, dict) else error}")
            usage = event.get("usage") or usage
            for choice in event.get("choices") or []:
                finish = choice.get("finish_reason") or finish
                entries += ((choice.get("logprobs") or {}).get("content") or []) if logprobs else []
                # Tool calls the server parsed itself, by index: their names and arguments as streamed (speed.tool_calls).
                for call in (choice.get("delta") or {}).get("tool_calls") or []:
                    slot = calls.setdefault(call.get("index", 0), {"name": "", "arguments": ""})
                    function = call.get("function") or {}
                    slot["name"] += function.get("name") or ""
                    slot["arguments"] += function.get("arguments") or ""
                piece = reply_text(choice.get("delta") or {})
                if piece:
                    last = time.monotonic()
                    first = first or last
                    pieces.append(piece)
                    on_tokens(token_ids(self.text, piece))
        if not usage or not usage.get("completion_tokens"):
            raise RuntimeError(f"{self.server.url} streamed no usage: it must support stream_options.include_usage")
        if not prompt_cache:
            prefilled(self.server, usage)
        if self.exact_prompts and tuple(prompt) in self.server.prompts and usage.get("prompt_tokens") != len(prompt):
            raise RuntimeError(rendered(self.server.url, usage.get("prompt_tokens"), len(prompt)))
        if not pieces:
            raise RuntimeError(f"{self.server.url} streamed no reply text")
        self.deadline.observe(usage["prompt_tokens"], first - started, usage["completion_tokens"], last - first)
        # crit serve reports output tokens per target pass after prefill when the engine counts its rounds.
        per_round = (telemetry or {}).get("tokens_per_round")
        rounds = (round(usage["completion_tokens"] / per_round)
                  if isinstance(per_round, (int, float)) and per_round > 0 else None)
        out = {"tokens": token_ids(self.text, "".join(pieces)), "rounds": rounds,
               "served": {"prompt_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"],
                          "finish_reason": finish, "cached_tokens": cached_tokens(usage)}}
        if calls:
            out["tool_calls"] = [calls[index] for index in sorted(calls)]
        # Draft telemetry crit serve passes on: drafts accepted per round, verify and draft milliseconds.
        accepted = (telemetry or {}).get("accepted")
        if isinstance(accepted, list) and all(type(a) is int for a in accepted):
            out.update(accepted=accepted, rounds=len(accepted))
            drafted = (telemetry or {}).get("drafted")
            if isinstance(drafted, list) and len(drafted) == len(accepted) and all(type(d) is int for d in drafted):
                out["drafted"] = drafted
        out.update({key: telemetry[key] for key in ("verify_ms", "draft_ms", "prefill_ms")
                    if isinstance((telemetry or {}).get(key), (int, float))})
        # The decode's routed expert reads and those its device cache served (protocol headroom.tiers).
        out.update({key: telemetry[key] for key in ("expert_reads", "expert_hits") if type((telemetry or {}).get(key)) is int})
        # crit serve's ttft_ms is the engine's own time to its first token: the client's beyond it is HTTP and serving.
        if isinstance((telemetry or {}).get("ttft_ms"), (int, float)):
            out["first_token_ms"] = telemetry["ttft_ms"]
        if logprobs and entries:
            # Tokens are named by their bytes (their text where a server gives none): the sampling check only matches
            # each token among its alternatives, and pieces of one character can share a text.
            name = lambda e: bytes(e["bytes"]) if e.get("bytes") is not None else e.get("token")  # noqa: E731
            out["logprobs"] = [{"token": name(e), "logprob": e.get("logprob"),
                                "top": [[name(a), a.get("logprob")] for a in e.get("top_logprobs") or []]}
                               for e in entries]
            try:   # the same reply as reply() gives it, ids by bytes, for the honesty check of a long reply
                out["reply"] = self._reply_rows(entries, finish, logprobs, usage)
                served_ids = self.served_ids(prompt, messages, tools, usage)
                if served_ids != prompt:
                    out["reply"]["prompt_ids"] = served_ids
            except RuntimeError as error:
                out["reply_unmapped"] = str(error)
        return out

    def served_ids(self, prompt, messages, tools, usage):
        """The prompt ids a reply was served after. A frame's prompt reaches the server as text decoded from its ids,
        which the server's chat template renders anew and may tokenize to other ids (a document cut where a merge
        spans the cut, a token that decodes to text encoding as two): the reply is scored after what the server
        rendered, the model's chat template applied to the same messages. RuntimeError when the server's usage counts
        neither."""
        if tuple(prompt) in self.server.prompts:
            return prompt
        rendered_ids = list(self.text.encode_chat(messages, tools=tools, add_generation_prompt=True,
                                                  enable_thinking=False))
        if usage.get("prompt_tokens") == len(rendered_ids):
            return rendered_ids
        if usage.get("prompt_tokens") == len(prompt):
            return prompt
        raise RuntimeError(rendered(self.server.url, usage.get("prompt_tokens"), len(rendered_ids)))

    def generate_sampled(self, prompt, max_new_tokens, on_tokens, sampling, logprobs=None):
        """A sampled reply (protocol sampled_speed) at the sampling fields given."""
        return self.generate(prompt, max_new_tokens, on_tokens, sampling=sampling, logprobs=logprobs)

    def _generate_plain(self, prompt, max_new_tokens, on_tokens):
        """Plain greedy on the served speculative engine (crit serve's crit.decode plain): crit-eval's exactness
        check."""
        return self.generate(prompt, max_new_tokens, on_tokens, plain=True)

    def reply(self, prompt, max_new_tokens, top_k):
        """A quality prompt's greedy reply with the log-probabilities the server reports for each token (protocol
        quality.rule): {"tokens", "logprobs", "top": [[id, logprob], ...] per token, "finish_reason",
        "prompt_tokens"}. An alternative whose bytes are no token of the model is left out of "top"."""
        prompt = list(prompt)
        messages, tools = self.messages(prompt)
        url = self.server.url
        response, timing = self.server.post_timed("/chat/completions", {
            "model": self.model, "messages": messages, "max_tokens": max_new_tokens, **self.applied(GREEDY_REQUEST),
            "logprobs": True, "top_logprobs": top_k, **({"tools": tools} if tools else {})},
            self.deadline.seconds(len(prompt), max_new_tokens))
        # A whole reply's time bounds its prefill from above.
        self.deadline.observe(len(prompt), timing["seconds"])
        usage = response.get("usage") or {}
        if usage.get("prompt_tokens") != len(prompt):
            raise RuntimeError(rendered(url, usage.get("prompt_tokens"), len(prompt)))
        prefilled(self.server, usage)
        choice = (response.get("choices") or [{}])[0]
        entries, finish = (choice.get("logprobs") or {}).get("content"), choice.get("finish_reason")
        if not entries:
            raise RuntimeError(f"{url} returned no logprobs: it must support logprobs and top_logprobs")
        return {**self._reply_rows(entries, finish, top_k, usage), "timing": timing}

    def _reply_rows(self, entries, finish, top_k, usage):
        """{"tokens", "logprobs", "top": [[id, logprob], ...] per token, "finish_reason", "prompt_tokens"} of a
        reply's logprobs entries, every token and alternative mapped to the model's ids by its bytes (an alternative
        whose bytes are no token is left out); RuntimeError when a token has no bytes, is no token, or has fewer than
        top_k alternatives."""
        url = self.server.url
        if self.by_bytes is None:
            self.by_bytes = ids_by_bytes(self.text.tokenizer)
        tokens, logprobs, top = [], [], []
        for position, entry in enumerate(entries):
            data = entry.get("bytes")
            if data is None:
                raise RuntimeError(f"{url} reports no bytes for reply token {position} ({entry.get('token')!r}): "
                                   "crit-eval maps each reply token to the model's ids by its bytes")
            if not data and position == len(entries) - 1 and finish == "stop":
                break  # the end token, which a server may report as no text
            token = self.by_bytes.get(bytes(data))
            if token is None:
                raise RuntimeError(f"{url} reports reply token {position} as {bytes(data)!r}, which is no token of the model")
            alternatives = entry.get("top_logprobs") or []
            if len(alternatives) < top_k:
                raise RuntimeError(f"{url} returned {len(alternatives)} top_logprobs for reply token {position}, not {top_k}")
            row = {}
            for alternative in alternatives:
                other = self.by_bytes.get(bytes(alternative.get("bytes") or b""))
                if other is not None:
                    row.setdefault(other, float(alternative["logprob"]))
            tokens.append(token)
            logprobs.append(float(entry["logprob"]))
            top.append([[other, value] for other, value in row.items()])
        if not tokens:
            raise RuntimeError(f"{url} returned an empty reply")
        return {"tokens": tokens, "logprobs": logprobs, "top": top, "finish_reason": finish,
                "prompt_tokens": usage["prompt_tokens"]}

    def score(self, prompt, continuation, top_k, prompt_rows=True):
        """Fixed text scored through the engine's serving paths (crit serve's chat-score-v1, POST /v1/crit/score):
        {"prompt": rows for prompt[1:] from its prefill, "continuation": rows for each continuation token from its
        decode forced to the continuation (none with prompt_rows false), "timing"}, rows {"logprob",
        "top": [[id, logprob], ...]}; timing: the
        request's seconds, the engine's own prefill_ms and decode_ms (crit_telemetry), the response's bytes and the
        seconds parsing it took. The rows come packed (unpacked) from a crit serve that offers it, else as JSON."""
        body = {"model": self.model, "prompt_ids": list(prompt), "continuation_ids": list(continuation),
                "top_logprobs": top_k, "prompt_rows": prompt_rows, **({} if self.packed is False else {"format": "packed"})}
        try:
            value, timing = self.server.post_timed("/crit/score", body, self.deadline.seconds(len(prompt), len(continuation)))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            if exc.code == 400 and self.packed is None and '"param": "format"' in detail:
                self.packed = False   # an earlier crit serve: JSON rows
                return self.score(prompt, continuation, top_k, prompt_rows)
            if exc.code == 404:
                raise RuntimeError(f"{self.server.url} scores no fixed text: serve the engine with crit serve and a "
                                   "package declaring chat-score-v1") from exc
            raise RuntimeError(f"{self.server.url} failed to score fixed text ({exc.code}): {detail}") from exc
        self.deadline.observe(len(prompt) + len(continuation), timing["seconds"])
        converted = time.monotonic()
        self.packed = value.get("format") == "packed"
        with paused_collector():
            rows = self._score_rows(value, list(prompt)[1:] if prompt_rows else [], list(continuation), top_k)
        engine = value.get("crit_telemetry") or {}
        rows["timing"] = {**timing, "parse_seconds": timing["parse_seconds"] + time.monotonic() - converted,
                          "engine_prefill_ms": engine.get("prefill_ms"), "engine_decode_ms": engine.get("decode_ms")}
        return rows

    def _score_rows(self, value, prompt, continuation, top_k):
        """A score response's {"prompt": rows, "continuation": rows} for the tokens it was sent, rows {"logprob",
        "top": [[id, logprob], ...]}."""
        rows = {}
        for part, tokens in (("prompt", prompt), ("continuation", continuation)):
            found = value.get(part)
            if self.packed and isinstance(found, dict):
                try:
                    found_tokens, logprobs, tops, values, k = unpacked(found)
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(f"{self.server.url} returned malformed packed {part} rows: {exc}") from exc
                if found_tokens != tokens:
                    raise RuntimeError(f"{self.server.url} scored other {part} tokens than it was sent")
                if k != top_k:
                    raise RuntimeError(f"{self.server.url} returned other than {top_k} alternatives per {part} token")
                # The rows straight from the arrays' Python numbers, each row's pairs a slice of one list of them.
                pairs = list(map(list, zip(tops, values)))
                rows[part] = [{"logprob": logprob, "top": pairs[i * k:(i + 1) * k]}
                              for i, logprob in enumerate(logprobs)]
                continue
            if not isinstance(found, list) or [row[0] for row in found] != tokens:
                raise RuntimeError(f"{self.server.url} scored other {part} tokens than it was sent")
            if any(len(row[2]) != top_k for row in found):
                raise RuntimeError(f"{self.server.url} returned other than {top_k} alternatives per {part} token")
            rows[part] = [{"logprob": float(row[1]), "top": [[int(t), float(v)] for t, v in row[2]]} for row in found]
        return rows

    def loads(self):
        """How many times the server's resident host has loaded an engine (crit serve's /health engine_loads), or None
        from a server that reports none: the closed track's one load for the run (protocol tracks)."""
        status = (self.server.health(self.deadline.seconds(0)) or {}).get("engine") or {}
        value = status.get("engine_loads") if isinstance(status, dict) else None
        return value if type(value) is int else None

    def device_free_bytes(self):
        """The device memory the server's engine process can still allocate, as it read it when it last became ready
        (crit serve's /health engine device_free_bytes), or None from a server that reports none."""
        status = (self.server.health(self.deadline.seconds(0)) or {}).get("engine") or {}
        value = status.get("device_free_bytes") if isinstance(status, dict) else None
        return value if type(value) is int and value >= 0 else None

    def telemetry(self):
        """The engine's own counters, as crit serve's /health reports them at its root (engine_telemetry:
        {"expert_cache": {"reads", "hits", ...}, ...}; CONTRACTS section 7), or None from a server that reports none:
        the expert cache's hit rate where requests report no expert reads (protocol headroom.tiers)."""
        value = (self.server.health(self.deadline.seconds(0)) or {}).get("engine_telemetry")
        return value if isinstance(value, dict) else None

    def capacity(self):
        """The requests the server runs at once: crit serve's max_concurrent_requests on the model's card, read again
        now (a concurrent package's engine declares it as it loads); None from a server that reports none."""
        cards = self.server.get("/models", self.deadline.seconds(0)).get("data") or []
        value = next((m for m in cards if m.get("id") == self.model), {}).get("max_concurrent_requests")
        return value if type(value) is int and value >= 1 else None

    def cohort(self, clients, max_new_tokens):
        """One closed-loop level of the concurrency part (protocol concurrency.loop): clients[c] is client c's prompts,
        each sent greedy when the client's previous reply has ended, every client starting at once. {"seconds": the
        level's wall interval, "requests": per request its client, index, prompt_tokens, the seconds from the level's
        start when it was sent, its first and last reply text arrived and its reply was read (started, first, last,
        finished), the usage's completion_tokens, finish_reason and cached_tokens, the stream's validation errors
        (crit_eval.run.outcomes.validate_completion, plus cached prompt tokens) and the reply's tokens}. A request
        waits for each streamed event as long as the engine's earlier requests say it should, times the clients (a
        server that runs them one at a time); one past that raises Hung."""
        start, rows, failures = threading.Event(), [], []
        epoch = [0.0]

        def client(index, prompts):
            start.wait()
            for position, prompt in enumerate(prompts):
                try:
                    rows.append({"client": index, "index": position,
                                 **self._streamed(list(prompt), max_new_tokens, len(clients), epoch[0])})
                except BaseException as error:   # Hung or a transport failure: the level stops with it
                    failures.append(error)
                    return
        threads = [threading.Thread(target=client, args=(c, prompts), name=f"crit-eval-client-{c}", daemon=True)
                   for c, prompts in enumerate(clients)]
        for thread in threads:
            thread.start()
        epoch[0] = time.perf_counter()
        start.set()
        for thread in threads:
            thread.join()
        seconds = time.perf_counter() - epoch[0]
        if failures:
            raise failures[0]
        rows.sort(key=lambda row: (row["client"], row["index"]))
        return {"seconds": seconds, "requests": rows}

    def _streamed(self, prompt, max_new_tokens, clients, epoch):
        """One greedy request of a cohort, read raw so the stream itself is validated (cohort)."""
        messages, tools = self.messages(prompt)
        body = {"model": self.model, "messages": messages, "max_tokens": max_new_tokens, "stream": True,
                "stream_options": {"include_usage": True}, **({"tools": tools} if tools else {}),
                **self.applied(GREEDY_REQUEST)}
        timeout = self.deadline.seconds(len(prompt), max_new_tokens) * max(1, clients)
        request = urllib.request.Request(self.server.url + "/chat/completions", data=json.dumps(body).encode(),
                                         headers=self.server.headers)
        lines, pieces, usage, first, last, telemetry = [], [], {}, None, None, {}
        started = time.perf_counter() - epoch
        try:
            with _bounded(timeout, "/chat/completions"), self.server.opener.open(request, timeout=timeout) as response:
                status = response.status
                for line in response:
                    lines.append(line)
                    data = line.strip()
                    if not data.startswith(b"data:") or data[5:].strip() in (b"", b"[DONE]"):
                        continue
                    event = json.loads(data[5:])
                    usage = event.get("usage") or usage
                    telemetry = event.get("crit_telemetry") or telemetry
                    for choice in event.get("choices") or []:
                        piece = reply_text(choice.get("delta") or {})
                        if piece:
                            last = time.perf_counter() - epoch
                            first = last if first is None else first
                            pieces.append(piece)
        except urllib.error.HTTPError as error:
            status, lines = error.code, [error.read()]
        except Hung:
            raise
        except (OSError, ValueError) as error:
            # The connection failed or a frame did not parse: the request failed, the level goes on.
            finished = time.perf_counter() - epoch
            return {"prompt_tokens": len(prompt), "started": started, "first": first, "last": last,
                    "finished": finished, "completion_tokens": None, "finish_reason": None, "cached_tokens": 0,
                    "errors": [f"Transport failed: {type(error).__name__}: {error}"[:500]],
                    "tokens": token_ids(self.text, "".join(pieces))}
        finished = time.perf_counter() - epoch
        outcome = validate_completion(status, lines)
        errors = list(outcome.errors)
        if cached_tokens(usage):
            errors.append(f"{self.server.url} served {cached_tokens(usage)} prompt tokens from its cache")
        # The request's routed expert reads and those the device cache served (crit serve's telemetry): the level's own
        # hit rate prices its batched ceiling (protocol concurrency.ceiling).
        telemetry = telemetry if isinstance(telemetry, dict) else {}
        counts = {key: telemetry[key] for key in ("expert_reads", "expert_hits") if type(telemetry.get(key)) is int}
        return {"prompt_tokens": len(prompt), "started": started, "first": first, "last": last, "finished": finished,
                "completion_tokens": outcome.completion_tokens, "finish_reason": outcome.finish_reason,
                "cached_tokens": cached_tokens(usage), "errors": errors,
                "tokens": token_ids(self.text, "".join(pieces)),
                **(counts if len(counts) == 2 and 0 <= counts["expert_hits"] <= counts["expert_reads"] else {})}

    def close(self):
        pass
