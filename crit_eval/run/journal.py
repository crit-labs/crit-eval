"""An Eval round's journal: every engine the round opens and every request it serves, one JSON line each, appended
and synced to disk the moment it completes, so a round that stops (interrupted, crashed, a hung engine, a disturbed
device) resumes where it stopped instead of from the start.

A resumed round runs the suite again from its first part. While its requests are those of the journal it resumes,
in the same order with the same inputs, each is answered from the journal, and the round's clock (`clock`, which
Eval's timings and budget read) gives back the readings the earlier round took, in the order it took them: every
sample, gate and stage decision the earlier round made is made again, from the same measurements, to the bit. An engine whose requests are all in the journal is never loaded;
one with requests left loads when its first live request comes, and the journal's requests stand as its earlier
ones. From the first request the journal does not hold (or one that differs from it: the round took another path),
the round measures live. A round that stopped on a disturbed device resumes from the start of the part it stopped in,
so the disturbed case is measured again rather than read back.

Lines (protocol journal): `round` (the round's identity, what it resumes), `open` (an engine: its context, info,
methods and load time), `mark` (a part starts), `call` (a request: method, inputs digest, delivered tokens with
their offsets, seconds, result and what a served engine's request deadline observed of it, or the error a request
raised, raised again on replay); each of these three carries the clock readings taken since the line before it
(`clock`), `load` (an engine the round opened only for its live requests), `diverged` (where
replay stopped early), `end` (the round's outcome). Replayed lines carry `replayed: true`, so each journal is
complete on its own and a resumed round can be resumed again."""
import datetime
import hashlib
import json
import os
import time
from pathlib import Path

JOURNAL = "crit-eval-journal-v1"
SUFFIX = ".journal.jsonl"
# Request methods of an engine (docs/ADAPTER.md) the journal records; every other attribute is the engine's own. A
# served engine's capacity and cohort (the concurrency part, crit_eval.run.concurrency) are each one line.
METHODS = ("generate", "generate_plain", "generate_sampled", "replay", "score", "reply", "capacity", "cohort", "loads")
# Info keys an engine opened for its live requests must answer as the journaled one did.
SAME_INFO = ("name", "max_context", "speculative", "served")   # a served engine's configuration names its port

_active = None   # the journal of the round in progress (Journal.__enter__)


def clock():
    """The round's clock, in wall seconds: while a resumed round is answered from its journal, the readings the earlier
    round took, in order; then wall time going on from the last of them. A journaling round keeps every reading with
    the line it precedes."""
    return _active.now() if _active is not None else time.perf_counter()


def active():
    return _active


def path_for(out):
    """The journal beside a result file: R.json's is R.journal.jsonl."""
    out = Path(out)
    return out.with_name((out.name[:-5] if out.name.endswith(".json") else out.name) + SUFFIX)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _plain(value):
    """value as JSON keeps it (tuples as lists, array-likes by tolist()); TypeError when it holds anything else."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    raise TypeError(f"{type(value).__name__} is not JSON")


def read(path):
    """The lines of a journal; a last line cut short by a crash is left out."""
    lines = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError:
                break
    if not lines or lines[0].get("event") != "round" or lines[0].get("journal") != JOURNAL:
        raise SystemExit(f"{path} is not a {JOURNAL} journal")
    return lines


def replayable(lines):
    """The open, mark and call lines a round resuming this journal answers from, in order: up to the start of the
    part a disturbed round stopped in, and before the first request whose result the journal could not keep."""
    rows = [line for line in lines if line["event"] in ("open", "mark", "call")]
    end = next((line for line in reversed(lines) if line["event"] == "end"), None)
    cut = next((i for i, row in enumerate(rows) if row["event"] == "call" and not row.get("recorded", True)), len(rows))
    rows = rows[:cut]
    if end is not None and end.get("outcome") == "disturbed":
        marks = [i for i, row in enumerate(rows) if row["event"] == "mark"]
        rows = rows[:marks[-1]] if marks else []
    return rows


def differences(identity, other):
    return sorted(key for key in set(identity) | set(other) if identity.get(key) != other.get(key))


class Journal:
    """with Journal(path, identity, resume=EARLIER_JOURNAL): the round's journal while the block runs. identity: what
    makes two rounds the same measurement (protocol, data, engine package, configuration, requirements); a resumed
    journal's must equal it."""

    def __init__(self, path, identity, resume=None, log=print):
        self.path, self.identity, self.log = Path(path), dict(identity), log
        self.offset = 0.0       # how far the live clock runs ahead of this process's wall clock (after a replay)
        self.last = time.perf_counter()
        self.reads = []         # clock readings since the last line, written with the next open, mark or call
        self.queue = None       # the next line's readings still to give back
        self.given = []         # the readings given back from it so far
        self.engines = 0
        self.pending = []       # the resumed journal's lines not yet answered
        self.resumed = None
        self.diverged = None
        self.counts = {"replayed_calls": 0, "replayed_seconds": 0.0, "replayed_opens": 0, "live_calls": 0,
                       "live_opens": 0, "unrecorded_calls": 0}
        self.replays = 0         # requests answered so far (HostLoad reads it: a replayed interval has no load)
        self.ended = None
        if resume is not None:
            earlier = read(resume)
            found = earlier[0]["identity"]
            changed = differences(self.identity, found)
            if changed:
                raise SystemExit(f"{resume} journals another round: {', '.join(changed)} differ; start a new round")
            self.pending = replayable(earlier)
            end = next((line["outcome"] for line in reversed(earlier) if line["event"] == "end"), None)
            self.resumed = {"path": str(resume), "sha256": _file_sha(resume), "outcome": end or "unfinished",
                            "lines": len(earlier), "replayable": len(self.pending)}
        if resume is not None and self.path.resolve() == Path(resume).resolve():
            raise SystemExit(f"{self.path}: a round resumes a journal into a new one, beside its own new result")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A new round at this path replaces its journal, as it replaces the result beside it.
        self.handle = open(self.path, "w", encoding="utf-8")
        self._write({"event": "round", "journal": JOURNAL, "identity": self.identity,
                     "identity_sha256": digest(self.identity), "started_at": _now(), "resumes": self.resumed})
        if self.resumed:
            log(f"journal: resuming {self.resumed['path']} ({self.resumed['outcome']}): "
                f"{self.resumed['replayable']} of its {self.resumed['lines']} lines can be answered from it")

    def __enter__(self):
        global _active
        if _active is not None:
            raise RuntimeError("an Eval round is already journaling in this process")
        _active = self
        return self

    def __exit__(self, kind, error, _traceback):
        global _active
        if self.ended is None:
            self.end("completed" if kind is None else "interrupted" if issubclass(kind, KeyboardInterrupt)
                     else "failed", None if error is None else f"{type(error).__name__}: {error}"[:2000])
        self.handle.close()
        _active = None
        return False

    def _write(self, line):
        self.handle.write(json.dumps(line, separators=(",", ":")) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def now(self):
        """A reading of the round's clock (clock())."""
        if self.pending:
            if self.queue is None:
                self.queue = list(self.pending[0].get("clock") or [])
            if self.queue:
                self.last = self.queue.pop(0)
                self.given.append(self.last)
                return self.last
            self._diverge("a clock reading")
        self.last = max(time.perf_counter() + self.offset, self.last)
        self.reads.append(self.last)
        return self.last

    def _live(self):
        """Replay is over: the clock goes on from its last reading, and the readings given back for a line not
        answered belong to the next line this round writes."""
        self.offset = self.last - time.perf_counter()
        self.reads, self.queue, self.given = self.given + self.reads, None, []
        self.pending = []

    def _peek(self, kind, matches):
        """The resumed journal's next line when it is `kind` and matches; else replay stops here for good."""
        if not self.pending:
            return None
        row = self.pending[0]
        if row["event"] == kind and matches(row):
            return row
        self._diverge(kind)
        return None

    def _pop(self):
        row = self.pending.pop(0)
        self.queue, self.given = None, []
        if not self.pending:
            self._live()
        return row

    def _diverge(self, found):
        row = self.pending[0]
        self.diverged = {"after_replayed_calls": self.counts["replayed_calls"], "expected": row["event"],
                         "found": found}
        self._write({"event": "diverged", "expected": {k: row.get(k) for k in ("event", "engine", "method", "part")},
                     "found": found, "replayed_calls": self.counts["replayed_calls"]})
        self.log(f"journal: the round took another path than the journal it resumes (expected {row['event']} "
                 f"{row.get('method') or row.get('part') or ''}, found {found}): measuring live from here")
        self._live()

    def _record(self, line):
        """Write a live open, mark or call line with the clock readings taken since the line before it."""
        line["clock"], self.reads = self.reads, []
        self._write(line)

    def end(self, outcome, detail=None):
        if self.ended is None:
            self.ended = outcome
            self._write({"event": "end", "outcome": outcome, "detail": detail, "ended_at": _now(),
                         **{k: round(v, 3) if isinstance(v, float) else v for k, v in self.counts.items()}})

    def mark(self, part):
        """A part of the round starts (runner PartTimes)."""
        row = self._peek("mark", lambda r: r["part"] == part)
        if row is not None:
            self._write({**self._pop(), "replayed": True})
        else:
            self._record({"event": "mark", "part": part})

    def open(self, opener, context, key=None):
        """The engine opener(context) gives, as (engine, info): the journaled engine's info without loading it while
        the resumed journal holds its requests, else the engine itself, opened now. key: what identifies the engine
        beside its context (default the context)."""
        index = self.engines
        self.engines += 1
        context_sha = digest(context if key is None else key)
        row = self._peek("open", lambda r: r["context_sha256"] == context_sha)
        if row is not None:
            self.counts["replayed_opens"] += 1
            self._write({**self._pop(), "engine": index, "replayed": True})
            wrapped = JournaledEngine(self, index, opener, context, context_sha, dict(row["info"]), row["methods"])
            wrapped.kind = _class(row.get("class"))
            return wrapped, dict(row["info"])
        started = time.perf_counter()
        engine, info = opener(context)
        seconds = time.perf_counter() - started
        methods = [name for name in METHODS if callable(getattr(engine, name, None))]
        self.counts["live_opens"] += 1
        self._record({"event": "open", "engine": index, "context_sha256": context_sha, "info": _plain(info),
                      "methods": methods, "class": _class_name(type(engine)), "load_seconds": round(seconds, 3)})
        wrapped = JournaledEngine(self, index, opener, context, context_sha, info, methods)
        wrapped.real, wrapped.kind = engine, type(engine)
        return wrapped, info

    def call(self, engine, method, args, kwargs):
        """One request on a journaled engine: answered from the resumed journal when it is the request it holds next,
        else served by the engine and recorded."""
        callbacks = [value for value in (*args, *kwargs.values()) if callable(value)]
        inputs = [[a for a in args if not callable(a)], {k: v for k, v in kwargs.items() if not callable(v)}]
        inputs_sha = digest([method, inputs])
        row = self._peek("call", lambda r: r["engine_key"] == engine.key and r["method"] == method
                         and r["inputs_sha256"] == inputs_sha)
        if row is not None:
            # The callbacks read the clock as they did when the tokens arrived: the line's readings answer them.
            for _, tokens in row["deliveries"]:
                for callback in callbacks:
                    callback(list(tokens))
            self._pop()
            self.counts["replayed_calls"] += 1
            self.counts["replayed_seconds"] += row["seconds"]
            self.replays += 1
            engine.served.append(row)
            self._write({**row, "engine": engine.index, "replayed": True})
            if row.get("error"):
                raise _error(row["error"])
            return row["result"]
        real = engine.load()
        deliveries = []
        started = time.perf_counter()

        def recorder(callback):
            def on_tokens(tokens):
                deliveries.append([round(time.perf_counter() - started, 6), _plain(list(tokens))])
                return callback(tokens)
            return on_tokens
        live_args = [recorder(a) if callable(a) else a for a in args]
        live_kwargs = {k: recorder(v) if callable(v) else v for k, v in kwargs.items()}
        line = {"event": "call", "engine": engine.index, "engine_key": engine.key, "method": method,
                "inputs_sha256": inputs_sha, "prompt_tokens": len(args[0]) if args and isinstance(args[0], list)
                and not any(isinstance(t, list) for t in args[0]) else None, "deliveries": deliveries}
        deadline = getattr(real, "deadline", None)
        seen = [len(getattr(deadline, name, ())) for name in ("prefills", "decodes")]
        try:
            result = getattr(real, method)(*live_args, **live_kwargs)
        except Exception as error:
            # A request that raised is part of the round: a round resuming this journal raises it at the same place.
            line.update(seconds=round(time.perf_counter() - started, 6), result=None,
                        error={"types": [kind.__name__ for kind in type(error).__mro__], "message": str(error)})
            self.counts["live_calls"] += 1
            self._record(line)
            raise
        line["seconds"] = round(time.perf_counter() - started, 6)
        if hasattr(deadline, "observe"):
            # What the engine's request deadline observed of this request, given back to it as it was on replay.
            line["observed"] = {name: _plain(getattr(deadline, name)[count:])
                                for name, count in zip(("prefills", "decodes"), seen)}
        try:
            line["result"] = _plain(result)
        except TypeError as error:
            # The request stands in this round; a round resuming this journal measures it and what follows again.
            line.update(result=None, recorded=False, reason=str(error))
            self.counts["unrecorded_calls"] += 1
        self.counts["live_calls"] += 1
        self._record(line)
        return result

    def summary(self, portable=str):
        """What the result records of the journal (result journal): its paths relative to the journal's folder when
        inside it (a result's journal is beside it), else as portable(path) gives them."""
        def named(path):
            path = Path(path).resolve()
            try:
                return path.relative_to(self.path.resolve().parent).as_posix()
            except ValueError:
                return portable(path)
        return {"journal": JOURNAL, "path": named(self.path), "identity_sha256": digest(self.identity),
                "resumed": None if self.resumed is None else {**self.resumed, "path": named(self.resumed["path"])},
                "diverged": self.diverged,
                **{k: round(v, 3) if isinstance(v, float) else v for k, v in self.counts.items()}}


class JournaledEngine:
    """An engine whose requests go through the round's journal. It loads the engine it stands for only when a request
    the journal cannot answer needs it, or when Eval reads an attribute that only the engine has."""

    def __init__(self, journal, index, opener, context, context_sha, info, methods):
        self.journal, self.index, self.opener, self.context = journal, index, opener, context
        self.info_value, self.methods = info, list(methods)
        self.key = f"{index}:{context_sha}"   # the engine a call line belongs to, across rounds
        self.real = None
        self.kind = None      # the engine's class: isinstance sees it through __class__ (a served ChatEngine)
        self.served = []      # the journal's requests this engine answered before it loaded

    def info(self):
        return self.info_value

    @property
    def __class__(self):
        return self.kind or JournaledEngine

    def load(self):
        if self.real is None:
            started = time.perf_counter()
            engine, info = self.opener(self.context)
            seconds = time.perf_counter() - started
            changed = [k for k in SAME_INFO if _plain(info.get(k)) != _plain(self.info_value.get(k))]
            if changed:
                engine.close()
                raise SystemExit("the engine opened to go on answers info() otherwise than the one in the journal "
                                 f"({', '.join(changed)}): resume only the same engine, or start a new round")
            self.real = engine
            self.journal.counts["live_opens"] += 1
            self.journal._write({"event": "load", "engine": self.index, "load_seconds": round(seconds, 3),
                                 "after_replayed_calls": len(self.served)})
            self.journal.log(f"journal: engine {self.index} loaded in {seconds:.1f} s after {len(self.served)} "
                             "requests answered from the journal")
            _observe(engine, self.served)
        return self.real

    def close(self):
        if self.real is not None:
            self.real.close()

    def __getattr__(self, name):
        if name in METHODS:
            if name not in self.methods:
                raise AttributeError(name)
            return lambda *args, **kwargs: self.journal.call(self, name, args, kwargs)
        if name == "deadline" and self.real is None:
            return self._deadline()
        return getattr(self.load(), name)

    def _deadline(self):
        """A served engine's request deadline as it stands after the journal's requests this engine answered, the same
        observations the engine's own deadline made of them (the stages' rates read it); None for an engine that keeps
        none."""
        if not self.info_value.get("served"):
            return None
        from .. import EVAL_DIR, load_json
        from .deadline import RequestDeadline
        seen = RequestDeadline(load_json(EVAL_DIR / "protocol.json")["requests"]["timeout"])
        _observe(seen, self.served)
        return seen

    def __setattr__(self, name, value):
        if name in ("journal", "index", "opener", "context", "info_value", "methods", "key", "real", "kind", "served"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.load(), name, value)


def _observe(engine, rows):
    """A served engine's request deadline (crit_eval.run.deadline), or the deadline itself, learns the journal's
    requests as if it had served them, so its first live request is bounded by the rates the round measured."""
    deadline = engine if hasattr(engine, "observe") else getattr(engine, "deadline", None)
    if deadline is None or not hasattr(deadline, "observe"):
        return
    for row in rows:
        if row.get("error"):
            continue    # a request that raised observed nothing
        if "observed" in row:
            deadline.prefills.extend(tuple(pair) for pair in row["observed"]["prefills"])
            deadline.decodes.extend(tuple(pair) for pair in row["observed"]["decodes"])
            continue
        prompt = row.get("prompt_tokens") or 0
        if row["deliveries"]:
            first, last = row["deliveries"][0][0], row["deliveries"][-1][0]
            tokens = sum(len(chunk) for _, chunk in row["deliveries"])
            deadline.observe(prompt, first, tokens, last - first)
        else:
            deadline.observe(prompt, row["seconds"])


def describe(path):
    """How far the round a journal records got, as text: what it resumed, its engines and parts, its requests
    replayed and live with their seconds, where replay stopped early and how it ended."""
    lines = read(path)
    head, rows = lines[0], lines[1:]
    calls = [r for r in rows if r["event"] == "call"]
    replayed = [r for r in calls if r.get("replayed")]
    parts = [r["part"] for r in rows if r["event"] == "mark"]
    end = next((r for r in reversed(rows) if r["event"] == "end"), None)
    out = [f"journal {path}: round started {head['started_at']} (identity {head['identity_sha256'][:12]})"]
    if head.get("resumes"):
        out.append(f"resumes {head['resumes']['path']} ({head['resumes']['outcome']}, "
                   f"{head['resumes']['replayable']} of its lines answerable)")
    opens = [r for r in rows if r["event"] in ("open", "load")]
    out.append(f"engines: {sum(r['event'] == 'open' for r in rows)} opened"
               f" ({sum(r['event'] == 'open' and bool(r.get('replayed')) for r in rows)} from the journal, "
               f"{sum(r['event'] == 'load' for r in rows)} loaded for live requests after it), "
               f"{sum(r.get('load_seconds', 0) for r in opens if not r.get('replayed')):.0f} s loading")
    out.append("parts: " + (", ".join(parts) or "none started"))
    out.append(f"requests: {len(calls)} ({len(replayed)} answered from the journal, "
               f"{sum(r['seconds'] for r in replayed):.0f} s; {len(calls) - len(replayed)} live, "
               f"{sum(r['seconds'] for r in calls if not r.get('replayed')):.0f} s)"
               + (f"; {sum(not r.get('recorded', True) for r in calls)} not recordable"
                  if any(not r.get("recorded", True) for r in calls) else ""))
    diverged = next((r for r in rows if r["event"] == "diverged"), None)
    if diverged:
        out.append(f"replay stopped early after {diverged['replayed_calls']} requests: expected "
                   f"{diverged['expected']}, the round made a {diverged['found']}")
    out.append(f"ended: {end['outcome']} at {end['ended_at']}" + (f" ({end['detail']})" if end.get("detail") else "")
               if end else "no end: the round is running, or its process stopped without one (killed, a crash); "
               "a round resuming it answers every whole line")
    return "\n".join(out)


def _error(recorded):
    """The exception a journaled request raised, raised again on replay: its own class when it is a builtin one, else
    the nearest builtin class it derives from (an HTTPError is an OSError), with its message."""
    import builtins
    for name in recorded["types"]:
        kind = getattr(builtins, name, None)
        if isinstance(kind, type) and issubclass(kind, Exception):
            return kind(recorded["message"])
    return RuntimeError(recorded["message"])


def _class_name(kind):
    return f"{kind.__module__}:{kind.__qualname__}"


def _class(name):
    """The class a journal names, when this process has its module (crit-eval's own, or the adapter's)."""
    import sys
    module, _, qualname = (name or "").partition(":")
    found = sys.modules.get(module)
    for part in qualname.split(".") if found else ():
        found = getattr(found, part, None)
    return found if isinstance(found, type) else None


def _file_sha(path):
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
