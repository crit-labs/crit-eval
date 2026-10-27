"""What the reference keeps between texts that share a prompt: the modeling code's cache captured after a prompt
chunk and restored for the next text, held within a memory budget, and the checks that a restored state is the
one a fresh pass computes."""
from contextlib import contextmanager
from pathlib import Path

from .. import load_json, write_json
from .checkpoints import STAGE_LIMIT, TORCH_DTYPES

# Tokens a text runs per chunk.
PROMPT_CHUNK = 8192


class CacheUnsupported(Exception):
    """The model's forward or its cache cannot be resumed after a stored prompt; the reference scores full passes."""


def _encode(value, path, tensors):
    """A JSON description of cache state `value` (a cache or cache layer attribute at `path`), its tensors put in
    `tensors` under their paths. Tensors, plain values, dtypes, devices, and dicts, lists and tuples of them are
    stored; a class is kept as the fresh cache's own (checked by name). Anything else cannot be resumed."""
    import torch
    if isinstance(value, torch.Tensor):
        tensors[path] = value.detach().to("cpu").clone().contiguous()
        return {"tensor": path}
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"value": value}
    if isinstance(value, torch.dtype):
        return {"dtype": str(value).removeprefix("torch.")}
    if isinstance(value, torch.device):
        return {"device": str(value)}
    if isinstance(value, dict) and all(isinstance(k, (int, str)) for k in value):
        return {"dict": [[k, _encode(v, f"{path}.{k}", tensors)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {type(value).__name__: [_encode(v, f"{path}.{i}", tensors) for i, v in enumerate(value)]}
    if isinstance(value, type):
        return {"class": f"{value.__module__}.{value.__qualname__}"}
    raise CacheUnsupported(f"cache state {path} is a {type(value).__name__}, which the reference cannot store")


def _decode(spec, tensor, current=None):
    """The value `spec` describes, its tensors made by tensor(path); current: the attribute's present value."""
    import torch
    (kind, body), = spec.items()
    if kind == "tensor":
        return tensor(body)
    if kind == "value":
        return body
    if kind == "dtype":
        return getattr(torch, body)
    if kind == "device":
        return torch.device(body)
    if kind == "dict":
        return {k: _decode(v, tensor) for k, v in body}
    if kind in ("list", "tuple"):
        return (list if kind == "list" else tuple)(_decode(v, tensor) for v in body)
    if not isinstance(current, type) or f"{current.__module__}.{current.__qualname__}" != body:
        raise CacheUnsupported(f"the fresh cache holds another class than the stored {body}")
    return current


def _state(obj, path, skip=()):
    """(description, tensors) of every attribute of cache or cache layer `obj` but `skip`."""
    tensors = {}
    return {name: _encode(value, f"{path}{name}", tensors) for name, value in vars(obj).items()
            if name not in skip}, tensors


def _apply(obj, spec, tensor):
    for name, value in spec.items():
        setattr(obj, name, _decode(value, tensor, getattr(obj, name, None)))


class PromptState:
    """The reference's cache after a prompt (the transformers Cache its own forward created, one entry per decoder
    layer), stored in `folder` as one safetensors file per decoder layer and one for the cache's own attributes, in
    FP32 as the reference computed them, with index.json (identity, files and their sha256) written last. Between
    uses a layer's tensors are meta placeholders of their shapes: the forward still reads lengths from them, only the
    layer being run holds its tensors on the device."""

    def __init__(self, folder, identity, ids, memory=None):
        self.folder, self.identity, self.ids, self.memory = Path(folder), identity, ids, memory
        self.cache, self.done, self.files = None, 0, {}

    def ready(self):
        index = self.folder / "index.json"
        if not index.is_file():
            return False
        stored = load_json(index)
        if stored.get("identity") != self.identity:
            return False
        self.files = stored["files"]
        return True

    def _file(self, name):
        return self.folder / name

    def _bytes(self, name):
        """A stored file's bytes, from host memory where it is kept there (_StateMemory), else from disk."""
        held = self.memory.get(self._file(name)) if self.memory is not None else None
        return held if held is not None else self._file(name).read_bytes()

    def _read(self, name):
        """(description, tensors) of a stored file, its bytes checked against the recorded sha256 (a file kept in
        memory has none: it is the bytes written)."""
        import hashlib
        import json
        from safetensors.torch import load
        data = self._bytes(name)
        if self.files.get(name) not in (None, hashlib.sha256(data).hexdigest()):
            raise SystemExit(f"{self._file(name)}: its bytes differ from the sha256 recorded in index.json")
        size = int.from_bytes(data[:8], "little")
        spec = json.loads(json.loads(data[8:8 + size])["__metadata__"]["state"])
        return spec, load(data)

    def _put(self, name, data, sha=None):
        """Store a file's bytes: in host memory while the scoring's budget allows (_StateMemory), else on disk with
        their sha256 (`sha` when the caller knows it)."""
        import hashlib
        path = self._file(name)
        if self.memory is not None and self.memory.put(path, data):
            path.unlink(missing_ok=True)
            self.files[name] = None
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.files[name] = sha or hashlib.sha256(data).hexdigest()

    def _write(self, name, spec, tensors):
        import json
        from safetensors.torch import save
        self._put(name, save(tensors, metadata={"state": json.dumps(spec)}))

    @staticmethod
    def _layer(index):
        return f"layer-{index:03d}.safetensors"

    def _placeholders(self, layer, spec, tensors):
        import torch
        _apply(layer, spec, lambda path: torch.empty(tensors[path].shape, dtype=tensors[path].dtype, device="meta"))

    def adopt(self, cache):
        """Take the cache a forward created (None when it created none)."""
        from transformers.cache_utils import Cache
        if not isinstance(cache, Cache):
            raise CacheUnsupported("the model's forward creates no transformers Cache")
        self.cache = cache

    def restore(self, cache, device):
        """Make `cache`, freshly created by the forward, this prompt's: its own attributes as stored, every layer's
        tensors placeholders until the layer runs."""
        self.adopt(cache)
        spec, tensors = self._read("cache.safetensors")
        _apply(cache, spec, lambda path: tensors[path].to(device))
        self._grow(len([n for n in self.files if n.startswith("layer-")]))
        for index in range(len(cache.layers)):
            name = self._layer(index)
            if name in self.files:
                header = self._header(name)
                self._placeholders(cache.layers[index], *header)

    def _header(self, name):
        """(description, {path: meta tensor}) of a stored layer from its header only."""
        import json
        import torch
        held = self.memory.get(self._file(name)) if self.memory is not None else None
        if held is not None:
            size = int.from_bytes(held[:8], "little")
            header = json.loads(held[8:8 + size])
        else:
            with self._file(name).open("rb") as f:
                size = int.from_bytes(f.read(8), "little")
                header = json.loads(f.read(size))
        spec = json.loads(header.pop("__metadata__")["state"])
        names = {short: name for name, short in TORCH_DTYPES.items()}
        return spec, {path: torch.empty(info["shape"], dtype=getattr(torch, names[info["dtype"]]), device="meta")
                      for path, info in header.items()}

    def _grow(self, count):
        """Layers a cache without a config creates as they are first updated."""
        layers, make = self.cache.layers, getattr(self.cache, "layer_class_to_replicate", None)
        while len(layers) < count and make is not None:
            layers.append(make())
        if len(layers) < count:
            raise CacheUnsupported("the cache holds fewer layers than the model has decoder layers")

    def enter(self, index, device):
        """Before decoder layer `index` runs: its stored tensors (if any) on the device."""
        name = self._layer(index)
        if name in self.files:
            self._grow(index + 1)
            spec, tensors = self._read(name)
            _apply(self.cache.layers[index], spec, lambda path: tensors[path].to(device))

    def leave(self, index, keep):
        """After decoder layer `index` ran: stored back (keep) or dropped, placeholders in its place."""
        self._grow(index + 1)
        layer = self.cache.layers[index]
        if keep:
            spec, tensors = _state(layer, f"{index}.")
            self._write(self._layer(index), spec, tensors)
        else:
            spec, tensors = self._header(self._layer(index))
        self._placeholders(layer, spec, tensors)

    def seal(self):
        """Store the cache's own attributes and the index: the prompt's state is complete."""
        import hashlib
        spec, tensors = _state(self.cache, "cache.", skip=("layers",))
        self._write("cache.safetensors", spec, tensors)
        for name in [name for name, sha in self.files.items() if sha is None]:   # sealed states live on disk
            data = self._bytes(name)
            self.memory.drop(self._file(name))
            self.folder.mkdir(parents=True, exist_ok=True)
            self._file(name).write_bytes(data)
            self.files[name] = hashlib.sha256(data).hexdigest()
        write_json(self.folder / "index.json", {"identity": self.identity, "files": self.files})


class _StateMemory:
    """Prompt-state files (PromptState) kept in host memory instead of on disk while their bytes stay within `limit`
    (shared by every text of one scoring); a file that does not fit goes to disk as before. The bytes are the same
    either way, so the scores are too; memory only saves writing and reading them back."""

    def __init__(self, limit):
        self.limit, self.used, self.blobs = limit, 0, {}

    def get(self, path):
        return self.blobs.get(path)

    def drop(self, path):
        held = self.blobs.pop(path, None)
        if held is not None:
            self.used -= len(held)

    def put(self, path, data):
        """Keep `data` as `path`'s bytes if it fits (replacing what `path` held); False when it goes to disk."""
        self.drop(path)
        if self.used + len(data) > self.limit:
            return False
        self.blobs[path] = data
        self.used += len(data)
        return True

    @staticmethod
    def budget():
        """Bytes of prompt state to keep in memory: half the host memory available now, less the layer staging
        buffer (STAGE_LIMIT); none where the platform does not report it."""
        import os
        try:
            available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (AttributeError, ValueError, OSError):
            return 0
        return max(0, available // 2 - STAGE_LIMIT)


class _Pass:
    """One forward over `ids` streamed through the decoder layers. state: the PromptState it resumes or extends
    (None: no cache); keep: whether the layers' new state is stored (a prompt chunk) or dropped (a reply)."""

    def __init__(self, ids, long, state=None, keep=False):
        self.ids, self.long, self.state, self.keep = ids, long, state, keep


def _hidden(args, kwargs):
    """(slot, hidden states) of a decoder layer call: the first positional argument, else hidden_states."""
    return (0, args[0]) if args else ("hidden_states", kwargs["hidden_states"])


class _OrderUnsafe(Exception):
    """The model's forward or a decoder layer read cache state that the layer-major order holds at another point
    than the chunk-major order does: the prompts run chunk by chunk instead."""


def _fingerprint(value):
    """What identifies a cache attribute and its contents: tensors by identity and version (an in-place change
    bumps it), containers by their items, plain values by value, anything else by identity."""
    import torch
    if isinstance(value, torch.Tensor):
        return ("tensor", id(value), value._version)
    if isinstance(value, dict):
        return ("dict", tuple((key, _fingerprint(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_fingerprint(item) for item in value))
    if value is None or isinstance(value, (bool, int, float, str)):
        return ("value", value)
    return ("object", id(value))


class _Own:
    """A cache's own attributes (all but its layers) at one point: what each pass after it reads."""

    def __init__(self, cache):
        self.order = list(vars(cache))
        self.values = {key: value for key, value in vars(cache).items() if key != "layers"}
        self.mark = _fingerprint(self.values)

    def held_by(self, cache):
        return _fingerprint({key: value for key, value in vars(cache).items() if key != "layers"}) == self.mark

    def apply(self, cache):
        """Make them the cache's again, in their order; their objects must be as they were."""
        if _fingerprint(self.values) != self.mark:
            raise _OrderUnsafe("the cache's own state was changed in place")
        attributes = vars(cache)
        layers = attributes["layers"]
        attributes.clear()
        attributes.update((key, layers if key == "layers" else self.values[key]) for key in self.order)


@contextmanager
def _watched(cache):
    """The indices of the cache's layers whose state (their own attributes, read by any code, their methods' included)
    is read while the block runs; class attributes (is_sliding) are no state. The layers' classes read through a
    recording __getattribute__ for the block's duration (not a subclass: transformers registers cache layer
    subclasses by type)."""
    ids = {id(layer): index for index, layer in enumerate(cache.layers)}
    seen = set()
    classes = {type(layer) for layer in cache.layers}
    saved = {cls: cls.__dict__.get("__getattribute__") for cls in classes}
    reads = {cls: cls.__getattribute__ for cls in classes}

    def recording(read):
        def __getattribute__(self, name):
            index = ids.get(id(self))
            if index is not None and not name.startswith("__"):
                try:
                    own = name in object.__getattribute__(self, "__dict__")
                except AttributeError:   # a layer without a __dict__: every read counts
                    own = True
                if own:
                    seen.add(index)
            return read(self, name)
        return __getattribute__
    for cls in classes:
        cls.__getattribute__ = recording(reads[cls])
    try:
        yield seen
    finally:
        for cls, original in saved.items():
            if original is None:
                del cls.__getattribute__
            else:
                cls.__getattribute__ = original


class _Prompt:
    """A text's state and its chunks in the layer-major order: per decoder layer the chunks it ran (done), and
    the cache's own attributes after the forward handed each chunk to the layers (own). A text whose first `shared`
    chunks are another text's (its leader's: the same ids, positions and attention) runs only the chunks after them:
    its cache starts as the leader's after those chunks, copied layer by layer as the leader leaves each layer
    (StreamingReference._fork, _fork_layer), and its outputs of the shared chunks are the leader's."""

    def __init__(self, state, long, count, leader=None, shared=0, chunk=PROMPT_CHUNK):
        self.state, self.long = state, long
        self.chunks = [state.ids[c:c + chunk] for c in range(0, len(state.ids), chunk)]
        self.done = [0] * count
        self.own = []
        self.leader, self.shared, self.followers = leader, shared, []
        if leader is not None:
            leader.followers.append(self)


def _shared_chunks(chunks, long):
    """Per text (leader index, chunks shared): the earliest text with the same attention whose leading chunks are
    this text's (the most of them, at least one), or (None, 0). A text keeps at least its last chunk to run, and a
    leader is never itself a follower."""
    out = []
    for t, own in enumerate(chunks):
        best = (None, 0)
        for s in range(t):
            if out[s][0] is not None or (s in long) != (t in long):
                continue
            k = 0
            while k < min(len(own) - 1, len(chunks[s])) and own[k] == chunks[s][k]:
                k += 1
            if k > best[1]:
                best = (s, k)
        out.append(best)
    return out


class _Job:
    """One pass of the layer-major order: chunk `chunk` (its ids) of `prompt`. calls: what the forward hands each
    layer; hidden: its hidden states before layer `at` (after the last, its output), on the host; waiting: the
    layers whose state its last capture read too early."""

    def __init__(self, prompt, ids, chunk):
        self.prompt, self.ids, self.chunk, self.long = prompt, ids, chunk, prompt.long
        self.calls = self.hidden = None
        self.at, self.waiting = 0, set()
