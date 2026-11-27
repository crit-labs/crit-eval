"""The fixed quality texts and their reference scores (protocol quality), read without torch.

Every quality text is its prompt_ids followed by its continuation_ids (crit_eval.quality.prepare). `crit eval reference
--model --data` scores each text at every position once and stores the scores under the data (STORE): one
safetensors file per text (`target` [n - 1] float64, the log-probability of each id after the first; `top_ids`
[n - 1, top_k] int64 and `top_logprobs` [n - 1, top_k] float64, the reference's ranked top_k) and index.json,
written last, with the identity they were scored under (the reference's, top_k, valid_vocab and the sha256 of
every text's ids) and each file's sha256. A run reads them here, in pure Python.
"""
import array
import hashlib
import json
import struct
import sys
from pathlib import Path

STORE = "reference"
# The quality modes a run judges in (protocol quality.modes): how much output drift from the checkpoint the user
# selected an engine may trade for speed, never how that checkpoint is quantized or stored. Both judge against the
# same scores, the checkpoint's stored weights decoded exactly with FP32 activations. speed: a
# budget eight times preserve's (quality.mode_thresholds), near-tie output differences allowed (forks); preserve: a
# budget that keeps the output close to the checkpoint's. The modes are crit Core's (crit_plugins.exactness), which every
# engine and run states.
# The mode a run judges in when it names none: Core's too.
from crit_plugins.exactness import DEFAULT_QUALITY_MODE, QUALITY_MODES   # noqa: E402 - Core's, in every crit-core it takes


def mode_of(requirements):
    """The quality mode a run's or result's requirements state (requirements.quality_mode), one of QUALITY_MODES;
    a missing or other one is refused: every result states the mode it was judged in."""
    mode = (requirements or {}).get("quality_mode")
    if mode not in QUALITY_MODES:
        raise ValueError(f"quality_mode is {mode!r}: a result states the quality mode it was judged in, "
                         f"{' or '.join(QUALITY_MODES)}")
    return mode


# How far the checkpoint run with its activations quantized as it declares lands from the stored scores, per set and
# path (crit_eval.quality.reference.store_allowance): what the format itself costs, beside the scores. A speed
# run's limits are raised to its factor times it (protocol quality.allowance, crit_eval.run.quality_checks.stored_allowance).
ALLOWANCE = "allowance.json"
# The reference's own spread (protocol quality.spread): how a second run of the long texts under an engine-sized
# perturbation (TF32 matmuls) disagrees with the stored scores, per set and path, with the identity it was computed
# under.
SPREAD = "spread.json"
# The checkpoint's 16-bit baseline (protocol quality.baseline, crit_eval.quality.reference.store_baseline): how far the
# texts scored in its own 16-bit dtype land from the stored scores, per set and path with its bootstrap resamples, which
# every run reports against its own limits (reported, not judged).
BASELINE = "baseline.json"
# The canaries (protocol quality.canaries, crit_eval.quality.reference.store_canaries): how far each deliberately
# broken forward lands from the stored scores, per set and path, which every run judges against its own limits to
# report whether they would pass (reported, not judged).
CANARIES = "canaries.json"
DTYPES = {"F64": "d", "F32": "f", "I64": "q", "I32": "i"}


def texts(prepared):
    """{text id: (ids, long)} of the prepared quality texts; the long ones use the reference's chunked attention."""
    return {t["id"]: (list(t["prompt_ids"]) + list(t["continuation_ids"]), t["set"] != "conversations")
            for t in prepared["quality"]}


def identity(texts_by_id, top_k, valid_vocab):
    """What the stored scores depend on besides the reference itself."""
    return {"top_k": top_k, "valid_vocab": valid_vocab,
            "texts": {text_id: hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                      for text_id, (ids, _) in texts_by_id.items()}}


def read_safetensors(data):
    """{name: (shape, values)} of a safetensors file's bytes, values flat in row-major order."""
    size = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8:8 + size])
    header.pop("__metadata__", None)
    out = {}
    for name, info in header.items():
        begin, end = info["data_offsets"]
        values = array.array(DTYPES[info["dtype"]])
        values.frombytes(data[8 + size + begin:8 + size + end])
        if sys.byteorder != "little":
            values.byteswap()
        out[name] = (info["shape"], values)
    return out


def text_file(text_id):
    return f"text-{hashlib.sha256(text_id.encode()).hexdigest()[:16]}.safetensors"


def safetensors(tensors, metadata=None):
    """The bytes of a safetensors file of {name: (dtype, shape, values)}, values flat and row-major."""
    header, blobs, offset = {}, [], 0
    for name, (dtype, shape, values) in tensors.items():
        data = array.array(DTYPES[dtype], values)
        if sys.byteorder != "little":
            data.byteswap()
        blob = data.tobytes()
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    if metadata:
        header["__metadata__"] = metadata
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs)


def write(data, identity_value, scores, store=STORE):
    """Store scores ({text id: (target [n - 1], top ids [n - 1][k], top logprobs [n - 1][k])}, sequences of
    numbers) under prepared data folder `data` (in `store`): one file per text, then
    index.json with `identity_value` and each file's sha256. Returns the bytes written."""
    folder = Path(data) / store
    folder.mkdir(parents=True, exist_ok=True)
    files, size = {}, 0
    for text_id, (target, top, values) in scores.items():
        k = len(top[0]) if len(top) else 0
        raw = safetensors({"target": ("F64", [len(target)], target),
                           "top_ids": ("I64", [len(top), k], [t for row in top for t in row]),
                           "top_logprobs": ("F64", [len(values), k], [v for row in values for v in row])},
                          {"text": text_id})
        (folder / text_file(text_id)).write_bytes(raw)
        files[text_id] = [text_file(text_id), hashlib.sha256(raw).hexdigest()]
        size += len(raw)
    temporary = folder / "index.json.tmp"
    temporary.write_text(json.dumps({"identity": identity_value, "files": files}, indent=1), encoding="utf-8")
    temporary.replace(folder / "index.json")
    return size


def write_beside(data, name, identity_value, body, store=STORE):
    """Store `body` (a dict) with the identity it was measured under as file `name` beside the reference scores of
    prepared data folder `data` (spread, allowance, baseline, canaries), written whole or not at all."""
    folder = Path(data) / store
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / (name + ".tmp")
    temporary.write_text(json.dumps({"identity": identity_value, **body}, indent=1), encoding="utf-8")
    temporary.replace(folder / name)


def load_beside(data, name, expected, store=STORE):
    """The body of file `name` beside the reference scores of `data` (write_beside) whose identity holds every key of
    `expected`, without the identity; None when none is stored or it is another identity's."""
    path = Path(data) / store / name
    if not path.is_file():
        return None
    stored = json.loads(path.read_text(encoding="utf-8"))
    found = stored.pop("identity", None) or {}
    if any(found.get(key) != value for key, value in expected.items()):
        return None
    return stored


def write_spread(data, identity_value, sets, store=STORE):
    """Store the reference's spread ({"SET/path": text_quality row}) under prepared data folder `data`."""
    write_beside(data, SPREAD, identity_value, {"sets": sets}, store)


def load_spread(data, expected, store=STORE):
    """The reference's spread stored under `data` whose identity holds every key of `expected`: {"SET/path": row};
    None when none is stored or it is another identity's."""
    stored = load_beside(data, SPREAD, expected, store)
    return None if stored is None else stored["sets"]


def write_allowance(data, identity_value, allowance):
    """Store the allowance ({"activations", "sets": {"SET/path": text_quality row}, ...}: what the checkpoint's declared
    activation quantization costs) beside the reference scores of prepared data folder `data`."""
    write_beside(data, ALLOWANCE, identity_value, allowance)


def load_allowance(data, expected):
    """The allowance stored under `data` whose identity holds every key of `expected` ({"activations", "sets", ...},
    without the identity); None when none is stored or it is another identity's."""
    return load_beside(data, ALLOWANCE, expected)


def load(data, expected, store=STORE):
    """The reference scores stored under prepared data folder `data` (in `store`) whose identity holds every key of
    `expected`: {text id: [{"logprob", "top_ids", "top_logprobs"} per position after the first]}; None when the
    store is missing or another identity's. A file whose bytes differ from its recorded sha256 is refused."""
    folder = Path(data) / store
    index = folder / "index.json"
    if not index.is_file():
        return None
    stored = json.loads(index.read_text(encoding="utf-8"))
    found = stored.get("identity") or {}
    if any(found.get(key) != value for key, value in expected.items()):
        return None
    out = {}
    for text_id, (name, sha) in stored["files"].items():
        raw = (folder / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise SystemExit(f"{folder / name}: its bytes differ from the sha256 recorded in index.json")
        tensors = read_safetensors(raw)
        (count,), target = tensors["target"]
        (_, k), ids = tensors["top_ids"]
        _, values = tensors["top_logprobs"]
        out[text_id] = [{"logprob": target[i], "top_ids": list(ids[i * k:(i + 1) * k]),
                         "top_logprobs": list(values[i * k:(i + 1) * k])} for i in range(count)]
    return out
