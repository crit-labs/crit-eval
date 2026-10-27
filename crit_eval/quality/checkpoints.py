"""The checkpoint the reference reads its weights from: safetensors read in staged spans, PyTorch state dicts and
ONNX exports behind one tensor interface, and the text config the modeling code is built from."""
import math
import re
from pathlib import Path
from types import SimpleNamespace

from ..limits.schemes import PRIMARY, quantization_of, refuse_unread

# torch dtype names -> safetensors dtype names
TORCH_DTYPES = {"bfloat16": "BF16", "float16": "F16", "float32": "F32", "float64": "F64", "float8_e4m3fn": "F8_E4M3",
                "float8_e5m2": "F8_E5M2", "float8_e8m0fnu": "F8_E8M0", "uint8": "U8", "int8": "I8", "int16": "I16",
                "int32": "I32", "uint32": "U32", "int64": "I64", "bool": "BOOL"}
# Readers that gather rows at once: enough to keep an NVMe queue busy with small reads.
GATHER_THREADS = 16
# A decoder layer's stored tensors are read ahead of it in large reads (Checkpoint.stage): neighbours in the file
# closer than STAGE_GAP bytes are read with the bytes between them, in pieces of STAGE_READ bytes by STAGE_THREADS
# readers, into one host buffer of at most STAGE_LIMIT bytes (a larger layer is read tensor by tensor).
STAGE_GAP = 1 << 20
STAGE_READ = 64 << 20
STAGE_THREADS = 4
STAGE_LIMIT = 8 << 30
LAYER_ZERO = re.compile(r"(?:^|(?<=\.))layers\.0\.")


def _read_spans(reads, threads=None, random=True):
    """Fill each (path, offset, out) of `reads` (out a contiguous uint8 numpy array) with the file's bytes at offset,
    by `threads` readers (GATHER_THREADS), each reading its share in file order. random: the files are opened for
    random reads, so the kernel reads no pages around them (Linux; elsewhere the reads are plain)."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    threads = threads or GATHER_THREADS
    by_file = {}
    for path, offset, out in reads:
        by_file.setdefault(path, []).append((offset, out.reshape(-1)))
    jobs = []
    for path, spans in by_file.items():
        spans.sort(key=lambda span: span[0])
        share = max(1, math.ceil(len(spans) / (4 * threads)))
        jobs += [(path, spans[i:i + share]) for i in range(0, len(spans), share)]

    def read(job):
        path, spans = job
        with open(path, "rb", buffering=0) as f:
            if random and hasattr(os, "posix_fadvise"):
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_RANDOM)
            for offset, out in spans:
                f.seek(offset)
                view, done = memoryview(out), 0
                while done < len(view):
                    count = f.readinto(view[done:])
                    if not count:
                        raise OSError(f"{path}: ends before byte {offset + len(view)}")
                    done += count
    if len(jobs) == 1:
        read(jobs[0])
    elif jobs:
        with ThreadPoolExecutor(min(threads, len(jobs))) as pool:
            list(pool.map(read, jobs))


def _view(flat, start, nbytes, dtype, shape):
    """Bytes [start, start + nbytes) of uint8 `flat` as a tensor of `dtype` and `shape`; copied first when they do
    not start at a multiple of the dtype's size, as a view of another dtype requires."""
    piece = flat[start:start + nbytes]
    if start % dtype.itemsize:
        piece = piece.clone()
    return piece.view(dtype).reshape(shape)


def _stored_tensor(checkpoint, key):
    """Whether `key` names a stored tensor, or (for a .weight name) a matrix stored in a quantized layout."""
    if checkpoint.has(key):
        return True
    return key.endswith(".weight") and any(checkpoint.has(key[: -len(".weight")] + s) for s in PRIMARY)


class TensorCheckpoint:
    """A checkpoint keyed by Hugging Face tensor names (has, raw, rows), which dense() decodes to FP32 with the scheme
    its tensors are stored in (crit_eval.limits.schemes). quantization: the config's quantization_config (or MLX's
    quantization) as a dict."""
    where = {}
    quantization = {}

    def has(self, key):
        return key in self.where

    def rows(self, key, index):
        """Rows `index` (a 1-D int64 CPU tensor) of stored tensor `key`, as stored."""
        return self.raw(key)[index]

    def rows_many(self, requests):
        """rows(key, index) for each (key, index) of `requests`."""
        return [self.rows(key, index) for key, index in requests]

    def fit(self, key, tensor, shape):
        """The tensor as the module stores it: Hugging Face checkpoints keep the modeling code's shapes."""
        return tensor

    def prefix(self, names=()):
        """The prefix of the text model's tensors: of the candidates the checkpoint's decoder layer 0 tensors give
        (vision towers and draft layers have their own), the one under which most of `names` (the modeling code's
        parameter names, such as layers.0.mlp.gate.weight) are stored."""
        candidates = {key[: match.start()] for key in self.where for match in [LAYER_ZERO.search(key)] if match}
        names = list(names) or ["layers.0.input_layernorm.weight"]
        scored = sorted(((sum(_stored_tensor(self, p + n) for n in names), p) for p in candidates), reverse=True)
        if not scored or scored[0][0] == 0:
            raise KeyError("cannot find the decoder layer prefix in the checkpoint")
        return scored[0][1]

    def dense(self, base, device):
        """FP32 weight for module path `base` (without the .weight suffix)."""
        from ..limits.schemes import dense
        return dense(self, base, device)

    def embedding(self, base):
        """The embedding table for gathering rows: as stored when it is plain, else decoded to FP32 on the CPU."""
        from ..limits.schemes import scheme
        if scheme(self, base) is None:
            return self.raw(base + ".weight")
        return self.dense(base, "cpu")


class _Substituted(TensorCheckpoint):
    """A view of `checkpoint` in which the tensors of `replace` ({key: tensor}) stand for stored ones (gathered rows,
    or parts joined), so crit_eval.limits.schemes decodes them with the checkpoint's other tensors."""

    def __init__(self, checkpoint, replace):
        self.checkpoint, self.replace, self.quantization = checkpoint, replace, checkpoint.quantization

    def has(self, key):
        return key in self.replace or self.checkpoint.has(key)

    def raw(self, key):
        return self.replace[key] if key in self.replace else self.checkpoint.raw(key)

    def header(self, key):
        if key not in self.replace:
            return self.checkpoint.header(key)
        tensor = self.replace[key]
        return SimpleNamespace(dtype=TORCH_DTYPES[str(tensor.dtype).removeprefix("torch.")], shape=tuple(tensor.shape))


class Checkpoint(TensorCheckpoint):
    """A safetensors checkpoint."""

    def __init__(self, model, quantization=None):
        from crit_core.model import inspect_model
        from safetensors import safe_open
        self.model = Path(model)
        self.quantization = dict(quantization or {})
        self.files = {}
        self.where = {}
        self.info = {}
        # The files and headers Core validates (crit_core.model.inspect_model) before torch reads any tensor.
        for shard in inspect_model(self.model).shards:
            handle = safe_open(str(shard.path), "pt", device="cpu")
            self.files[shard.path.name] = handle
            for key in handle.keys():
                self.where[key] = handle
            for tensor in shard.tensors:
                self.info[tensor.name] = tensor

    def _mapped(self, key, read):
        """read(bytes [rows, row bytes]) over a mapping of tensor `key`'s bytes, viewed as its dtype; None when the
        tensor is empty or of a dtype torch lacks. The file is mapped per call: the pages read stay in the page
        cache, not in this process, and no open mapping keeps the file from being moved or deleted (Windows)."""
        import numpy as np
        import torch
        tensor = self.info.get(key)
        name = next((name for name, short in TORCH_DTYPES.items() if short == getattr(tensor, "dtype", None)), None)
        if tensor is None or name is None or not tensor.nbytes:
            return None
        rows = tensor.shape[0] if tensor.shape else 1
        mapped = np.memmap(tensor.shard, dtype=np.uint8, mode="r", offset=tensor.data_begin,
                           shape=(rows, tensor.nbytes // rows))
        picked = torch.from_numpy(np.array(read(mapped)))
        del mapped
        return picked.view(getattr(torch, name))

    def raw(self, key):
        if key in self.placed:
            return self.placed[key]
        tensor = self._mapped(key, lambda data: data)
        if tensor is None:
            return self.where[key].get_tensor(key)
        return tensor.reshape(self.info[key].shape)

    placed = {}

    def stage(self, keys, buffer=None, pin=False):
        """Read stored tensors `keys` into one uint8 host buffer (`buffer` when it is large enough, else a new one,
        pinned when `pin`), in large reads in file order: (buffer, {key: (start, nbytes, dtype, shape)}). Each run
        of tensors keeps its offset modulo 64 from the file, so the tensors stay as aligned as they are stored.
        place() then puts them on a device for raw()."""
        import torch
        known = set(TORCH_DTYPES.values())
        keys = [key for key in keys if key in self.info and self.info[key].nbytes and self.info[key].dtype in known]
        runs = []
        for key in sorted(keys, key=lambda k: (str(self.info[k].shard), self.info[k].data_begin)):
            tensor = self.info[key]
            if runs and runs[-1][0] == tensor.shard and tensor.data_begin - runs[-1][2] <= STAGE_GAP:
                runs[-1][2] = max(runs[-1][2], tensor.data_begin + tensor.nbytes)
                runs[-1][3].append(key)
            else:
                runs.append([tensor.shard, tensor.data_begin, tensor.data_begin + tensor.nbytes, [key]])
        layout, starts, size = {}, [], 0
        for _, begin, end, members in runs:
            size += (begin - size) % 64
            starts.append(size)
            for key in members:
                tensor = self.info[key]
                name = next(name for name, short in TORCH_DTYPES.items() if short == tensor.dtype)
                layout[key] = (size + tensor.data_begin - begin, tensor.nbytes, getattr(torch, name), tuple(tensor.shape))
            size += end - begin
        if buffer is None or buffer.numel() < size:
            buffer = None
            try:
                buffer = torch.empty(size, dtype=torch.uint8, pin_memory=pin)
            except RuntimeError:   # no page-locked memory to be had: pageable
                buffer = torch.empty(size, dtype=torch.uint8)
        host = buffer.numpy()
        reads = [(shard, begin + offset, host[start + offset:start + min(end - begin, offset + STAGE_READ)])
                 for (shard, begin, end, _), start in zip(runs, starts)
                 for offset in range(0, end - begin, STAGE_READ)]
        _read_spans(reads, STAGE_THREADS, random=False)
        return buffer, layout

    def place(self, buffer, layout, device):
        """Make raw() read the staged tensors from `device`: the buffer copied there in one transfer (a copy on the
        CPU too, so the buffer can be read into again). Returns the placed copy."""
        size = max((start + nbytes for start, nbytes, _, _ in layout.values()), default=0)
        placed = buffer[:size].to(device, copy=True)
        self.placed = {key: _view(placed, *where) for key, where in layout.items()}
        return placed

    def rows(self, key, index):
        return self.rows_many([(key, index)])[0]

    def rows_many(self, requests):
        """Rows read from the files: each distinct row once, with a read of exactly its bytes (consecutive rows in
        one read), the reads of every request issued together by GATHER_THREADS readers. A lookup touches rows
        scattered over a table far larger than memory; faulting them in through a mapping reads one page at a time
        and around it, the readers keep the device's queue full and the file is read only where asked."""
        import numpy as np
        import torch
        plans, reads = [], []
        for key, index in requests:
            tensor = self.info.get(key)
            name = next((name for name, short in TORCH_DTYPES.items() if short == getattr(tensor, "dtype", None)), None)
            if tensor is None or name is None or not tensor.nbytes or not tensor.shape:
                plans.append(None)
                continue
            unique, inverse = np.unique(index.numpy(), return_inverse=True)
            width = tensor.nbytes // tensor.shape[0]
            out = np.empty((len(unique), width), dtype=np.uint8)
            cuts = np.flatnonzero(np.diff(unique) != 1) + 1
            for start, stop in zip([0, *cuts.tolist()], [*cuts.tolist(), len(unique)]):
                if stop > start:
                    reads.append((tensor.shard, tensor.data_begin + int(unique[start]) * width, out[start:stop]))
            plans.append((out, inverse, name, tensor.shape))
        _read_spans(reads)
        found = []
        for (key, index), plan in zip(requests, plans):
            if plan is None:
                found.append(super().rows(key, index))
                continue
            out, inverse, name, shape = plan
            if not len(index):
                found.append(torch.empty((0, *shape[1:]), dtype=getattr(torch, name)))
                continue
            picked = torch.from_numpy(out).view(getattr(torch, name))
            found.append(picked[torch.from_numpy(inverse.reshape(-1))].reshape(len(index), *shape[1:]))
        return found

    def header(self, key):
        part = self.where[key].get_slice(key)
        return SimpleNamespace(dtype=part.get_dtype(), shape=tuple(part.get_shape()))


class StateDictCheckpoint(TensorCheckpoint):
    """PyTorch state dicts (crit_core.model.TorchCheckpoint), memory-mapped by torch.load once Core has read
    their pickles without running them; weights_only keeps torch's own loader to tensors as well."""

    def __init__(self, weights, quantization=None):
        import torch
        self.quantization = dict(quantization or {})
        self.where = {}
        for shard in weights.shards:
            state = torch.load(shard.path, map_location="cpu", weights_only=True, mmap=True)
            for tensor in shard.tensors:
                self.where[tensor.name] = state[tensor.name]

    def raw(self, key):
        return self.where[key]

    def header(self, key):
        tensor = self.where[key]
        return SimpleNamespace(dtype=TORCH_DTYPES[str(tensor.dtype).removeprefix("torch.")], shape=tuple(tensor.shape))


class OnnxCheckpoint(TensorCheckpoint):
    """An ONNX graph's weights (crit_core.model.OnnxModel) under Hugging Face names, read from the bytes Core
    located; a weight stored as its transpose (a MatMul's [in, out] operand) is transposed back."""

    def __init__(self, weights, quantization=None):
        self.quantization = dict(quantization or {})
        self.where = {tensor.name: tensor for tensor in weights.tensors}

    def raw(self, key):
        import numpy as np
        import torch
        tensor = self.where[key]
        values = np.fromfile(tensor.shard, dtype=np.uint8, count=tensor.nbytes, offset=tensor.data_begin)
        dtype = getattr(torch, next(name for name, short in TORCH_DTYPES.items() if short == tensor.dtype))
        out = torch.from_numpy(values.copy()).view(dtype).reshape(tensor.shape)
        return out.T.contiguous() if tensor.transposed else out

    def header(self, key):
        tensor = self.where[key]
        shape = tuple(reversed(tensor.shape)) if tensor.transposed else tuple(tensor.shape)
        return SimpleNamespace(dtype=tensor.dtype, shape=shape, attributes=tensor.attributes)


def open_checkpoint(model, config):
    """The reference's checkpoint for the model (crit_core.model.open_model): its safetensors (Checkpoint), its
    PyTorch state dicts (StateDictCheckpoint) or its GGUF (GgufCheckpoint). config: transformers' text config."""
    from crit_core.model import open_model
    source = open_model(model)
    if source.format == "gguf":
        from ..architectures.gguf_layout import GgufCheckpoint
        return GgufCheckpoint(source.weights, config)
    quantization = quantization_of(source.config())
    refuse_unread(quantization)
    if source.format == "torch":
        return StateDictCheckpoint(source.weights, quantization)
    if source.format == "onnx":
        return OnnxCheckpoint(source.weights, quantization)
    if source.format != "safetensors":
        raise SystemExit(f"{model}: no weight files the reference reads (safetensors, GGUF, PyTorch or ONNX)")
    return Checkpoint(source.path, quantization)


def _text_config(model):
    """transformers' own config object, which its modeling classes need, from the model's config.json (the
    folder's own, or the one a GGUF derives); it must first pass Core's rules (crit_core.model.ModelSource.config),
    as it did when the suite was prepared."""
    import json
    from crit_core.model import CONFIG_NAME, open_model
    from transformers import AutoConfig, CONFIG_MAPPING
    source = open_model(model)
    source.config()
    if CONFIG_NAME in source.derived:
        data = json.loads(source.files.read(CONFIG_NAME))
        if data.get("model_type") not in CONFIG_MAPPING:
            raise SystemExit(f"{model}: the config derived from the GGUF names no transformers model type "
                             f"({data.get('model_type')!r}); the reference needs config rules for its architecture "
                             "(crit_core.model.gguf_files) or the base model's config.json beside it")
        config = CONFIG_MAPPING[data["model_type"]].from_dict(data)
    else:
        config = AutoConfig.from_pretrained(str(source.folder or model))
    text = getattr(config, "text_config", None) or config
    text._attn_implementation = "eager"
    # The modeling code's own per-expert loop (FP32 matmuls), not a grouped GEMM kernel.
    text._experts_implementation = "eager"
    return text
