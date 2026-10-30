"""The quality reference: `crit eval reference --model DIR --data DIR`.

The reference is the checkpoint the data was prepared from, the one the user chose. It runs the model's official
Hugging Face `transformers` modeling code with FP32 activations and the stored weights dequantized exactly to FP32.
Decoder layers are streamed onto the GPU one at a time, so a model larger than device memory still runs; each layer's
stored bytes are read in large reads while the previous layer runs and reach the GPU in one transfer, where they are
dequantized. It scores every position of the prepared quality texts (protocol quality: each conversation's prompt
followed by its written answer) teacher-forced, once per model, suite and reference version, and stores each
position's target log-probability and top-k under the data (score_data, crit_eval.quality.fixed); every engine's scores of
the same texts are judged against them. The conversations use eager attention with the masks the modeling code
builds; the long texts the same FP32 arithmetic in blocks of query rows (chunked_attention), with boolean masks only
where the modeling code requires one (sdpa's mask rules), since a 32K-token float mask and score matrix would not
fit; attention with sinks, which has no sdpa form, takes eager's additive masks instead (chunked_sink_attention).
It needs the device to itself: stop the engine's server first.

The model's own forward drives every pass, so whatever its text model does before, between and after the decoder
layers (embedding scales, repeated residual streams, per-layer inputs, position ids, masks, a final norm or mixer)
is its own code. A pass first records the arguments the forward hands each decoder layer, without running it; then
each layer is loaded once and run for every sequence; finally the forward runs again with the last layer's output,
so its own tail produces the hidden states the head scores. The forward must hand each layer the previous layer's
output unchanged (checked). Mixture-of-experts layers use transformers' eager experts loop.

A text runs in chunks of PROMPT_CHUNK tokens that resume the transformers Cache the forward builds, kept per decoder
layer in FP32 scratch files (PromptState) until the text is scored. All chunks run layer-major (_layer_major): each
layer, once loaded, runs every chunk it can, a chunk once the layers whose cache state its forward reads have run
the chunks before it; reads of cache state the chunk-major order (each chunk a round through every layer) would hold
otherwise make the texts run that way, and a model whose cache cannot be resumed runs each text whole.

Checkpoint keys reach the modeling code's parameter names through transformers' own conversion mapping for the
model type (transformers.conversion_mapping): renamings, prefixes and converters such as per-expert matrices merged
into fused expert tensors or a sharded table concatenated. Each source matrix is decoded on its own first, unless the
checkpoint stores the scales of the converted tensor itself (one scale for a table stored in shards), in which case
the stored parts are joined before decoding. Embedding tables are never materialized: a lookup reads only the bytes
of the rows it needs, by parallel reads (Checkpoint.rows_many), and decodes those (plain, FP8 and NVFP4 storage, and
every GGUF block type: GgufCheckpoint.row_source; other schemes decode the table once on the CPU). Tensors outside the
text model (vision towers, draft layers) are never read.

Supported stored formats: safetensors, PyTorch state dicts and ONNX graphs (crit_core.model.open_model), their
weights unquantized or in any scheme of crit_eval.limits.schemes (FP8, NVFP4, compressed-tensors int, AWQ, GPTQ, MLX,
bitsandbytes, ONNX Runtime's MatMulNBits), each decoded to exactly the FP32 values its own library produces.
A GGUF (a lone .gguf, a split set or a folder holding one) is read through crit_eval.architectures.gguf_layout:
every block type gguf-py decodes, with GGUF's reference converter layout undone per architecture, and the
config.json its metadata derives (crit_core.model.gguf_files) unless its folder has the base model's.
Requires torch, transformers and safetensors, gguf for a GGUF and numpy for ONNX. This is the slow, trusted path,
run once per prepared data.

Quality mode as_quantized (protocol quality.allowance): when the checkpoint declares quantized input activations,
the same run scores the texts once more with the activations quantized as the checkpoint's own inference does: every
matrix (and the output head) whose stored scheme declares input activations (crit_eval.limits.schemes.activations:
NVFP4 in groups of 16 with the stored global scale, FP8 per token or per tensor) multiplies its inputs quantized and
dequantized in FP32, the weights as in preserve mode. The KV cache and every other operation stay as in preserve mode.
Only how far that run lands from the stored scores is kept, per set and path (store_allowance, fixed.ALLOWANCE): an
engine that runs the checkpoint as quantized is judged against the exact scores with its limits widened by that drift.
Fused experts (MergeModulelist, then Concatenate along the output rows) quantize each expert matrix's input with that
matrix's own declared scale (_FusedInputs). A matrix with declared activations that the modeling code multiplies any
other way is refused rather than left exact.
"""
import math
import re
import time
from contextlib import contextmanager
from pathlib import Path

from .. import EVAL_DIR, load_json
from ..architectures.indexers import FORM_VERSIONS, exact_forms
from .prepare import PREPARED, model_identity, prepared_template
from .fixed import QUALITY_MODES, STORE
from ..limits.schemes import PRIMARY, SchemeError
from .attention import CHUNKED, CHUNKED_SINKS, _sink_attention, chunked_attention, chunked_sink_attention
from .checkpoints import (STAGE_LIMIT, TORCH_DTYPES, TensorCheckpoint, _Substituted, _stored_tensor, _text_config,
                          open_checkpoint)
from .prompt_state import (PROMPT_CHUNK, CacheUnsupported, PromptState, _Job, _OrderUnsafe, _Own, _Pass, _Prompt,
                           _StateMemory, _apply, _hidden, _shared_chunks, _state, _watched)

# Changes whenever what the reference computes does, so nothing stored by another version is read.
REFERENCE_VERSION = 3
# Schemes whose stored rows are the table's rows, with scales per tensor or per row (crit_eval.limits.schemes; None is
# plain): an embedding stored so is gathered row by row.
GATHERABLE = (None, "fp8", "nvfp4")
# Positions a fixed text scores through the head at once (a full row of logits per position, in FP64).
SCORE_ROWS = 512
# What a forward that cannot resume a stored cache raises.
RESUME_ERRORS = (RuntimeError, NotImplementedError, TypeError, ValueError, AttributeError)


@contextmanager
def _meta_parameters():
    """Build modules with their parameters on the meta device and their buffers real: the checkpoint supplies
    every parameter, while buffers a module computes (rotary frequencies, an embedding's scale) keep their values.
    torch.empty (how modules allocate parameters before initialising them) allocates nothing either, so a table
    larger than host memory can be declared; a buffer filled from torch.empty stays meta and fails loudly if used."""
    import torch
    from torch.overrides import TorchFunctionMode
    register = torch.nn.Module.register_parameter

    def meta(module, name, param):
        register(module, name, None if param is None else
                 torch.nn.Parameter(param.to("meta"), requires_grad=param.requires_grad))

    class EmptyOnMeta(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = dict(kwargs or {})
            if func is torch.empty and kwargs.get("device") is None:
                kwargs["device"] = "meta"
            return func(*args, **kwargs)
    torch.nn.Module.register_parameter = meta
    try:
        with EmptyOnMeta():
            yield
    finally:
        torch.nn.Module.register_parameter = register


_GATHERED = {}


def _gathering(module, rows):
    """Turn nn.Embedding `module` into one whose lookups call rows(index) -> FP32 rows instead of reading its
    (meta) weight. A subclass's own forward (an embedding scale) still runs: the gathering class sits between it and
    nn.Embedding in the method order."""
    import torch
    if "base" not in _GATHERED:
        class GatheredRows(torch.nn.Embedding):
            def forward(self, input):
                if self.max_norm is not None:
                    raise SchemeError("the reference does not gather rows of a max_norm embedding")
                found = self._crit_rows(input.reshape(-1).to("cpu", torch.int64))
                return found.reshape(*input.shape, found.shape[-1]).to(input.device)
        _GATHERED["base"] = GatheredRows
    base = _GATHERED["base"]
    cls = type(module)
    if not issubclass(cls, base):
        if cls is not torch.nn.Embedding and cls not in _GATHERED:
            _GATHERED[cls] = type(cls.__name__, (cls, base), {})
        module.__class__ = base if cls is torch.nn.Embedding else _GATHERED[cls]
    module._crit_rows = rows


class _StreamedExperts:
    """A fused experts parameter [experts, ...] that never exists whole on the device. The modeling code's eager
    experts loop reads it one expert at a time (self.gate_up_proj[expert_idx]); indexing this decodes that expert's
    own stored matrices and runs them through the parameter's converter (MergeModulelist, then Concatenate), the
    FP32 values the whole converted tensor holds at that index. The experts' stored bytes stay on the device while
    the layer runs; `placeholder` is the meta parameter it stands for, put back when the layer is released."""

    def __init__(self, reference, name, placeholder):
        self.reference, self.name, self.placeholder = reference, name, placeholder
        self.converter, self.parts = reference.converted[name]
        checkpoint, device = reference.checkpoint, reference.device
        self.held = {key: checkpoint.raw(key).to(device).clone() for sources in self.parts.values()
                     for base, _ in sources for key in reference.members.get(base, ())}
        self.shape, self.dtype = tuple(placeholder.shape), reference.torch.float32

    def __getitem__(self, index):
        expert = int(index)
        view = _Substituted(self.reference.checkpoint, self.held)
        inputs = {pattern: [view.dense(sources[expert][0], self.reference.device)]
                  for pattern, sources in self.parts.items()}
        for operation in self.converter.operations:
            inputs = operation.convert(inputs, source_patterns=self.converter.source_patterns,
                                       target_patterns=self.converter.target_patterns, full_layer_name=self.name,
                                       config=self.reference.config)
        (tensor,) = inputs.values()
        tensor = tensor.float()[0]
        if _FusedInputs.active is not None:
            _FusedInputs.active.tag(self, expert, tensor)
        return tensor


class _FusedInputs:
    """Quality mode as_quantized for a fused experts module (StreamingReference._fused_inputs): while its forward
    runs, an expert's matrix read from one of its fused parameters (param[expert], or a _StreamedExperts index) is
    remembered, and a linear with it as the weight multiplies each block of output rows by the input quantized as
    that block's stored matrix declares (crit_eval.limits.schemes.quantize_activations), the rows joined in order."""
    active = None

    def __init__(self, module, fused):
        self.module, self.fused, self.tags = module, fused, {}

    def tag(self, source, expert, tensor):
        attribute = next((name for name in self.fused if getattr(self.module, name, None) is source), None)
        if attribute is not None:
            self.tags[id(tensor)] = (tensor, self.fused[attribute][expert])

    def linear(self, x, weight, bias):
        import torch
        from ..limits.schemes import quantize_activations
        _, parts = self.tags.pop(id(weight))
        out, row = [], 0
        for _, rows, declared in parts:
            block = weight[row: row + rows]
            out.append(torch.nn.functional.linear(quantize_activations(x, *declared) if declared else x, block,
                                                  None if bias is None else bias[row: row + rows]))
            row += rows
        return torch.cat(out, dim=-1)


def _quantize_fused_inputs(module, fused):
    """Run `module`'s forward with _FusedInputs for its fused parameters `fused` (set once per module)."""
    import torch
    from torch.overrides import TorchFunctionMode
    forward = module.forward

    class Mode(TorchFunctionMode):
        def __init__(self, inputs):
            super().__init__()
            self.inputs = inputs

        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func is torch.Tensor.__getitem__ and isinstance(args[0], torch.nn.Parameter):
                out = func(*args, **kwargs)
                if isinstance(args[1], (int, torch.Tensor)) and (not isinstance(args[1], torch.Tensor)
                                                                 or args[1].numel() == 1):
                    self.inputs.tag(args[0], int(args[1]), out)
                return out
            if func is torch.nn.functional.linear:
                weight = args[1] if len(args) > 1 else kwargs.get("weight")
                if id(weight) in self.inputs.tags and self.inputs.tags[id(weight)][0] is weight:
                    bias = args[2] if len(args) > 2 else kwargs.get("bias")
                    return self.inputs.linear(args[0] if args else kwargs["input"], weight, bias)
            return func(*args, **kwargs)

    def quantized(*args, **kwargs):
        inputs = _FusedInputs(module, fused)
        previous, _FusedInputs.active = _FusedInputs.active, inputs
        try:
            with Mode(inputs):
                return forward(*args, **kwargs)
        finally:
            _FusedInputs.active = previous

    module.forward = quantized


class _Captured(Exception):
    """Raised by the last decoder layer's stand-in once a forward has handed every layer its arguments."""


class StreamingReference:
    # Texts whose leading chunks are the same (the long quality texts share their document) run them once in the
    # layer-major order (_Prompt); the rows are the same bytes as running each text on its own.
    share_prefixes = True
    # Bytes of prompt state score_texts keeps in host memory instead of scratch files (_StateMemory); None: half the
    # memory available when it starts, less the staging buffer.
    state_memory = None
    # Fused expert parameters decoded expert by expert as the forward reads them (_StreamedExperts): None where one
    # would take more than half the device's free memory in FP32, True for every one, False for none.
    stream_experts = None

    def __init__(self, model, device="cuda", quality_mode="preserve"):
        import os
        if quality_mode not in QUALITY_MODES:
            raise SystemExit(f"quality mode must be one of {', '.join(QUALITY_MODES)}")
        self.quality_mode = quality_mode
        # as_quantized: {checkpoint base: crit_eval.limits.schemes.activations} as found, and the output head's
        self.inputs, self.head_inputs = {}, None
        # Chunked attention allocates a score block of another width for every block of query rows; torch's default
        # segments then fragment until the device is full (a 32K-token text on a 31B model), while expandable
        # segments reuse the memory. Read when the process first allocates on the device; arithmetic is unchanged.
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        import torch
        from transformers import AutoModel
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.torch, self.device = torch, device
        self.config = _text_config(model)
        self.checkpoint = open_checkpoint(model, self.config)
        with _meta_parameters():
            self.skeleton = AutoModel.from_config(self.config)
        # Faster forms of modeling-code loops, bound where the code is the one they were checked against
        # (crit_eval.architectures.indexers): the same selection, without a Python loop per token.
        self.forms = exact_forms(self.skeleton)
        # The buffers each decoder layer computes when built (rotary frequencies a layer holds itself, as attention
        # compressors do): a used layer is released to meta, and its next load puts them back (_restore).
        self.built = [{name: buffer for name, buffer in layer.named_buffers() if not buffer.is_meta}
                      for layer in self.skeleton.layers]
        self.names = dict(self.skeleton.state_dict(keep_vars=True))
        self.prefix = self.checkpoint.prefix(self.names)
        self.configs = list({id(c): c for m in self.skeleton.modules() for c in [getattr(m, "config", None)]
                             if c is not None and hasattr(c, "_attn_implementation")}.values())
        # The long texts' attention: chunked_attention with sdpa's masks, or for attention with sinks
        # chunked_sink_attention (the modeling code's eager attention in blocks of rows) with eager's.
        self.chunked = CHUNKED_SINKS if _sink_attention(self.skeleton) else CHUNKED
        self.converted, self.members = self._conversions()
        self.renamed = self._renamed()
        self.tables = {}
        # Decoder layers' stored tensors by layer, and the read of the next layer under way (Checkpoint.stage).
        self.layer_keys, self.ahead, self.reader, self.host = None, None, None, None

    def _conversions(self):
        """({parameter name: (converter, {source pattern: [(base, stored key), ...]})}, {base: [stored keys]}): the
        checkpoint's matrices that transformers' conversion mapping for the model type turns into a parameter of
        another name or shape (only for checkpoints keyed by Hugging Face names; a GGUF is presented in the
        modeling code's layout already)."""
        if not isinstance(self.checkpoint, TensorCheckpoint):
            return {}, {}
        members = {}
        for key in self.checkpoint.where:
            members.setdefault(key.rsplit(".", 1)[0], []).append(key)
        from transformers.conversion_mapping import get_model_conversion_mapping
        from transformers.core_model_loading import WeightConverter, WeightRenaming, dot_natural_key, rename_source_key
        transforms = get_model_conversion_mapping(self.skeleton)
        renamings = [t for t in transforms if isinstance(t, WeightRenaming)]
        converters = [t for t in transforms if isinstance(t, WeightConverter)]
        found = {}
        if not converters:
            return found, members
        for key in self.checkpoint.where:
            suffix = next((s for s in PRIMARY if key.endswith(s)), None)
            if suffix is None:
                continue
            base = key[: -len(suffix)]
            renamed, pattern = rename_source_key(base + ".weight", renamings, converters,
                                                 self.skeleton.base_model_prefix, self.names)
            if pattern is None or renamed not in self.names:
                continue
            converter = next(c for c in converters if pattern in c.source_patterns)
            found.setdefault(renamed, (converter, {}))[1].setdefault(pattern, []).append((base, key))
        for _, parts in found.values():
            for pattern in parts:
                parts[pattern].sort(key=lambda part: dot_natural_key(part[1]))
        return found, members

    def _renamed(self):
        """{modeling-code name: stored key} of the checkpoint's tensors that transformers' renamings for the model
        type give another name (a checkpoint in the model's own original layout): a matrix under its
        .weight name in whatever layout it is stored. Read only for a name not stored under the prefix (_key)."""
        if not isinstance(self.checkpoint, TensorCheckpoint):
            return {}
        from transformers.conversion_mapping import get_model_conversion_mapping
        from transformers.core_model_loading import WeightRenaming, rename_source_key
        renamings = [t for t in get_model_conversion_mapping(self.skeleton) if isinstance(t, WeightRenaming)]
        found = {}
        for key in self.checkpoint.where if renamings else ():
            suffix = next((s for s in PRIMARY if key.endswith(s)), None)
            source = key[: -len(suffix)] + ".weight" if suffix else key
            renamed, _ = rename_source_key(source, renamings, [], self.skeleton.base_model_prefix, self.names)
            if renamed != source:
                found.setdefault(renamed, source)
        return found

    def _key(self, name, prefix=None):
        """The stored key of modeling-code name `name` (for a matrix, its .weight name in any stored layout): under
        the checkpoint's prefix (`prefix` when given), else where transformers' renamings find it."""
        key = (self.prefix if prefix is None else prefix) + name
        if name in self.renamed and not _stored_tensor(self.checkpoint, key):
            return self.renamed[name]
        return key

    def _target(self, name):
        """The checkpoint base the converted parameter `name` would have if it were stored whole."""
        return self.prefix + (name[: -len(".weight")] if name.endswith(".weight") else name)

    def _convert(self, name, device):
        """Parameter `name` in FP32 from its sources through its converter's operations. Sources are decoded one by
        one, unless the checkpoint stores scales for the converted tensor itself: then they are its stored parts,
        joined before decoding with those scales."""
        from ..limits.schemes import dense
        converter, parts = self.converted[name]
        target = self._target(name)
        joined = target in self.members
        inputs = {pattern: [self.checkpoint.raw(key).to(device) if joined else self.checkpoint.dense(base, device)
                            for base, key in sources] for pattern, sources in parts.items()}
        for operation in converter.operations:
            inputs = operation.convert(inputs, source_patterns=converter.source_patterns,
                                       target_patterns=converter.target_patterns, full_layer_name=name,
                                       config=self.config)
        (tensor,) = inputs.values()
        if joined:
            suffix = next(key[len(base):] for base, key in next(iter(parts.values())))
            tensor = dense(_Substituted(self.checkpoint, {target + suffix: tensor}), target, device)
        return tensor.float()

    def _tensor(self, name, value):
        """Parameter or persistent buffer `name` (a modeling-code name) on the device: floating point in FP32, other
        buffers in their own dtype; None for a buffer the checkpoint does not store."""
        if name in self.converted:
            tensor = self._convert(name, self.device)
        else:
            key = self._key(name)
            if name.rsplit(".", 1)[-1] == "weight":
                tensor = self.checkpoint.dense(key[: -len(".weight")], self.device)
            elif self.checkpoint.has(key) or isinstance(value, self.torch.nn.Parameter):
                tensor = self.checkpoint.raw(key).to(self.device)
                tensor = tensor.float() if tensor.is_floating_point() else tensor.to(value.dtype)
            else:
                return None
            tensor = self.checkpoint.fit(key, tensor, value.shape)
        return tensor

    def _row_keys(self, base, primary, full_rows):
        """The stored keys whose rows make rows of matrix `base` (stored key `primary`): it and its per-row scales
        (scales per tensor are read whole)."""
        keys = [primary]
        for key in self.members.get(base, ()):
            shape = tuple(self.checkpoint.header(key).shape)
            if key == primary or math.prod(shape) <= 1:
                continue
            if shape[0] != full_rows:
                raise SchemeError(f"{key}: scales not per row or per tensor; cannot gather rows of {base}")
            keys.append(key)
        return keys

    def _decode_rows(self, base, gathered):
        """FP32 rows of matrix `base` on the device from its gathered stored rows ({key: rows})."""
        from ..limits.schemes import dense
        return dense(_Substituted(self.checkpoint, gathered), base, self.device)

    def _gather_rows(self, base, primary, index, full_rows, rows=None):
        """FP32 rows `index` of matrix `base` (stored key `primary`), on the device: its stored rows (or `rows`,
        already gathered) decoded with its scales, per-row scales gathered alike."""
        keys = self._row_keys(base, primary, full_rows)
        wanted = keys if rows is None else keys[1:]
        gathered = dict(zip(wanted, self.checkpoint.rows_many([(key, index) for key in wanted])))
        if rows is not None:
            gathered[primary] = rows
        return self._decode_rows(base, gathered)

    def _primary(self, base):
        return next((base + s for s in PRIMARY if self.checkpoint.has(base + s)), None)

    def _gatherable(self, base, primary, rows=None):
        """Whether matrix `base` is stored as rows of the table (GATHERABLE schemes)."""
        from ..limits.schemes import scheme
        if primary is None:
            return False
        view = self.checkpoint if rows is None else _Substituted(self.checkpoint, {primary: rows})
        return scheme(view, base) in GATHERABLE

    def _row_source(self, name):
        """rows(index) -> FP32 rows of the embedding table parameter `name`, read without materializing the table
        where its storage allows (see the module docstring), else from the table decoded once on the CPU."""
        key = self._key(name)
        base = key[: -len(".weight")]
        if isinstance(self.checkpoint, TensorCheckpoint):
            if name in self.converted:
                source = self._converted_rows(name)
                if source is not None:
                    return source
            else:
                primary = self._primary(base)
                if self._gatherable(base, primary):
                    full = self.checkpoint.header(primary).shape[0]
                    return lambda index: self._gather_rows(base, primary, index, full)
        elif hasattr(self.checkpoint, "row_source"):
            source = self.checkpoint.row_source(key)
            if source is not None:
                return lambda index: source(index).to(self.device)

        def table(index):
            if name not in self.tables:
                self.tables[name] = (self._convert(name, "cpu") if name in self.converted
                                     else self.checkpoint.embedding(base))
            return self.tables[name][index].to(self.device).float()
        return table

    def _converted_rows(self, name):
        """A row source for a table its converter concatenates from parts along its rows, or None."""
        import torch
        from transformers.core_model_loading import Concatenate
        converter, parts = self.converted[name]
        operations = converter.operations
        if len(parts) != 1 or len(operations) != 1 or not isinstance(operations[0], Concatenate) or operations[0].dim:
            return None
        sources = next(iter(parts.values()))
        sizes = [self.checkpoint.header(key).shape[0] for _, key in sources]
        ends = torch.tensor(sizes).cumsum(0)
        target = self._target(name)
        if target in self.members:
            # Parts of one stored tensor, decoded together with the target's own scales.
            first = sources[0][1]
            suffix = first[len(sources[0][0]):]
            header = self.checkpoint.header(first)
            dtype = getattr(torch, next(name for name, short in TORCH_DTYPES.items() if short == header.dtype))
            meta = torch.empty((sum(sizes), *header.shape[1:]), device="meta", dtype=dtype)
            if not self._gatherable(target, target + suffix, meta):
                return None

            def joined(index):
                part = torch.searchsorted(ends, index, right=True)
                picks = [(part == j, j) for j in part.unique().tolist()]
                rows = None
                for (where, j), found in zip(picks, self.checkpoint.rows_many(
                        [(sources[j][1], index[where] - (ends[j] - sizes[j])) for where, j in picks])):
                    if rows is None:
                        rows = torch.empty((len(index), *found.shape[1:]), dtype=found.dtype)
                    rows[where] = found
                return self._gather_rows(target, target + suffix, index, sum(sizes), rows)
            return joined
        if not all(self._gatherable(base, key) for base, key in sources):
            return None

        def each(index):
            part = torch.searchsorted(ends, index, right=True)
            pieces = []
            for j in part.unique().tolist():
                where = part == j
                base, key = sources[j]
                pieces.append((where, base, self._row_keys(base, key, sizes[j]), index[where] - (ends[j] - sizes[j])))
            found = iter(self.checkpoint.rows_many([(key, local) for _, _, keys, local in pieces for key in keys]))
            out = None
            for where, base, keys, _ in pieces:
                rows = self._decode_rows(base, {key: next(found) for key in keys})
                if out is None:
                    out = torch.empty((len(index), *rows.shape[1:]), dtype=rows.dtype, device=rows.device)
                out[where.to(rows.device)] = rows
            return out
        return each

    def _gather_embeddings(self, module, path):
        """Make every embedding in `module` (at modeling-code path `path`) gather rows; returns their weight names
        relative to `module`, which stay meta."""
        names = set()
        for sub, child in module.named_modules():
            if isinstance(child, self.torch.nn.Embedding):
                weight = (sub + "." if sub else "") + "weight"
                _gathering(child, self._row_source(path + weight))
                names.add(weight)
        return names

    def _load(self, module, path):
        """Load `module` (modeling-code path `path`) from the checkpoint onto the device: parameters and the
        persistent buffers the checkpoint stores, weights dequantized to FP32; embeddings gather rows."""
        gathered = self._gather_embeddings(module, path)
        streamed = self._streamed(module, path)
        state = {}
        for name, value in module.state_dict(keep_vars=True).items():
            tensor = None if name in gathered or name in streamed else self._tensor(path + name, value)
            state[name] = value if tensor is None else tensor
        module.load_state_dict(state, strict=True, assign=True)
        self._place(module)
        if self.quality_mode == "as_quantized":
            self._quantize_inputs(module, path)
        for name, experts in streamed.items():
            owner, _, attribute = name.rpartition(".")
            owner = module.get_submodule(owner) if owner else module
            del owner._parameters[attribute]
            owner.__dict__[attribute] = experts
        self._config_dt_limit(module)
        return module

    def declared_activations(self):
        """The input activation formats the checkpoint declares for the text model's matrices and the output head
        (crit_eval.limits.schemes.activations), sorted; empty for a checkpoint that declares none (weight-only schemes,
        plain weights, a GGUF). SchemeError for a declared form crit does not emulate."""
        head = self._key("lm_head.weight", "")[: -len(".weight")]
        return sorted({declared[0] for base in self.members if base.startswith(self.prefix) or base == head
                       for declared in [self._activations(base)] if declared})

    def _activations(self, base):
        from ..limits.schemes import activations
        if base not in self.inputs:
            self.inputs[base] = activations(self.checkpoint, base)
        return self.inputs[base]

    def _quantize_inputs(self, module, path):
        """Quality mode as_quantized: the nn.Linear matrices of `module` (modeling-code path `path`) whose checkpoint
        declares input activations quantize their inputs as it declares (crit_eval.limits.schemes.quantize_activations), by
        a forward pre-hook set once per module, and fused experts per expert matrix (_fused_inputs). In a decoder layer,
        every stored matrix with declared activations must be one of them: one the modeling code multiplies otherwise
        (a converted parameter of another form) is refused."""
        from ..limits.schemes import quantize_activations
        hooked = set()
        for sub, child in module.named_modules():
            if not isinstance(child, self.torch.nn.Linear):
                continue
            name = path + (sub + "." if sub else "") + "weight"
            if name in self.converted:
                continue
            base = self._key(name)[: -len(".weight")]
            declared = self._activations(base)
            hooked.add(base)
            if declared and getattr(child, "_crit_inputs", None) != declared:
                if getattr(child, "_crit_hook", None) is not None:
                    child._crit_hook.remove()
                kind, scale = declared
                child._crit_hook = child.register_forward_pre_hook(
                    lambda _, args, kind=kind, scale=scale: (quantize_activations(args[0], kind, scale),) + tuple(args[1:]))
                child._crit_inputs = declared
        for sub, child in module.named_modules():
            if getattr(child, "_crit_fused", None) is None:
                fused = self._fused_inputs(child, path + (sub + "." if sub else ""))
                if fused:
                    _quantize_fused_inputs(child, fused)
                child._crit_fused = fused
            hooked.update(base for specs in (child._crit_fused or {}).values() for parts in specs
                          for base, _, _ in parts)
        if path:
            prefix = self.prefix + path
            missed = sorted(base for base in self.members if base.startswith(prefix) and base not in hooked
                            and self._activations(base))
            if missed:
                raise SchemeError(f"as_quantized: {', '.join(missed[:3])} declare quantized input activations, but the "
                                  "modeling code multiplies them other than through an nn.Linear (fused experts or a "
                                  "converted parameter); crit does not emulate their activations")

    def _fused_inputs(self, module, path):
        """Quality mode as_quantized: {attribute: per expert [(base, rows, activations), ...] in output-row order} for
        `module`'s own fused expert parameters (modeling-code path `path`) whose stored matrices declare input
        activations: the converter only stacks the experts' matrices (MergeModulelist along dim 0) and joins them
        along the output rows (Concatenate along dim 1), so each output row of expert e comes from one stored matrix
        and multiplies the input quantized as that matrix declares. Empty when none declares activations."""
        found = {}
        for attribute, value in module.named_parameters(recurse=False):
            name = path + attribute
            if name not in self.converted or value.dim() != 3 or self._target(name) in self.members:
                continue
            converter, parts = self.converted[name]
            operations = converter.operations
            if not operations or type(operations[0]).__name__ != "MergeModulelist" or operations[0].dim != 0 \
                    or any(type(op).__name__ != "Concatenate" or op.dim != 1 for op in operations[1:]) \
                    or any(len(sources) != value.shape[0] for sources in parts.values()):
                if any(self._activations(base) for sources in parts.values() for base, _ in sources):
                    raise SchemeError(f"as_quantized: {name} joins matrices that declare quantized input activations "
                                      "in a way crit does not emulate")
                continue
            patterns = [pattern for pattern in converter.source_patterns if pattern in parts]
            specs = [[(parts[pattern][expert][0], int(self.checkpoint.header(parts[pattern][expert][1]).shape[0]),
                       self._activations(parts[pattern][expert][0])) for pattern in patterns]
                     for expert in range(value.shape[0])]
            if sum(rows for _, rows, _ in specs[0]) != value.shape[1]:
                raise SchemeError(f"as_quantized: {name}: its stored matrices' rows do not make its {value.shape[1]}")
            if any(declared for parts_ in specs for _, _, declared in parts_):
                found[attribute] = specs
        return found

    def _streamed(self, module, path):
        """{name: _StreamedExperts} of `module`'s fused expert parameters to decode expert by expert: those whose
        converter only stacks the experts' own matrices and joins them (MergeModulelist, then Concatenate) and that
        would take more than half the device's free memory in FP32 (stream_experts True: every such parameter;
        False: none)."""
        if self.stream_experts is False:
            return {}
        out = {}
        for name, value in module.state_dict(keep_vars=True).items():
            full = path + name
            if full not in self.converted or value.dim() != 3 or self._target(full) in self.members:
                continue
            converter, parts = self.converted[full]
            kinds = [type(operation).__name__ for operation in converter.operations]
            if not kinds or kinds[0] != "MergeModulelist" or set(kinds) - {"MergeModulelist", "Concatenate"} \
                    or any(len(sources) != value.shape[0] for sources in parts.values()):
                continue
            if self.stream_experts is None:
                if not str(self.device).startswith("cuda"):
                    continue
                free, _ = self.torch.cuda.mem_get_info(self.device)
                if value.numel() * 4 <= free // 2:
                    continue
            out[name] = _StreamedExperts(self, full, value)
        return out

    @staticmethod
    def _release(layer):
        """Put the meta parameters back for a released layer's streamed experts (their stored bytes go with them)."""
        for module in layer.modules():
            for attribute, value in list(vars(module).items()):
                if isinstance(value, _StreamedExperts):
                    del module.__dict__[attribute]
                    module._parameters[attribute] = value.placeholder

    def _place(self, module):
        """Move what `module` holds off the device (buffers it computed) onto it; meta tensors stay meta."""
        for child in module.modules():
            for group in (child._parameters, child._buffers):
                for key, tensor in group.items():
                    if tensor is not None and not tensor.is_meta and tensor.device != self.torch.device(self.device):
                        moved = tensor.to(self.device)
                        group[key] = (self.torch.nn.Parameter(moved, requires_grad=tensor.requires_grad)
                                      if isinstance(tensor, self.torch.nn.Parameter) else moved)

    def _load_outer(self):
        """Everything of the text model outside its decoder layers (embeddings, final norm or mixer, rotary
        frequencies)."""
        for name, child in self.skeleton.named_children():
            if child is not self.skeleton.layers:
                self._load(child, name + ".")
        for name, value in self.skeleton.state_dict(keep_vars=True).items():
            tensor = None if "." in name else self._tensor(name, value)
            if tensor is not None:
                setattr(self.skeleton, name, self.torch.nn.Parameter(tensor, requires_grad=False)
                        if isinstance(value, self.torch.nn.Parameter) else tensor)
        self._place(self.skeleton)

    def layer(self, index, upcoming=None):
        """Decoder layer `index` with its parameters and the persistent buffers the checkpoint stores (such as a
        layer scalar), weights dequantized to FP32, on the device. Its stored tensors arrive in one transfer, read
        ahead while the previous layer ran where the checkpoint stages reads (Checkpoint.stage); `upcoming`, the
        layer to load next, is read while this one runs."""
        placed = None
        if hasattr(self.checkpoint, "stage"):
            ahead, self.ahead = self.ahead, None
            staged = ahead[1].result() if ahead is not None else None   # the host buffer is shared: finish it
            if ahead is None or ahead[0] != index:
                staged = self._stage(index)
            if staged is not None:
                placed = self.checkpoint.place(*staged, self.device)
            if upcoming is not None:
                if self.reader is None:
                    from concurrent.futures import ThreadPoolExecutor
                    self.reader = ThreadPoolExecutor(1)
                self.ahead = (upcoming, self.reader.submit(self._stage, upcoming))
        try:
            module = self._load(self._restore(index), f"layers.{index}.")
        finally:
            if placed is not None:
                self.checkpoint.placed = {}
        if placed is not None:
            self._own(module, placed)
        return module

    def _restore(self, index):
        """Decoder layer `index` with the buffers it computed when built (self.built) where a release left meta ones."""
        layer = self.skeleton.layers[index]
        for name, buffer in self.built[index].items():
            owner, _, attribute = name.rpartition(".")
            module = layer.get_submodule(owner) if owner else layer
            if module._buffers.get(attribute) is not None and module._buffers[attribute].is_meta:
                module._buffers[attribute] = buffer
        return layer

    def _stage(self, index):
        """(buffer, layout) of decoder layer `index`'s stored tensors read into the host buffer, or None when it has
        none to stage or more than STAGE_LIMIT bytes. Its embedding tables' tensors are left out: they are gathered
        by rows, never read whole."""
        if self.layer_keys is None:
            self.layer_keys = {}
            pattern = re.compile(re.escape(self.prefix) + r"layers\.(\d+)\.")
            for key in self.checkpoint.where:
                found = pattern.match(key)
                if found:
                    self.layer_keys.setdefault(int(found.group(1)), []).append(key)
        path = f"layers.{index}."
        tables = [path + sub for sub, child in self.skeleton.layers[index].named_modules()
                  if isinstance(child, self.torch.nn.Embedding)]
        skip = {key for name in tables if name + ".weight" in self.converted
                for sources in self.converted[name + ".weight"][1].values() for _, key in sources}
        skip.update(key for name in tables for key in self.members.get(self._target(name + ".weight"), ()))
        stems = tuple(self._key(name + ".weight")[: -len("weight")] for name in tables)
        keys = [key for key in self.layer_keys.get(index, ()) if key not in skip and not key.startswith(stems)]
        if not keys or sum(self.checkpoint.info[key].nbytes for key in keys) > STAGE_LIMIT:
            return None
        self.host, layout = self.checkpoint.stage(keys, self.host, pin=str(self.device).startswith("cuda"))
        return self.host, layout

    def _own(self, module, placed):
        """Give every tensor of `module` that is a view of the placed copy (a stored FP32 tensor as it is) its own
        memory, so the copy is freed once the layer is loaded."""
        storage = placed.untyped_storage().data_ptr()
        for child in module.modules():
            for group in (child._parameters, child._buffers):
                for key, tensor in group.items():
                    if tensor is not None and not tensor.is_meta and tensor.untyped_storage().data_ptr() == storage:
                        group[key] = (self.torch.nn.Parameter(tensor.clone(), requires_grad=tensor.requires_grad)
                                      if isinstance(tensor, self.torch.nn.Parameter) else tensor.clone())

    def _config_dt_limit(self, layer):
        """Mamba-2 mixers take the model's own dt limit. transformers sets some mixers' limit to
        (time_step_min, inf) although config.json declares time_step_limit (usually [0, inf]) and the model's own
        kernels (mamba_chunk_scan_combined) apply that; the clamp makes long-memory heads forget, which shows
        as retrieval failing beyond ~64K tokens and as drift against engines that follow the config."""
        limit = getattr(self.config, "time_step_limit", None)
        if not isinstance(limit, (list, tuple)) or len(limit) != 2:
            return
        lo, hi = (float(v) for v in limit)
        for module in layer.modules():
            current = getattr(module, "time_step_limit", None)
            if isinstance(current, tuple) and len(current) == 2 and tuple(map(float, current)) != (lo, hi):
                module.time_step_limit = (lo, hi)

    def _implementation(self, implementation):
        """Point the model's attention (and the masks its forward builds) at `implementation` ("eager" or
        self.chunked)."""
        for config in self.configs:
            config._attn_implementation = implementation

    def _layer_count(self):
        return min(len(self.skeleton.layers), self.config.num_hidden_layers)

    def _forward(self, ids, stand_in, cache=None, use_cache=False):
        """Run the model's own forward on `ids` with decoder layer i replaced by stand_in(i) (its weights never
        read), after the prompt `cache` holds (use_cache: let the forward create one); returns the forward's output."""
        layers = self.skeleton.layers
        for i, layer in enumerate(layers):
            layer.forward = stand_in(i)
        try:
            return self.skeleton(input_ids=self.torch.tensor([ids], device=self.device), past_key_values=cache,
                                 use_cache=use_cache or cache is not None)
        finally:
            for layer in layers:
                layer.__dict__.pop("forward", None)

    def _capture(self, ids, cache=None, use_cache=False):
        """[(args, kwargs, slot)] per decoder layer: what the model's forward hands each layer for `ids`."""
        count = self._layer_count()
        calls, handed = [None] * count, []

        def stand_in(i):
            def forward(*args, **kwargs):
                slot, hidden = _hidden(args, kwargs)
                if i >= count or (i and (calls[i - 1] is None or hidden is not handed[-1])):
                    raise SchemeError("the model's forward does not hand each decoder layer the previous layer's "
                                      "output; the layer-streamed reference cannot drive it")
                calls[i] = (args, kwargs, slot)
                handed.append(hidden)
                if i == count - 1:
                    raise _Captured
                return hidden
            return forward
        try:
            self._forward(ids, stand_in, cache, use_cache)
        except _Captured:
            return calls
        raise SchemeError("the model's forward did not call every decoder layer")

    def _tail(self, ids, final):
        """The model's own forward after its decoder layers (final norm, mixer) on the last layer's output. It runs
        without a cache: the tail acts on each position's hidden state, and a cache must not be advanced twice."""
        last = self._layer_count() - 1
        out = self._forward(ids, lambda i: lambda *args, **kwargs: final if i == last else _hidden(args, kwargs)[1])
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

    def identity(self, model_sha):
        """What the prompt states and scores the reference stores depend on besides their ids."""
        import transformers
        device = self.torch.cuda.get_device_name(self.device) if str(self.device).startswith("cuda") else "cpu"
        return {"reference": REFERENCE_VERSION, "transformers": transformers.__version__,
                "torch": self.torch.__version__, "device": device, "model_sha256": model_sha,
                **({"quality_mode": self.quality_mode} if self.quality_mode != "preserve" else {}),
                **({"forms": [f"{name}@{FORM_VERSIONS.get(name, 1)}" for name in self.forms]} if self.forms else {}),
                # Only attention with sinks runs its long texts otherwise (chunked_sink_attention).
                **({"long_attention": self.chunked} if self.chunked != CHUNKED else {})}

    @staticmethod
    def _register():
        from transformers import AttentionInterface
        from transformers.masking_utils import AttentionMaskInterface, eager_mask, sdpa_mask
        AttentionInterface.register(CHUNKED, chunked_attention)
        AttentionMaskInterface.register(CHUNKED, sdpa_mask)
        AttentionInterface.register(CHUNKED_SINKS, chunked_sink_attention)
        AttentionMaskInterface.register(CHUNKED_SINKS, eager_mask)

    @staticmethod
    def _cache_of(calls):
        """The cache the forward hands its decoder layers (None without one)."""
        from transformers.cache_utils import Cache
        args, kwargs, _ = calls[0]
        return next((value for value in (*args, *kwargs.values()) if isinstance(value, Cache)), None)

    def _resume(self, ids, state):
        """_capture for `ids` after the prompt `state` holds."""
        if state.cache is None and not state.files:
            calls = self._capture(ids, use_cache=True)   # a prompt's first chunk: the forward makes the cache
            state.adopt(self._cache_of(calls))
            return calls
        if state.cache is None:   # a stored prompt: a fresh cache from the forward, made the prompt's
            state.restore(self._cache_of(self._capture(ids[:1], use_cache=True)), self.device)
        return self._capture(ids, state.cache)

    def _stream(self, passes, log, check, label=""):
        """Run `passes` through the decoder layers, each layer loaded once: first capture what the forward hands
        every layer for each pass, then layer by layer run every pass, its cache layer (a pass with a PromptState)
        on the device only while it runs. Returns each pass's last layer output."""
        torch = self.torch
        count = self._layer_count()
        records, states = [], []
        for p in passes:
            check()
            self._implementation(self.chunked if p.long else "eager")
            state = p.state
            if state is None:
                calls = self._capture(p.ids)
            else:
                try:
                    calls = self._resume(p.ids, state)
                except RESUME_ERRORS as exc:
                    raise CacheUnsupported(f"the model's forward cannot resume its cache ({exc})") from exc
            states.append(_hidden(*calls[0][:2])[1])
            records.append(calls)
        for index in range(count):
            started = time.time()
            layer = self.layer(index, index + 1 if index + 1 < count else None)
            loaded = time.time() - started
            for s, (p, calls) in enumerate(zip(passes, records)):
                check()
                self._implementation(self.chunked if p.long else "eager")
                args, kwargs, slot = calls[index]
                if slot == 0:
                    args = (states[s],) + tuple(args[1:])
                else:
                    kwargs = dict(kwargs, hidden_states=states[s])
                if p.state is not None:
                    p.state.enter(index, self.device)
                states[s] = layer(*args, **kwargs)
                if p.state is not None:
                    p.state.leave(index, p.keep)
                calls[index] = None
            self._release(layer)
            layer.to("meta")
            del layer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log(f"{label}layer {index + 1}/{count} {time.time() - started:.1f}s (load {loaded:.1f}s)")
        return states

    def _capture_job(self, job):
        """Capture what the forward hands each layer for prompt chunk `job` when the cache state its forward reads
        is the state the chunk-major order hands it: every layer it reads having run exactly the chunks before it.
        Returns whether it did; if not, job.waiting names the layers read too early, and nothing is changed."""
        prompt = job.prompt
        self._implementation(self.chunked if job.long else "eager")
        if job.chunk == 0:
            try:
                calls = self._capture(job.ids, use_cache=True)   # the forward makes the cache
                prompt.state.adopt(self._cache_of(calls))
            except RESUME_ERRORS as exc:
                raise CacheUnsupported(f"the model's forward cannot resume its cache ({exc})") from exc
        else:
            cache, before = prompt.state.cache, prompt.own[job.chunk - 1]
            before.apply(cache)
            calls, seen = None, set()
            try:
                with _watched(cache) as seen:
                    calls = self._capture(job.ids, cache)
            except RESUME_ERRORS as exc:
                if all(prompt.done[j] == job.chunk for j in seen):
                    raise CacheUnsupported(f"the model's forward cannot resume its cache ({exc})") from exc
            job.waiting = {j for j in seen if prompt.done[j] != job.chunk}
            if calls is None or job.waiting:
                before.apply(cache)
                return False
        prompt.own.append(_Own(prompt.state.cache))
        for follower in prompt.followers:
            if follower.shared == job.chunk + 1:
                self._fork(follower, prompt)
        # The job carries its hidden states on the host; the calls (every layer's handed the same tensor) do not.
        job.hidden = _hidden(*calls[0][:2])[1].to("cpu")
        job.calls = [((None,) + tuple(args[1:]), kwargs, slot) if slot == 0 else
                     (args, dict(kwargs, hidden_states=None), slot) for args, kwargs, slot in calls]
        return True

    @staticmethod
    def _runnable(job):
        """Whether prompt chunk `job` can run its next layer: its prompt has run that layer on the chunks before it."""
        return job.calls is not None and job.at < len(job.calls) and job.prompt.done[job.at] == job.chunk

    def _run_job(self, job, layer):
        """Run decoder layer job.at on prompt chunk `job`, with its cache as the chunk-major order holds it then: the
        cache's own attributes as the forward left them for this chunk, and every other layer whose state the layer
        reads as far as that order has run it (raises _OrderUnsafe otherwise)."""
        prompt, index = job.prompt, job.at
        self._implementation(self.chunked if job.long else "eager")
        args, kwargs, slot = job.calls[index]
        hidden = job.hidden.to(self.device)
        if slot == 0:
            args = (hidden,) + tuple(args[1:])
        else:
            kwargs = dict(kwargs, hidden_states=hidden)
        state, own = prompt.state, prompt.own[job.chunk]
        own.apply(state.cache)
        state.enter(index, self.device)
        with _watched(state.cache) as seen:
            out = layer(*args, **kwargs)
        if not own.held_by(state.cache):
            raise _OrderUnsafe("a decoder layer changes the cache's own state")
        early = sorted(j for j in seen if j != index and prompt.done[j] != job.chunk + (j < index))
        if early:
            raise _OrderUnsafe(f"decoder layer {index} reads the cache of layer {early[0]}")
        state.leave(index, True)
        prompt.done[index] += 1
        for follower in prompt.followers:
            if follower.shared == prompt.done[index]:
                self._fork_layer(follower, prompt, index)
        job.calls[index] = None
        job.at += 1
        job.hidden = out.to("cpu")

    def _fork(self, follower, leader):
        """Give `follower` its cache as the leader's after the chunks they share, the leader having just handed the
        last of them to the layers: a fresh cache from the forward with the leader's own attributes copied, every
        layer the leader has already left after those chunks in place (_fork_layer copies the others as it leaves
        them). The follower's next chunk can then be captured once the layers its forward reads are copied."""
        spec, tensors = _state(leader.state.cache, "cache.", skip=("layers",))
        state = follower.state
        self._implementation(self.chunked if follower.long else "eager")
        try:
            state.adopt(self._cache_of(self._capture(state.ids[:1], use_cache=True)))
        except RESUME_ERRORS as exc:
            raise CacheUnsupported(f"the model's forward cannot resume its cache ({exc})") from exc
        _apply(state.cache, spec, lambda path: tensors[path].to(self.device))
        follower.own = [None] * (follower.shared - 1) + [_Own(state.cache)]
        for index, done in enumerate(follower.done):
            if done == follower.shared:
                self._place_forked(follower, index)

    def _fork_layer(self, follower, leader, index):
        """Copy the leader's state of decoder layer `index`, just stored after the chunks it shares with `follower`,
        to the follower: its bytes, so the follower resumes exactly where its own run of those chunks would."""
        name = PromptState._layer(index)
        source, target = leader.state, follower.state
        target._put(name, source._bytes(name), source.files[name])
        follower.done[index] = follower.shared
        if target.cache is not None:
            self._place_forked(follower, index)

    @staticmethod
    def _place_forked(follower, index):
        """Meta placeholders of a copied layer's state in the follower's cache, until the layer runs."""
        state = follower.state
        state._grow(index + 1)
        state._placeholders(state.cache.layers[index], *state._header(PromptState._layer(index)))

    def _layer_major(self, prompts, jobs, log, check):
        """Run every prompt chunk (jobs, in order per prompt) through the decoder layers, loading each layer when its
        turn comes and running every chunk it can on it before the next: always the lowest layer that has one. A
        chunk is captured once the layers whose state its forward reads have run the chunks before it (usually the
        first layer of each kind, so only the layers up to those run chunk by chunk), and retried as those layers
        advance. Returns the number of layer loads."""
        torch = self.torch
        count = self._layer_count()

        def capture(prompt):
            """Capture the prompt's next chunk once the layers its last attempt read too early have caught up."""
            for job in jobs:
                if job.prompt is not prompt or job.calls is not None or job.chunk != len(prompt.own):
                    continue
                if any(prompt.done[j] != job.chunk for j in job.waiting):
                    return
                check()
                if not self._capture_job(job):
                    return
        for prompt in prompts:
            capture(prompt)
        loads = 0
        while True:
            waiting = [job.at for job in jobs if self._runnable(job)]
            if not waiting:
                if all(job.calls is not None and job.at == count for job in jobs):
                    return loads
                raise _OrderUnsafe("no chunk can run: the forward reads the cache of layers that cannot run yet")
            index, started = min(waiting), time.time()
            layer = self.layer(index, index + 1 if index + 1 < count else None)
            loaded, ran, loads = time.time() - started, 0, loads + 1
            while True:
                ready = [job for job in jobs if job.at == index and self._runnable(job)]
                if not ready:
                    break
                for job in ready:
                    if job.at == index and self._runnable(job):
                        check()
                        self._run_job(job, layer)
                        ran += 1
                        for prompt in (job.prompt, *job.prompt.followers):
                            capture(prompt)
            self._release(layer)
            layer.to("meta")
            del layer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log(f"layer-major: layer {index + 1}/{count} {time.time() - started:.1f}s (load {loaded:.1f}s), "
                f"{ran} passes")

    def score_texts(self, texts, top_k, valid_vocab, scratch, log=print, long=(), check=lambda: None, chunk=None):
        """Every position of each text (a list of ids) teacher-forced: per text (target log-probabilities of ids[1:]
        [n - 1] float64, the top_k ids [n - 1, top_k] int64 and their log-probabilities [n - 1, top_k] float64).
        Texts run in chunks of `chunk` tokens (PROMPT_CHUNK) that resume the cache, all of them layer-major (_layer_major); the
        chunks' cache states are scratch files in `scratch`, removed after. Where the layer-major order is unsafe
        the texts run chunk-major, and a model whose cache cannot be resumed runs each text whole. The texts whose
        indices are in `long` use chunked_attention."""
        import shutil
        torch = self.torch
        self._register()
        scratch, count = Path(scratch), self._layer_count()
        chunk = chunk or PROMPT_CHUNK
        chunks = [[ids[c:c + chunk] for c in range(0, len(ids), chunk)] for ids in texts]
        self.memory = _StateMemory(_StateMemory.budget() if self.state_memory is None else self.state_memory)
        outputs, unsafe = None, None
        with torch.no_grad():
            self._load_outer()
            try:
                prompts, jobs = [], []
                sharing = _shared_chunks(chunks, long) if self.share_prefixes else [(None, 0)] * len(texts)
                for t, (ids, (leader, shared)) in enumerate(zip(texts, sharing)):
                    state = PromptState(scratch / f"text-{t:03d}", {}, list(ids), self.memory)
                    prompts.append(_Prompt(state, t in long, count, None if leader is None else prompts[leader], shared,
                                           chunk))
                    jobs += [_Job(prompts[-1], piece, c) for c, piece in enumerate(prompts[-1].chunks) if c >= shared]
                    if shared:
                        log(f"reference: text {t + 1} shares its first {shared * chunk:,} tokens with text "
                            f"{leader + 1}; they run once")
                try:
                    self._layer_major(prompts, jobs, log, check)
                    own = {id(prompt): [job.hidden for job in jobs if job.prompt is prompt] for prompt in prompts}
                    outputs = [own[id(prompt)] if prompt.leader is None else
                               own[id(prompt.leader)][:prompt.shared] + own[id(prompt)] for prompt in prompts]
                except _OrderUnsafe as exc:
                    unsafe = str(exc)   # (outside the handler, so the traceback frees the passes' tensors)
                del prompts, jobs
                if unsafe is not None:
                    log(f"reference: {unsafe}; the texts run chunk by chunk")
                    shutil.rmtree(scratch, ignore_errors=True)
                    self.memory = _StateMemory(self.memory.limit)
                    outputs = self._text_rounds(texts, chunks, long, scratch, log, check)
            except CacheUnsupported as exc:
                log(f"reference: the cache cannot be resumed ({exc}); each text runs whole")
                outputs = self._stream([_Pass(ids, t in long) for t, ids in enumerate(texts)], log, check, "texts: ")
                outputs, chunks = [[out] for out in outputs], [[ids] for ids in texts]
            finally:
                shutil.rmtree(scratch, ignore_errors=True)
                self.memory = None
            head, cap = self._head(valid_vocab)
            return [self._text_rows(ids, pieces, hidden, head, cap, top_k, t in long)
                    for t, (ids, pieces, hidden) in enumerate(zip(texts, chunks, outputs))]

    def _text_rounds(self, texts, chunks, long, scratch, log, check):
        """score_texts' chunk-major order: round c runs every text's chunk c through all layers. Returns per text its
        chunks' last layer outputs."""
        states = [PromptState(scratch / f"text-{t:03d}", {}, list(ids), self.memory) for t, ids in enumerate(texts)]
        outputs = [[] for _ in texts]
        for c in range(max(map(len, chunks), default=0)):
            todo = [t for t in range(len(texts)) if c < len(chunks[t])]
            passes = [_Pass(chunks[t][c], t in long, states[t], keep=True) for t in todo]
            for t, out in zip(todo, self._stream(passes, log, check, f"texts {c + 1}/{max(map(len, chunks))}: ")):
                outputs[t].append(out.to("cpu"))
        return outputs

    def _head(self, valid_vocab):
        """(the output head [valid_vocab, hidden] in FP32 on the device, the final logit soft-cap or None)."""
        self.tables.clear()
        stored = self._key("lm_head.weight", "")
        tied = not _stored_tensor(self.checkpoint, stored) and getattr(self.config, "tie_word_embeddings", False)
        embedding = next(name for name, module in self.skeleton.named_modules()
                         if module is self.skeleton.get_input_embeddings())
        base = (self._key(embedding + ".weight") if tied else stored)[: -len(".weight")]
        head = self.checkpoint.dense(base, self.device)[:valid_vocab]
        # as_quantized: the head's inputs quantized as the checkpoint declares for it (FP8 per token, typically).
        self.head_inputs = self._activations(base) if self.quality_mode == "as_quantized" else None
        return head, getattr(self.config, "final_logit_softcapping", None)

    def _text_rows(self, ids, pieces, outputs, head, cap, top_k, long):
        """score_texts' rows of one text from its pieces' (consecutive chunks of ids) last layer outputs, through the
        model's tail and head, SCORE_ROWS positions at a time."""
        torch = self.torch
        targets, tops, values = [], [], []
        start = 0
        for piece, out in zip(pieces, outputs):
            self._implementation(self.chunked if long else "eager")
            hidden = self._tail(piece, out.to(self.device))[0]
            following = torch.tensor(ids[start + 1:start + len(piece) + 1], device=self.device)
            for row in range(0, len(following), SCORE_ROWS):
                rows = hidden[row:row + SCORE_ROWS][:len(following) - row]
                if self.head_inputs:
                    from ..limits.schemes import quantize_activations
                    rows = quantize_activations(rows, *self.head_inputs)
                logits = rows @ head.T
                if cap:
                    logits = torch.tanh(logits / cap) * cap
                logprobs = torch.log_softmax(logits.double(), dim=-1)
                block = following[row:row + SCORE_ROWS]
                targets.append(logprobs.gather(1, block[:, None])[:, 0].cpu())
                top = logprobs.topk(top_k, dim=-1)
                tops.append(top.indices.cpu())
                values.append(top.values.cpu())
            start += len(piece)
        return torch.cat(targets), torch.cat(tops), torch.cat(values)


def store_texts(reference, texts, top_k, valid_vocab, data, model_sha, log=print, check=lambda: None, spread=None,
                layout=None, store=STORE, budget=None):
    """Score every fixed text once per identity and store the scores under prepared data folder `data`
    (crit_eval.quality.fixed: each position's target log-probability, top_k ids and their log-probabilities, with the identity:
    the reference's, top_k, valid_vocab and every text's ids). texts: {text id: (ids, long)}. Returns how they were
    obtained: "stored" when the data holds them under the same identity, else "scored" with its seconds and bytes.
    spread (protocol quality.spread) with layout ({text id: (set, prompt tokens)}): the texts of at least
    spread["from_tokens"] tokens are scored once more under spread["perturbation"] (store_spread), the model's own
    arithmetic perturbed as an engine's would be, and how far that run lands from the stored scores is stored per set
    and path (fixed.write_spread), the spread a run judges those sets against; "spread" says whether it was stored,
    scored or probed (store_spread). store: the scores' folder (crit_eval.quality.fixed STORE). budget: the drift
    budget the spread's probe is measured against (protocol quality.thresholds)."""
    import shutil
    from . import fixed
    identity = dict(reference.identity(model_sha), **fixed.identity(texts, top_k, valid_vocab))
    positions = sum(len(ids) - 1 for ids, _ in texts.values())
    if fixed.load(data, identity, store) is not None:
        out = {"method": "stored", "positions": positions}
    else:
        shutil.rmtree(Path(data) / store, ignore_errors=True)
        names = list(texts)
        started = time.time()
        rows = reference.score_texts([texts[name][0] for name in names], top_k, valid_vocab,
                                     Path(data) / store / "scratch", log=log,
                                     long={i for i, name in enumerate(names) if texts[name][1]}, check=check)
        size = fixed.write(data, identity, {name: (target.tolist(), top.tolist(), values.tolist())
                                            for name, (target, top, values) in zip(names, rows)}, store)
        out = {"method": "scored", "seconds": round(time.time() - started, 1), "positions": positions, "bytes": size}
    if spread and layout:
        out["spread"] = store_spread(reference, texts, top_k, valid_vocab, data, identity, spread, layout, log, check,
                                     store, budget)
    return out


SPREAD_PERTURBATIONS = ("tf32",)


def store_spread(reference, texts, top_k, valid_vocab, data, identity, rule, layout, log=print, check=lambda: None,
                 store=STORE, budget=None):
    """The reference's own spread on the texts of at least rule["from_tokens"] tokens (store_texts): scored once more
    under rule["perturbation"] and compared with the stored scores by crit_eval.reports.metrics.text_quality per set and path
    (SET/prompt: the prompt positions, SET/served: the continuation's), stored with the identity plus the perturbation
    (fixed.write_spread). "tf32": the FP32 matmuls run with TF32 inputs (10-bit mantissas, about 1e-3 relative), the
    size of perturbation a BF16 or FP8 engine's arithmetic introduces, so a model that amplifies rounding (routing
    near-ties, hard top-k, a recurrent state) shows how far such an engine must be expected to land; FP32 rounding alone
    (another chunking, 1e-6) proved a hundred times too small on a sparse MoE model. On a device without TF32 (the
    CPU) the pass is the same arithmetic and the spread is zero. With rule["probe"] and a budget, the texts shorter
    than probe["below_tokens"] run first, and the longer ones only when the probe's spread is not tiny (_spread_tiny):
    a spread stored so holds the probed sets and names the texts it skipped (spread_skipped), whose sets are judged
    against the budget alone. Returns "stored", "scored", "probed" (the longer texts skipped) or "none" (no text that
    long)."""
    from . import fixed
    perturbation = rule.get("perturbation", "tf32")
    if perturbation not in SPREAD_PERTURBATIONS:
        raise SystemExit(f"quality.spread.perturbation must be one of {', '.join(SPREAD_PERTURBATIONS)}")
    long_texts = {name: ids_long for name, ids_long in texts.items() if len(ids_long[0]) >= int(rule["from_tokens"])}
    spread_identity = {**identity, "spread_perturbation": perturbation, "spread_texts": sorted(long_texts)}
    if not long_texts:
        return "none"
    if fixed.load_spread(data, spread_identity, store) is not None:
        return "stored"
    probe = rule.get("probe") if budget else None
    first = [name for name in long_texts if probe is None or len(long_texts[name][0]) < int(probe["below_tokens"])]
    later = [name for name in long_texts if name not in first]
    if not first or not later:
        first, later = list(long_texts), []
    log(f"reference spread: {len(first)} text(s) scored again with {perturbation} matmuls"
        + (f" (the {len(later)} of at least {int(probe['below_tokens']):,} tokens only if their spread is not tiny)"
           if later else ""))
    rows = _spread_rows(reference, {name: long_texts[name] for name in first}, top_k, valid_vocab, data, store, log,
                        check)
    sets = _spread_sets(rows, fixed.load(data, identity, store), layout)
    skipped = []
    if later:
        tiny = _spread_tiny(sets, budget, float(rule["factor"]), float(probe["margin"]))
        if tiny:
            skipped = sorted(later)
            log(f"reference spread: below 1/{probe['margin']:g} of the budget over factor on every probed set; "
                f"{', '.join(skipped)} not scored again (their sets are judged against the budget alone)")
        else:
            log(f"reference spread: not tiny on the probed sets; {len(later)} more text(s) scored again")
            rows.update(_spread_rows(reference, {name: long_texts[name] for name in later}, top_k, valid_vocab, data,
                                     store, log, check))
            sets = _spread_sets(rows, fixed.load(data, identity, store), layout)
    for name, row in sets.items():
        log(f"reference spread {name}: KLD mean {row['kld_mean']:.2e}, p99 {row['kld_p99']:.2e}, "
            f"PPL x{row['ppl_ratio']:.4f} over {row['targets']:,} positions")
    fixed.write_spread(data, {**spread_identity, **({"spread_skipped": skipped,
                                                      "spread_probe": {key: probe[key] for key in ("below_tokens",
                                                                                                   "margin")}}
                                                     if skipped else {})}, sets, store)
    return "probed" if skipped else "scored"


def _spread_rows(reference, texts, top_k, valid_vocab, data, store, log, check):
    """{text id: rows} of `texts` ({id: (ids, long)}) scored with TF32 matmuls (store_spread)."""
    torch = reference.torch
    names = list(texts)
    previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    try:
        rows = reference.score_texts([texts[name][0] for name in names], top_k, valid_vocab,
                                     Path(data) / store / "scratch-spread", log=log,
                                     long={i for i, name in enumerate(names) if texts[name][1]}, check=check)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous
    return dict(zip(names, rows))


def _spread_sets(rows, stored, layout):
    """{"SET/path": text_quality row} of a run's rows ({text id: (target, top ids, top log-probabilities)} tensors)
    against the stored scores, the texts' positions split as crit_eval.run.quality_checks.run_quality splits them."""
    return _compare_sets({name: [{"logprob": lp, "top": [[int(i), float(v)] for i, v in zip(ids, tops)]}
                                 for lp, ids, tops in zip(target.tolist(), top.tolist(), values.tolist())]
                          for name, (target, top, values) in rows.items()}, stored, layout)


def _compare_sets(found_rows, stored, layout):
    """_spread_sets of rows in an engine's form ({text id: [{"logprob", "top": [[id, logprob], ...]}, ...]})."""
    from ..reports import metrics
    parts = {}
    for name, again in found_rows.items():
        set_name, prompt = layout[name]
        last = prompt - 1
        for part, pick in (("prompt", slice(0, last)), ("served", slice(last, None))):
            reference_rows, found = parts.setdefault(f"{set_name}/{part}", ([], []))
            reference_rows += stored[name][pick]
            found += again[pick]
    return {name: metrics.text_quality(reference_rows, found)
            for name, (reference_rows, found) in parts.items() if found}


def _spread_tiny(sets, budget, factor, margin):
    """Whether every set and path's spread stays below 1/margin of the level where it would raise a limit
    (crit_eval.run.quality_checks.set_limits: the budget over factor), the perplexity ratio judged on the served
    positions only (protocol quality.prompt_perplexity)."""
    for name, row in sets.items():
        if row["kld_mean"] * factor * margin > budget["kld_mean_max"] or \
                row["kld_p99"] * factor * margin > budget["kld_p99_max"]:
            return False
        if name.endswith("/served") and abs(row["ppl_ratio"] - 1.0) * factor * margin > budget["ppl_ratio_max"] - 1.0:
            return False
    return True


def store_allowance(model, data, texts, top_k, valid_vocab, identity, layout, declared, log=print,
                    check=lambda: None, device="cuda"):
    """protocol quality.allowance: how far the checkpoint run with its activations quantized as it declares
    (`declared`: StreamingReference.declared_activations) lands from the stored scores, per set and path
    (_compare_sets), stored beside them (fixed.write_allowance) for runs in quality mode as_quantized. A checkpoint
    that declares none stores an empty allowance (as_quantized is then preserve). Scores of the as_quantized mode
    stored by crit-eval before protocol crit-eval-v0.3.0 (fixed.LEGACY_AS_QUANTIZED) of the same model, texts and
    reference give it without the second run. Returns "stored", "none", "derived", "measured" or "unsupported" (a
    declared form crit does not emulate, recorded so runs in that mode say why)."""
    from . import fixed
    allowance_identity = {**identity, "allowance": "as_quantized"}
    if fixed.load_allowance(data, allowance_identity) is not None:
        return "stored"
    if isinstance(declared, Exception):
        fixed.write_allowance(data, allowance_identity, {"unsupported": str(declared), "sets": {}})
        log(f"reference allowance: not measured ({declared})")
        return "unsupported"
    if not declared:
        fixed.write_allowance(data, allowance_identity, {"activations": [], "sets": {}})
        return "none"
    stored = fixed.load(data, identity)
    legacy = fixed.load(data, {**{key: identity[key] for key in ("reference", "model_sha256", "top_k", "valid_vocab",
                                                                 "texts")}, "quality_mode": "as_quantized"},
                        fixed.LEGACY_AS_QUANTIZED)
    started = time.time()
    if legacy is not None:
        found = {name: [{"logprob": row["logprob"], "top": list(zip(row["top_ids"], row["top_logprobs"]))}
                        for row in rows] for name, rows in legacy.items()}
        sets, how = _compare_sets(found, stored, layout), "derived"
        log(f"reference allowance: derived from the {fixed.LEGACY_AS_QUANTIZED} scores stored beside the data")
    else:
        names = list(texts)
        log(f"reference allowance: the texts scored again with the activations quantized as the checkpoint declares "
            f"({', '.join(declared)})")
        try:
            quantized = StreamingReference(model, device, quality_mode="as_quantized")
            rows = quantized.score_texts([texts[name][0] for name in names], top_k, valid_vocab,
                                         Path(data) / STORE / "scratch-allowance", log=log,
                                         long={i for i, name in enumerate(names) if texts[name][1]}, check=check)
        except SchemeError as error:
            fixed.write_allowance(data, allowance_identity, {"unsupported": str(error), "sets": {}})
            log(f"reference allowance: not measured ({error})")
            return "unsupported"
        sets, how = _spread_sets(dict(zip(names, rows)), stored, layout), "measured"
    for name, row in sets.items():
        log(f"reference allowance {name}: KLD mean {row['kld_mean']:.2e}, p99 {row['kld_p99']:.2e}, "
            f"PPL x{row['ppl_ratio']:.4f} over {row['targets']:,} positions")
    fixed.write_allowance(data, allowance_identity, {"activations": declared, "sets": sets, "source": how,
                                                     "seconds": round(time.time() - started, 1)})
    return how


def score_data(model, data, log=print, check=lambda: None, quality_mode=None):
    """Score every quality text of the prepared data at every position on the checkpoint it was prepared from and
    store the scores under the data (crit_eval.quality.fixed), with the reference's own spread and, for a checkpoint
    that declares quantized activations, the as_quantized allowance (store_allowance), unless they are stored already
    under the same identity: one run serves both quality modes (quality_mode is accepted from callers of earlier
    versions and changes nothing). check() runs between layers and texts: it raises to stop. Returns how they were
    obtained."""
    from . import fixed
    protocol = load_json(EVAL_DIR / "protocol.json")
    prepared = load_json(Path(data) / PREPARED)
    if model_identity(model, prepared_template(prepared))["sha256"] != prepared["model"]["sha256"]:
        raise SystemExit(f"{model} is not the checkpoint {data} was prepared from: the reference is that checkpoint")
    layout = {t["id"]: (t["set"], len(t["prompt_ids"])) for t in prepared["quality"]}
    texts, top_k = fixed.texts(prepared), protocol["quality"]["top_k"]
    reference = StreamingReference(model)
    out = store_texts(reference, texts, top_k, prepared["valid_vocab"], data, prepared["model"]["sha256"], log=log,
                      check=check, spread=protocol["quality"].get("spread"), layout=layout,
                      budget=protocol["quality"]["thresholds"])
    identity = dict(reference.identity(prepared["model"]["sha256"]), **fixed.identity(texts, top_k, prepared["valid_vocab"]))
    try:
        declared = reference.declared_activations()
    except SchemeError as error:
        declared = error
    del reference   # its device memory and staging buffers, before the quantized pass builds its own
    import gc
    gc.collect()
    out["allowance"] = store_allowance(model, data, texts, top_k, prepared["valid_vocab"], identity, layout, declared,
                                       log, check)
    return out
