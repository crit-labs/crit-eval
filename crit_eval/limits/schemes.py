"""How a checkpoint keyed by Hugging Face names stores its weight matrices, decided in one place for the ceiling and
the reference: the recognition is crit_core.model.schemes (header-only, shared with engines that load matrices as
stored), the decoders are here.

A matrix `base` (a module path such as model.layers.0.mlp.down_proj) is stored as one scheme, recognised from the
tensors under `base.` and the model's quantization config (quantization_config, or MLX's quantization):

    plain         base.weight in BF16, F16 or F32
    fp8           base.weight F8_E4M3 with base.weight_scale per tensor, per row or per block (compressed-tensors
                  block_structure), or base.weight_scale_inv per block (weight_block_size)
    nvfp4         E2M1 pairs (low nibble first) with E4M3 group-16 scales and an FP32 global factor:
                  compressed-tensors (U8 base.weight_packed, base.weight_global_scale as a divisor) or ModelOpt
                  (U8 base.weight, base.weight_scale_2 as a multiplier)
    int           compressed-tensors pack-quantized (I32 base.weight_packed: codes of num_bits offset by
                  2**(num_bits - 1), packed densely along the input; base.weight_scale per group or row, optional
                  base.weight_zero_point packed along the output and base.weight_g_idx), and int-quantized (I8
                  base.weight with base.weight_scale)
    awq           AutoAWQ GEMM (base.qweight [in, out/8] in AWQ's interleaved order, base.qzeros, base.scales)
    gptq          GPTQ (base.qweight [in * bits / 32, out] packed densely along the input, base.qzeros along the
                  output, base.scales, base.g_idx); checkpoint_format "gptq" stores zero points minus one, as
                  packed integers, "gptq_v2" stores them as they are
    mlx           MLX (U32 base.weight, codes packed densely along the input, base.scales per group), its
                  bits, group size and mode from config.json's quantization with per-module overrides: affine
                  (w = scale * q + base.biases), mxfp4 and nvfp4 (E2M1 codes with E8M0 or E4M3 scales) and mxfp8
                  (E4M3 codes with E8M0 scales)
    bnb4          bitsandbytes NF4/FP4 (U8 base.weight, two codes a byte, high nibble first; base.weight.absmax
                  per block, itself 8-bit with base.weight.nested_* when double-quantized; base.weight.quant_map
                  the code table; base.weight.quant_state.bitsandbytes__<type> a JSON blob with the shape)
    bnb8          bitsandbytes LLM.int8 (I8 base.weight, base.SCB per row: w = q * SCB / 127)
    nbits         ONNX Runtime's MatMulNBits (base.nbits_weight and its scales, zero points and g_idx, as
                  crit_core.model.onnx names a MatMulNBits node's inputs, with the node's K, N, bits and block size)
    exl3          EXL3 (base.trellis [in / 16, out / 16, 16 K] I16 trellis codes for K bits a weight, 1 to 8 or 1.5 /
                  2.5 / 3.5, the sign vectors base.suh / base.svh or their bitfields base.su / base.sv, a base.mcg or
                  base.mul1 marker naming the codebook): w = diag(su) H Q H diag(sv), Q the tiles as the format's
                  decoder gives them (FP16, exact) and H the 128-point Hadamard, the rotations in FP32

stored(tensors, base, quantization) gives the scheme's name, its GEMM rate key for the prefill floor and the
matrix's element count from headers alone (no torch). dense(checkpoint, base, device) decodes it to FP32 [out, in]
with the same arithmetic the format's own library performs in FP32 (torch). Weight-only integer schemes count at
the bf16 rate (engines widen their codes to 16 bits for the GEMM); 8-bit integer weights with per-row scales at
the fp8 rate (int8 tensor cores run at the FP8 rate). A new scheme is one entry in crit_core's RECOGNISERS and one
in DECODERS.

activations(checkpoint, base) says how the checkpoint's own inference quantizes the activations a quantized matrix
multiplies (its declared input activations: compressed-tensors input_activations with base.input_global_scale or
base.input_scale, ModelOpt base.input_scale), and quantize_activations applies it in FP32 in the reference's
allowance pass (crit_eval.quality.reference, reported only; both quality modes judge against exact activations).
"""
import json
import math

from crit_core.model.schemes import (PRIMARY, RECOGNISERS, UNREAD, SchemeError, quantization_of,  # noqa: F401
                                     refuse_unread, stored)
from crit_core.model.schemes import (CSF_EXCEPTIONS, CSF_FIXED, csf_scale_shape, ct_group as _ct_group,
                                     ct_weight_args as _ct_args, fp8_block as _fp8_block, mlx_settings as _mlx)
from crit_core.model.schemes import gptq_bits as _gptq_bits, quant_method as _method

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
FP8_BLOCK = 128


FP8_MAX = 448.0
E2M1_MAX = 6.0


def activations(checkpoint, base):
    """How the checkpoint's own inference quantizes the activations quantized matrix `base` multiplies: None when
    it declares none (weight-only schemes, plain matrices), else (kind, scale):
        ("nvfp4", g)      E2M1 values in groups of 16 along the input with E4M3 group scales, global scale g a
                          divisor (compressed-tensors base.input_global_scale; ModelOpt 1 / base.input_scale)
        ("fp8_token", None) E4M3 values with one scale per token, amax / 448 (dynamic, strategy token)
        ("fp8_tensor", s) E4M3 values with the stored static scale s (base.input_scale, a multiplier)
    SchemeError for activations declared in a form crit does not emulate."""
    kind = scheme(checkpoint, base)
    if kind not in ("fp8", "nvfp4"):
        return None
    quantization = checkpoint.quantization or {}
    if _method(quantization) == "compressed-tensors":
        group = _ct_group(quantization, base, linear=True)
        args = (group or {}).get("input_activations")
        if not args:
            return None
        bits, numeric = int(args.get("num_bits", 8)), str(args.get("type", "float"))
        if numeric == "float" and bits == 4 and int(args.get("group_size") or 0) == 16:
            if not checkpoint.has(base + ".input_global_scale"):
                raise SchemeError(f"{base}: NVFP4 input activations without {base}.input_global_scale")
            return "nvfp4", float(checkpoint.raw(base + ".input_global_scale").float().reshape(-1)[0])
        if numeric == "float" and bits == 8 and args.get("dynamic") is True and args.get("strategy") == "token":
            return "fp8_token", None
        if numeric == "float" and bits == 8 and not args.get("dynamic") and checkpoint.has(base + ".input_scale"):
            return "fp8_tensor", float(checkpoint.raw(base + ".input_scale").float().reshape(-1)[0])
        raise SchemeError(f"{base}: input activations {args} are not a form crit emulates")
    if checkpoint.has(base + ".input_scale"):   # ModelOpt: a static input scale per matrix
        scale = float(checkpoint.raw(base + ".input_scale").float().reshape(-1)[0])
        return ("nvfp4", 1.0 / scale) if kind == "nvfp4" else ("fp8_tensor", scale)
    return None


def _round_e2m1(magnitude):
    """Magnitudes (at most 6) to the nearest E2M1 value, ties to the even code, as the hardware conversion rounds."""
    import torch
    grid = torch.tensor(E2M1, dtype=magnitude.dtype, device=magnitude.device)
    middles = (grid[1:] + grid[:-1]) / 2
    index = torch.bucketize(magnitude, middles)   # a tie goes to the lower value, code index
    upward = (index < len(middles)) & (magnitude == middles[index.clamp(max=len(middles) - 1)]) & (index % 2 == 1)
    return grid[index + upward.long()]


def quantize_activations(x, kind, scale):
    """FP32 activations `x` [..., in] quantized and dequantized as `activations()` describes (kind, scale)."""
    import torch
    if kind == "nvfp4":
        shape = x.shape
        if shape[-1] % 16:
            raise SchemeError(f"NVFP4 activations need a multiple of 16 inputs, not {shape[-1]}")
        blocks = x.reshape(*shape[:-1], shape[-1] // 16, 16)
        amax = blocks.abs().amax(-1, keepdim=True)
        stored = (amax / E2M1_MAX * scale).clamp(max=FP8_MAX).to(torch.float8_e4m3fn).float()
        step = stored / scale
        safe = torch.where(step > 0, step, torch.ones_like(step))
        codes = _round_e2m1((blocks.abs() / safe).clamp(max=E2M1_MAX)) * blocks.sign()
        return (codes * step).reshape(shape)
    if kind == "fp8_token":
        step = (x.abs().amax(-1, keepdim=True) / FP8_MAX).clamp(min=torch.finfo(torch.float32).tiny)
    elif kind == "fp8_tensor":
        step = torch.tensor(scale, dtype=x.dtype, device=x.device)
    else:
        raise SchemeError(f"unknown activation quantization {kind!r}")
    return (x / step).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * step


# Each scheme: stored(tensors, base, quantization) -> (rate, elements) or None, and decode(checkpoint, base,
# device) -> FP32 [out, in]. Checked in order; the first whose tensors are present names the matrix.


def _plain_decode(checkpoint, base, device):
    return checkpoint.raw(base + ".weight").to(device).float()


def _fp8_decode(checkpoint, base, device):
    weight = checkpoint.raw(base + ".weight").to(device).float()
    import torch
    inverse = checkpoint.has(base + ".weight_scale_inv")   # despite the name, a multiplier too
    scale = checkpoint.raw(base + (".weight_scale_inv" if inverse else ".weight_scale")).to(device)
    # E8M0 scales (ue8m0) come as float8_e8m0fnu or as their bytes: 2 ** (s - 127).
    scale = (torch.ldexp(torch.ones_like(scale, dtype=torch.float32), scale.to(torch.int32) - 127)
             if scale.dtype == torch.uint8 else scale.float())
    rows, cols = weight.shape[-2:]
    if scale.numel() == 1:
        return weight * scale.reshape(-1)[0]
    if scale.shape[-2:] == (rows, 1):
        return weight * scale
    # Block scales: one per block (crit_core's fp8_block: weight_block_size, compressed-tensors' block_structure,
    # else the block the scales' grid gives, such as modelopt MXFP8's one scale per 32 inputs of a row), the edge
    # blocks cut short.
    block = _fp8_block(checkpoint.quantization, base, (rows, cols), scale.shape)
    if block is None:
        raise SchemeError(f"{base}: FP8 scales of shape {list(scale.shape)} fit no block of [{rows}, {cols}]")
    expanded = scale.repeat_interleave(int(block[0]), -2).repeat_interleave(int(block[1]), -1)
    return weight * expanded[..., :rows, :cols]


def csf_scales(checkpoint, base, device):
    """The E4M3 [rows, groups] scales of a ModelOpt NVFP4 matrix stored CSF-compressed (crit_core's csf_scale_shape):
    each row's base byte plus its 4-bit code, then the exceptions' bytes at their positions."""
    import torch
    fixed = checkpoint.raw(base + CSF_FIXED).to(device)
    rows, groups = csf_scale_shape(tuple(fixed.shape))
    bases = fixed[:, :16].reshape(rows, 1).to(torch.int16)
    codes = fixed[:, 16:].reshape(rows, groups // 2)
    nibbles = torch.stack((codes & 0x0F, codes >> 4), dim=-1).reshape(rows, groups).to(torch.int16)
    out = (bases + nibbles).reshape(-1)
    if checkpoint.has(base + CSF_EXCEPTIONS):
        words = checkpoint.raw(base + CSF_EXCEPTIONS).to(device).reshape(-1).to(torch.int64) & 0xFFFFFFFF
        positions = words & 0xFFFFFF
        if positions.numel() and int(positions.max()) >= rows * groups:
            raise SchemeError(f"{base}{CSF_EXCEPTIONS}: a position past the {rows} x {groups} scales")
        out[positions] = (words >> 24).to(torch.int16)
    return out.to(torch.uint8).view(torch.float8_e4m3fn).reshape(rows, groups)


def _nvfp4_decode(checkpoint, base, device):
    import torch
    if checkpoint.has(base + ".weight_packed"):
        packed = checkpoint.raw(base + ".weight_packed").to(device)
        scale = checkpoint.raw(base + ".weight_scale").to(device).float()
        scale = scale / checkpoint.raw(base + ".weight_global_scale").to(device).float()
    else:
        packed = checkpoint.raw(base + ".weight").to(device)
        scale = (checkpoint.raw(base + ".weight_scale").to(device) if checkpoint.has(base + ".weight_scale")
                 else csf_scales(checkpoint, base, device)).float()
        scale = scale * checkpoint.raw(base + ".weight_scale_2").to(device).float()
    table = torch.tensor(E2M1, dtype=torch.float32, device=device)
    low, high = packed & 0x0F, packed >> 4
    nibbles = torch.stack((low, high), dim=-1).reshape(packed.shape[0], -1).long()
    values = table[nibbles & 7] * torch.where(nibbles & 8 > 0, -1.0, 1.0)
    values = values.reshape(values.shape[0], scale.shape[1], -1) * scale[..., None]
    return values.reshape(packed.shape[0], -1)


def unpack_bits(words, bits, count):
    """Codes of `bits` bits packed densely, little end first, along the last dimension of int32 `words`: code i at
    bit i * bits of the row's bit stream (across word boundaries for 3, 5 and 6 bits). Returns int64 [..., count]."""
    import torch
    stream = words.to(torch.int64) & 0xFFFFFFFF
    positions = torch.arange(count, device=words.device) * bits
    word, shift = positions // 32, positions % 32
    low = stream[..., word] >> shift
    spill = (shift + bits > 32)
    high = torch.zeros_like(low)
    if bool(spill.any()):
        nxt = torch.clamp(word + 1, max=stream.shape[-1] - 1)
        high = torch.where(spill, stream[..., nxt] << (32 - shift), 0)
    return (low | high) & ((1 << bits) - 1)


def _ct_int_decode(checkpoint, base, device):
    import torch
    args = _ct_args(checkpoint.quantization, base)
    scale = checkpoint.raw(base + ".weight_scale").to(device).float()
    if checkpoint.has(base + ".weight"):          # int-quantized: int8 codes as they are
        codes = checkpoint.raw(base + ".weight").to(device).to(torch.int64)
    else:
        bits = int(args.get("num_bits", 4))
        cols = int(checkpoint.raw(base + ".weight_shape").tolist()[-1])
        packed = checkpoint.raw(base + ".weight_packed").to(device)
        codes = unpack_bits(packed, bits, cols) - (1 << (bits - 1))
    rows, cols = codes.shape
    groups = scale.shape[1] if scale.ndim == 2 else 1
    if checkpoint.has(base + ".weight_g_idx"):
        group = checkpoint.raw(base + ".weight_g_idx").to(device).long()
    else:
        group = torch.arange(cols, device=device) // math.ceil(cols / groups)
    if checkpoint.has(base + ".weight_zero_point"):
        zero = checkpoint.raw(base + ".weight_zero_point").to(device)
        if zero.dtype == torch.int32 and checkpoint.has(base + ".weight_packed"):   # packed along the output
            bits = int(args.get("num_bits", 4))
            zero = unpack_bits(zero.transpose(0, 1).contiguous(), bits, rows).transpose(0, 1) - (1 << (bits - 1))
        codes = codes - zero.to(torch.int64).reshape(rows, -1)[:, group]
    scale = scale.reshape(rows, -1)
    return codes.float() * scale[:, group]


def _awq_decode(checkpoint, base, device):
    import torch
    quantization = checkpoint.quantization
    version = str(quantization.get("version") or quantization.get("format") or "gemm").lower()
    if version != "gemm" or int(quantization.get("bits", quantization.get("w_bit", 4))) != 4:
        raise SchemeError(f"{base}: AWQ {version} with {quantization.get('bits', 4)} bits is not decoded "
                          "(the reference reads AWQ GEMM, 4 bits)")
    qweight = checkpoint.raw(base + ".qweight").to(device)
    qzeros = checkpoint.raw(base + ".qzeros").to(device)
    scales = checkpoint.raw(base + ".scales").to(device).float()

    def unpack(words):
        shifts = torch.arange(0, 32, 4, device=device)
        codes = ((words.to(torch.int64)[..., None] >> shifts) & 0xF).reshape(words.shape[0], -1)
        order = torch.arange(codes.shape[1], device=device).reshape(-1, 8)[:, AWQ_REVERSE_ORDER].reshape(-1)
        return codes[:, order]
    weight, zeros = unpack(qweight), unpack(qzeros)
    group = weight.shape[0] // scales.shape[0]
    zeros = zeros.repeat_interleave(group, dim=0)
    return ((weight - zeros).float() * scales.repeat_interleave(group, dim=0)).T.contiguous()


def _gptq_decode(checkpoint, base, device):
    import torch
    quantization = checkpoint.quantization
    bits = _gptq_bits(quantization)
    form = str(quantization.get("checkpoint_format") or quantization.get("format") or "gptq").lower()
    if form not in ("gptq", "gptq_v2") or bits not in (2, 3, 4, 8):
        raise SchemeError(f"{base}: GPTQ format {form!r} with {bits} bits is not decoded (the reference reads "
                          "gptq and gptq_v2 checkpoints of 2, 3, 4 or 8 bits)")
    qweight = checkpoint.raw(base + ".qweight").to(device)
    qzeros = checkpoint.raw(base + ".qzeros").to(device).to(torch.int64) & 0xFFFFFFFF
    scales = checkpoint.raw(base + ".scales").to(device).float()
    out = qweight.shape[1]
    zeros = unpack_bits(qzeros, bits, out)
    if form == "gptq":
        # v1 stores every zero point minus one: 2, 4 and 8 bits subtracted as packed integers, which adding back
        # the same way undoes exactly (borrows between fields included); 3 bits, whose fields straddle words, field
        # by field (as GPTQModel reads and writes them).
        if bits == 3:
            zeros = (zeros + 1) & 7
        else:
            one = sum(1 << (k * bits) for k in range(32 // bits))
            zeros = unpack_bits((qzeros + one) & 0xFFFFFFFF, bits, out)
    cols = qweight.shape[0] * 32 // bits
    if checkpoint.has(base + ".g_idx"):
        group = checkpoint.raw(base + ".g_idx").to(device).long()
        cols = group.shape[0]
    else:
        size = int(quantization.get("group_size", -1))   # -1: one group per row
        group = torch.arange(cols, device=device) // (size if size > 0 else cols)
    codes = unpack_bits(qweight.transpose(0, 1).contiguous(), bits, cols).transpose(0, 1)
    return (scales[group] * (codes - zeros[group]).float()).T.contiguous()


def _mlx_decode(checkpoint, base, device):
    import torch
    bits, group, mode = _mlx(checkpoint.quantization, base)
    weight = checkpoint.raw(base + ".weight").to(device)
    scales = checkpoint.raw(base + ".scales").to(device)
    cols = weight.shape[-1] * 32 // bits
    codes = unpack_bits(weight, bits, cols)
    shape = codes.shape
    codes = codes.reshape(*shape[:-1], -1, group)
    if mode == "affine":
        biases = checkpoint.raw(base + ".biases").to(device).float()
        values = codes.float() * scales.float()[..., None] + biases[..., None]
    elif mode in ("mxfp4", "nvfp4", "mxfp8"):
        if mode == "mxfp8":
            values = codes.to(torch.uint8).view(torch.float8_e4m3fn).float()
        else:
            table = torch.tensor(E2M1, dtype=torch.float32, device=device)
            values = table[codes & 7] * torch.where(codes & 8 > 0, -1.0, 1.0)
        if mode == "nvfp4":   # E4M3 group scales
            scale = scales.view(torch.float8_e4m3fn).float()
        else:                 # E8M0 group scales: 2 ** (s - 127)
            scale = torch.ldexp(torch.ones_like(scales, dtype=torch.float32), scales.to(torch.int32) - 127)
        values = values * scale[..., None]
    else:
        raise SchemeError(f"{base}: MLX quantization mode {mode!r} is not decoded")
    return values.reshape(shape)


def _bnb4_state(checkpoint, base):
    for name in checkpoint.where:
        if name.startswith(base + ".weight.quant_state.bitsandbytes__"):
            return json.loads(bytes(checkpoint.raw(name).tolist()).decode())
    return None


def _bnb4_decode(checkpoint, base, device):
    import torch
    state = _bnb4_state(checkpoint, base)
    if state is None:
        raise SchemeError(f"{base}: a bitsandbytes 4-bit weight without its quant_state")
    packed = checkpoint.raw(base + ".weight").to(device).reshape(-1)
    code = checkpoint.raw(base + ".weight.quant_map").to(device).float()
    absmax = checkpoint.raw(base + ".weight.absmax").to(device)
    if checkpoint.has(base + ".weight.nested_absmax"):
        nested = checkpoint.raw(base + ".weight.nested_absmax").to(device).float()
        nested_code = checkpoint.raw(base + ".weight.nested_quant_map").to(device).float()
        values = nested_code[absmax.long()]
        size = int(state["nested_blocksize"])
        values = _blockwise(values, nested, size)
        absmax = values + torch.tensor(float(state["nested_offset"]), device=device)
    codes = torch.empty(packed.shape[0] * 2, dtype=torch.int64, device=device)
    codes[1::2] = (packed & 0xF).long()
    codes[::2] = (packed >> 4).long()
    shape = [int(n) for n in state["shape"]]
    values = code[codes][:math.prod(shape)]
    return _blockwise(values, absmax.float(), int(state["blocksize"])).reshape(shape)


def _blockwise(values, absmax, size):
    """values [n] times absmax [ceil(n / size)], block by block (bitsandbytes' FP32 order)."""
    full = values.shape[0] - values.shape[0] % size
    out = (values[:full].reshape(-1, size) * absmax[:full // size].reshape(-1, 1)).reshape(-1)
    if full < values.shape[0]:
        out = __import__("torch").cat((out, values[full:] * absmax[full // size]))
    return out


def _bnb8_decode(checkpoint, base, device):
    weight = checkpoint.raw(base + ".weight").to(device)
    scale = checkpoint.raw(base + ".SCB").to(device).float()
    return weight * scale.view(-1, 1) * 7.874015718698502e-3   # bitsandbytes' 1/127


def _nbits_decode(checkpoint, base, device):
    """ONNX Runtime's MatMulNBits: B [N, blocks, blob] of `bits`-bit codes (low bits first) along K in blocks of
    block_size, scales per block, zero points packed the same way (2 ** (bits - 1) when absent) or as floats, and
    an optional g_idx mapping each of K to its block."""
    import torch
    attributes = checkpoint.header(base + ".nbits_weight").attributes
    rows, cols = int(attributes["N"]), int(attributes["K"])
    bits, size = int(attributes["bits"]), int(attributes["block_size"])
    packed = checkpoint.raw(base + ".nbits_weight").to(device)
    blocks = packed.shape[1]
    words = packed.reshape(rows, -1).to(torch.int64)
    codes = _byte_codes(words, bits).reshape(rows, blocks, -1)[..., :size].reshape(rows, -1)[:, :cols]
    scales = checkpoint.raw(base + ".nbits_scales").to(device).float().reshape(rows, blocks)
    if checkpoint.has(base + ".nbits_zero_points"):
        zero = checkpoint.raw(base + ".nbits_zero_points").to(device)
        if zero.dtype == torch.uint8:
            zero = _byte_codes(zero.reshape(rows, -1).to(torch.int64), bits)[:, :blocks]
        zero = zero.reshape(rows, blocks).float()
    else:
        zero = torch.full((rows, blocks), float(1 << (bits - 1)), device=device)
    if checkpoint.has(base + ".nbits_g_idx"):
        group = checkpoint.raw(base + ".nbits_g_idx").to(device).long()
    else:
        group = torch.arange(cols, device=device) // size
    return (codes.float() - zero[:, group]) * scales[:, group]


def _byte_codes(data, bits):
    """`bits`-bit codes packed low bits first along the last dimension of bytes [..., n] (int64)."""
    import torch
    count = data.shape[-1] * 8 // bits
    positions = torch.arange(count, device=data.device) * bits
    byte, shift = positions // 8, positions % 8
    value = data[..., byte] >> shift
    spill = shift + bits > 8
    if bool(spill.any()):
        nxt = torch.clamp(byte + 1, max=data.shape[-1] - 1)
        value = value | torch.where(spill, data[..., nxt] << (8 - shift), 0)
    return value & ((1 << bits) - 1)


EXL3_HAD = 128          # the Hadamard blocks both sides of an EXL3 matrix are rotated by
EXL3_MASK = 0xFFFFFFFF


def exl3_codebook(words, codebook):
    """FP16 values of 16-bit trellis states (int64 tensor of 0..65535) in EXL3's procedural codebooks: 0 3INST, 1 MCG,
    2 MUL1, each the integer arithmetic of its definition and one FP16 rounding (a sum of two FP16 halves for 3INST and
    MCG, a fused multiply-add for MUL1), as the format's decoder rounds them."""
    import torch
    if codebook == 2:
        x = (words * 0x83DCD12D) & EXL3_MASK
        total = sum((x >> s) & 0xFF for s in (0, 8, 16, 24)) + 0x6400
        h = (total & 0xFFFF).to(torch.int16).view(torch.float16).double()
        k_inv = torch.tensor(0x1EEE, dtype=torch.int16).view(torch.float16).double()
        k_bias = torch.tensor(0xC931 - 0x10000, dtype=torch.int16).view(torch.float16).double()
        return (h * k_inv + k_bias).half()
    if codebook == 0:
        x = (words * 89226354 + 64248484) & EXL3_MASK
    else:
        x = (words * 0xCBAC1FED) & EXL3_MASK
    x = (x & 0x8FFF8FFF) ^ 0x3B603B60

    def half(bits):
        return torch.where(bits >= 0x8000, bits - 0x10000, bits).to(torch.int16).view(torch.float16).float()
    return (half(x & 0xFFFF) + half(x >> 16)).half()


def exl3_tiles(trellis, codebook, bits, half_bit):
    """The 16 x 16 tiles [k / 16, n / 16, 16, 16] (FP16) of a trellis [k / 16, n / 16, 16 K] (K = bits, or bits + 0.5 with
    half_bit): each tile a circular bitstream of 256 K bits, read most significant bit first from 32-bit words of two
    stored 16-bit words each (the first the low half); position p's state is the 16 bits ending at bit (p + 1) K (for
    a half-integer K, odd positions take bits + 1 bits, so position p ends at (p // 2)(2 bits + 1) + bits for even p,
    (p // 2 + 1)(2 bits + 1) for odd); positions map to the tile's rows and columns in the tensor-core fragment order:
    lane L = p // 8 and j = p % 8 give row 2 (L % 4) + j % 2 + 8 ((j // 2) % 2), column 2 (L // 8 + 4 (j // 4)) +
    (L // 4) % 2."""
    import torch
    device = trellis.device
    kt, nt, width = trellis.shape
    stored = trellis.to(torch.int64) & 0xFFFF
    words = (stored[..., 0::2] | (stored[..., 1::2] << 16)).reshape(kt * nt, width // 2)
    total = words.shape[1] * 32
    p = torch.arange(256, device=device)
    if half_bit:
        pair = 2 * bits + 1
        end = (p // 2) * pair + torch.where(p % 2 == 0, bits, pair)
    else:
        end = (p + 1) * bits
    end = end + total            # windows near the start wrap to the stream's end
    last = ((end - 1) // 32) % words.shape[1]
    first = ((end - 16) // 32) % words.shape[1]
    shift = (((end - 1) // 32) + 1) * 32 - end
    hi, lo = words[:, first], words[:, last]
    state = (((hi << 32) | lo) >> shift) & 0xFFFF
    values = exl3_codebook(state, codebook).view(kt * nt, 32, 8)
    lane = torch.arange(32, device=device)[:, None]
    j = torch.arange(8, device=device)[None, :]
    row = 2 * (lane % 4) + j % 2 + 8 * ((j // 2) % 2)
    col = 2 * (lane // 8 + 4 * (j // 4)) + (lane // 4) % 2
    tiles = torch.empty(kt * nt, 16, 16, dtype=torch.float16, device=device)
    tiles[:, row, col] = values
    return tiles.view(kt, nt, 16, 16)


def exl3_hadamard(device):
    """The normalized 128-point Sylvester Hadamard matrix (FP32)."""
    import torch
    h = torch.ones(1, 1, device=device)
    while h.shape[0] < EXL3_HAD:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / math.sqrt(EXL3_HAD)


def exl3_signs(packed, size, device):
    """FP32 +-1 of a sign bitfield (int16, bit i of word w for element 16 w + i: 1 is -1)."""
    import torch
    words = packed.to(device).view(-1).to(torch.int32) & 0xFFFF
    bits = (words[:, None] >> torch.arange(16, device=device)) & 1
    return (1.0 - 2.0 * bits.reshape(-1)[:size].float())


def _exl3_decode(checkpoint, base, device):
    """EXL3 trellis codes: base.trellis [in / 16, out / 16, 16 K] int16, the input and output sign
    vectors base.suh / base.svh (FP16) or their bitfields base.su / base.sv, the codebook named by a base.mcg or
    base.mul1 marker (3INST without one). W [in, out] = diag(su) H Q H diag(sv), Q the decoded tiles and H the
    block-diagonal normalized 128-point Hadamard on each side: the tiles exactly as the format decodes them (FP16),
    the rotations in FP32."""
    trellis = checkpoint.raw(base + ".trellis").to(device)
    kt, nt, width = trellis.shape
    k, n = 16 * kt, 16 * nt
    bits, half_bit = width // 16, width % 16 == 8
    def has(suffix):
        return checkpoint.has(base + suffix)
    codebook = 1 if has(".mcg") else 2 if has(".mul1") else 0
    if not 1 <= bits <= 8 or (width % 16 not in (0, 8)) or (half_bit and (codebook != 2 or bits > 3)):
        raise SchemeError(f"{base}: an EXL3 trellis of {width} words a tile is not a bitrate the format defines")
    if k % EXL3_HAD or n % EXL3_HAD:
        raise SchemeError(f"{base}: EXL3 matrices are multiples of {EXL3_HAD} on both sides, not {k} x {n}")
    su = (checkpoint.raw(base + ".suh").to(device).float() if has(".suh")
          else exl3_signs(checkpoint.raw(base + ".su"), k, device))
    sv = (checkpoint.raw(base + ".svh").to(device).float() if has(".svh")
          else exl3_signs(checkpoint.raw(base + ".sv"), n, device))
    q = exl3_tiles(trellis, codebook, bits, half_bit).permute(0, 2, 1, 3).reshape(k, n).float()
    h = exl3_hadamard(device)
    q = (h @ q.view(k // EXL3_HAD, EXL3_HAD, n)).view(k, n)              # rows: H Q
    q = (q.view(k, n // EXL3_HAD, EXL3_HAD) @ h).view(k, n)              # columns: (H Q) H
    return (su[:, None] * q * sv[None, :]).T.contiguous()


# Each scheme crit_core recognises, decoded to FP32 [out, in] with the format library's own arithmetic.
DECODERS = {
    "nvfp4": _nvfp4_decode,
    "fp8": _fp8_decode,
    "int": _ct_int_decode,
    "awq": _awq_decode,
    "gptq": _gptq_decode,
    "mlx": _mlx_decode,
    "bnb4": _bnb4_decode,
    "bnb8": _bnb8_decode,
    "nbits": _nbits_decode,
    "exl3": _exl3_decode,
    "plain": _plain_decode,
}
if set(DECODERS) != set(RECOGNISERS):
    raise ImportError(f"crit_core recognises {sorted(RECOGNISERS)} but crit-eval decodes {sorted(DECODERS)}")


def scheme(checkpoint, base):
    """The scheme name of matrix `base` in a reference checkpoint (crit_eval.quality.checkpoints.TensorCheckpoint), or None
    when it holds no matrix there."""
    found = stored(_Headers(checkpoint), base, checkpoint.quantization)
    return None if found is None or found[0] == "plain" else found[0]


def dense(checkpoint, base, device):
    """FP32 [out, in] weight of matrix `base` in a reference checkpoint."""
    found = stored(_Headers(checkpoint), base, checkpoint.quantization)
    if found is None:
        raise KeyError(f"the checkpoint holds no weight for {base!r}")
    return DECODERS[found[0]](checkpoint, base, device)


class _Headers:
    """A reference checkpoint's tensors seen as headers (.dtype as a safetensors name, .shape) for stored()."""

    def __init__(self, checkpoint):
        self.checkpoint = checkpoint

    def __contains__(self, name):
        return self.checkpoint.has(name)

    def get(self, name):
        if not self.checkpoint.has(name):
            return None
        return self.checkpoint.header(name)
