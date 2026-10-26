"""crit eval kernels: trace parsing, phase split, ranking and share of peak (no GPU)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crit_eval import kernels  # noqa: E402


def write_trace(directory, pid, offset, rows, sms=170):
    lines = [f"#clock 1000 {1000 + offset}"]
    lines += [f"{s} {e} {blocks} 256 {name}" for s, e, blocks, name in rows]
    lines += [f"#clock 9000000000 {9000000000 + offset}", f"#device {sms} Test GPU"]
    (Path(directory) / f"trace-{pid}.txt").write_text("\n".join(lines) + "\n")


class KernelTraceTest(unittest.TestCase):
    def test_phase_split_ranking_and_peaks(self):
        with tempfile.TemporaryDirectory() as directory:
            # CUPTI clock runs 5,000 ns behind realtime; marks are realtime.
            write_trace(directory, 1, 5_000, [
                (10_000, 1_010_000, 340, "gemm_kernel"),        # prefill, 1 ms
                (1_100_000, 1_600_000, 96, "scan_kernel"),      # prefill, 0.5 ms, underfills 170 SMs
                (2_100_000, 2_200_000, 400, "gemv_kernel"),     # decode
                (2_300_000, 2_400_000, 400, "gemv_kernel"),     # decode
            ])
            marks = {"info": {"name": "toy", "kernel_work": [
                        {"match": "gemm", "flops_per_token": 1e9, "format": "fp8"},
                        {"match": "gemv", "bytes_per_call": 1e8}]},
                     "cases": [{"id": "c", "prompt_tokens": 1000, "output_tokens": 2, "rounds": 2,
                                "start": 0, "first": 2_000_000, "end": 3_000_000}]}
            found, device = kernels.read_traces(directory)
            self.assertEqual(device, {"sms": 170, "name": "Test GPU"})
            self.assertEqual(found[0][0], 15_000)  # converted to the realtime clock
            self.assertEqual(kernels.read_traces(directory, "cupti")[0][0][0], 10_000)  # marks on the tracer's clock
            result = kernels.analyze(found, device, marks, {"tflops": {"fp8": 2000.0}, "bandwidth_gbs": 4000.0})
        prefill, decode = result["phases"]["prefill"], result["phases"]["decode"]
        self.assertEqual([r["kernel"] for r in prefill["kernels"]], ["gemm_kernel", "scan_kernel"])
        self.assertAlmostEqual(prefill["kernel_ms"], 1.5)
        self.assertAlmostEqual(prefill["wall_ms"], 2.0)
        gemm, scan = prefill["kernels"]
        self.assertAlmostEqual(gemm["tflops"], 1000.0)  # 1e12 FLOPs in 1 ms
        self.assertAlmostEqual(gemm["flops_of_peak"], 0.5)
        self.assertFalse(gemm["underfills"])
        self.assertTrue(scan["underfills"])
        (gemv,) = decode["kernels"]
        self.assertEqual(gemv["calls"], 2)
        self.assertAlmostEqual(gemv["gbs"], 1000.0)  # 2e8 bytes in 0.2 ms
        self.assertAlmostEqual(gemv["bandwidth_of_peak"], 0.25)
        self.assertEqual(result["cases"]["c"]["prefill"]["top"][0]["kernel"], "gemm_kernel")
        text = kernels.render({"engine": {"name": "toy"}, **result})
        self.assertIn("only 96 blocks for 170 SMs", text)
        self.assertIn("50% of peak", text)

    def test_overlapping_launch_is_charged_only_after_its_predecessor(self):
        with tempfile.TemporaryDirectory() as directory:
            # norm_kernel launches early (programmatic dependent launch) and waits 0.9 ms on gemm_kernel.
            write_trace(directory, 1, 0, [
                (0, 1_000_000, 340, "gemm_kernel"),
                (100_000, 1_100_000, 340, "norm_kernel"),
                (1_200_000, 1_300_000, 340, "gemm_kernel"),
            ])
            marks = {"info": {"name": "toy"}, "cases": [{"id": "c", "prompt_tokens": 10, "output_tokens": 1,
                                                          "start": 0, "first": 2_000_000, "end": 2_000_001}]}
            found, device = kernels.read_traces(directory)
            result = kernels.analyze(found, device, marks)
        prefill = result["phases"]["prefill"]
        rows = {r["kernel"]: r for r in prefill["kernels"]}
        self.assertAlmostEqual(rows["norm_kernel"]["ms"], 0.1)
        self.assertAlmostEqual(rows["norm_kernel"]["span_ms"], 1.0)
        self.assertAlmostEqual(rows["gemm_kernel"]["ms"], 1.1)
        self.assertAlmostEqual(prefill["kernel_ms"], 1.2)  # never more than the GPU was busy
        self.assertAlmostEqual(prefill["idle_ms"], 0.8)
        self.assertIn("not charged", kernels.render({"engine": {"name": "toy"}, **result}))

    def test_prefill_is_a_fresh_one_token_request_after_a_warmup_no_cache_matches(self):
        calls = []

        class Engine:
            def generate(self, ids, n, on_tokens):
                calls.append((list(ids), n))
                on_tokens([7])
                return {"tokens": [7] * n, "rounds": n}

            def close(self):
                pass
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "prepared.json").write_text(json.dumps(
                {"speed": [{"id": "c", "prompt_ids": [1, 2, 3, 4, 5], "output_tokens": 3}]}))
            spec = {"data": directory, "adapter": "unused", "model": "m", "config": {}, "cases": [], "dir": directory}
            with patch.dict(kernels.os.environ, {"CUDA_INJECTION64_PATH": "/no/such/library.so"}), \
                    patch("crit_eval.runner.load_adapter"), \
                    patch("crit_eval.runner.open_engine", return_value=(Engine(), {"name": "toy"})), \
                    patch.object(kernels.time, "sleep"):
                kernels._child(spec)
            marks = json.loads((Path(directory) / "marks.json").read_text())
        # The warmup's prompt has the case's length; the prefill's begins with a token no request began with before
        # (not the warmup's 3, not the case's own 1); then the whole request on the case's prompt.
        self.assertEqual(calls, [([3, 4, 5, 1, 2], 3), ([2, 2, 3, 4, 5], 1), ([1, 2, 3, 4, 5], 3)])
        self.assertEqual(marks["clock"], "realtime")   # no tracer library loaded: the host's clock
        case = marks["cases"][0]
        self.assertLessEqual(case["prefill_start"], case["prefill_first"])
        self.assertLessEqual(case["prefill_end"], case["start"])
        self.assertEqual(kernels.warmup_prompt([9]), [9])
        self.assertEqual(kernels.fresh_prompt([1, 2, 3], {1, 2, 3}), [4, 2, 3])
        self.assertEqual(kernels.fresh_prompt([5], set()), [6])

    def test_a_speculative_round_before_the_first_token_is_not_prefill(self):
        with tempfile.TemporaryDirectory() as directory:
            write_trace(directory, 1, 0, [
                (100_000, 900_000, 340, "gemm_nvfp4_kernel"),     # the one-token request: prefill matmul
                (900_000, 1_000_000, 340, "flash_prefill_kernel"),   # its attention
                (1_000_000, 1_050_000, 170, "rmsnorm_kernel"),
                (1_050_000, 1_100_000, 170, "mtp_draft_gemv"),       # a drafter primed after the prefill
                (2_000_000, 2_100_000, 340, "gemm_nvfp4_kernel"),     # the whole request: its (cached) prefill
                (2_100_000, 2_400_000, 340, "gemv_kernel"),           # the first verify round, before the first token
                (2_600_000, 2_700_000, 340, "gemv_kernel"),           # decode
            ])
            marks = {"info": {"name": "toy", "kernel_work": [{"match": "gemm_nvfp4", "term": "matmul", "format": "nvfp4"}]},
                     "cases": [{"id": "c", "prompt_tokens": 100, "output_tokens": 4, "rounds": 2,
                                "prefill_start": 0, "prefill_first": 1_000_000, "prefill_end": 1_200_000,
                                "start": 2_000_000, "first": 2_500_000, "end": 3_000_000}]}
            found, device = kernels.read_traces(directory)
            result = kernels.analyze(found, device, marks)
        prefill = result["cases"]["c"]["prefill"]
        self.assertEqual(prefill["terms"], {"matmul:nvfp4": 0.8, "attention": 0.1, "other": 0.05, "draft": 0.05})
        self.assertEqual(prefill["classified_ms"], {"rules": 0.8, "names": 0.2})
        self.assertAlmostEqual(prefill["first_token_ms"], 1.0)
        self.assertAlmostEqual(prefill["gpu_before_first_ms"], 0.9)
        self.assertNotIn("gemv_kernel", [r["kernel"] for r in result["phases"]["prefill"]["kernels"]])
        self.assertEqual(result["cases"]["c"]["decode"]["terms"], {"matmul": 0.1})   # by name: a gemv
        self.assertTrue(result["prefill_request"].startswith("one output token on a fresh prompt"))
        text = kernels.render({"engine": {"name": "toy"}, **result})
        self.assertIn("matmul:nvfp4 0.8 ms, attention 0.1 ms", text)
        self.assertIn("first token 1.0 ms, GPU busy 0.9 ms of it", text)
        self.assertEqual(kernels.classify(None, "void ns::(anonymous namespace)::delta_scan_kernel<128>"),
                         ("recurrent", None, "names"))
        self.assertEqual(kernels.classify(None, "convert_fp16_kernel"), ("other", None, "names"))
        self.assertEqual(prefill["unclassed"], [{"kernel": "rmsnorm_kernel", "ms": 0.05}])
        self.assertIn("other by name: rmsnorm_kernel 0.1 ms (class them with kernel_work rules)", text)

    def test_projections_named_by_scheme_are_matmul_with_the_format_their_name_gives(self):
        for name, found in (
                # crit's stored-tile projections: the BF16-activation prompt GEMM and the decode GEMV.
                ("void crit::stored_tile::(anonymous namespace)::a16_kernel<true, false>(crit::A16Args)", ("matmul", None)),
                ("void crit::stored_tile::(anonymous namespace)::gemv_kernel<false, 1, 8, 2, 4>(Args)", ("matmul", None)),
                ("void marlin::Marlin<256, 4>(int4 const*)", ("matmul", None)),
                ("w4a16_gemm_kernel", ("matmul", None)),
                ("void crit::stored_tile::gemm_fp4_kernel(unsigned char const*)", ("matmul", "nvfp4")),
                ("cutlass::Kernel2<GemmUniversal<e4m3_t, float>>", ("matmul", "fp8")),
                ("w4a8_gemm_fp4_fp8_kernel", ("matmul", "nvfp4")),   # FP4 weights times FP8 activations
                ("fp8_kv_prefill_kernel", ("attention", None))):      # a format in the name of another term
            self.assertEqual(kernels.classify(None, name)[:2], found, name)

    def test_short_name_drops_parameters_not_templates(self):
        self.assertEqual(kernels.short_name("void ns::k<a(b), 2>(float const*, int)"), "ns::k<a(b), 2>")
        self.assertTrue(kernels.short_name("x" * 200).endswith("..."))

    def test_short_name_keeps_anonymous_namespaces(self):
        # c++filt writes a kernel in an unnamed namespace as "(anonymous namespace)::"; that parenthesis is part of
        # the name, not the parameter list.
        self.assertEqual(kernels.short_name("void ns::(anonymous namespace)::gemm_kernel<true, 8>(ns::Args)"),
                         "ns::(anonymous namespace)::gemm_kernel<true, 8>")
        self.assertEqual(kernels.short_name("(anonymous namespace)::k(int)"), "(anonymous namespace)::k")

    def test_internal_linkage_prefix(self):
        self.assertEqual(kernels._mangled("__nv_static_54__110d7f6d_27_f_cu_1f652c57_6886__ZN2ns1kEv"), "_ZN2ns1kEv")
        self.assertEqual(kernels._mangled("_ZN2ns1kEv"), "_ZN2ns1kEv")


if __name__ == "__main__":
    unittest.main()
