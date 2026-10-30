"""crit eval workbench on the host: the resource report, and the harness run in a folder's own Python. The harness's
GPU measurements are checked on a device with a folder crit kernels new made."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crit_eval import workbench  # noqa: E402
from crit_eval.cli import main  # noqa: E402


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / "kernel"
        self.folder.mkdir()

    def test_the_resource_report_reads_cuobjdump(self):
        text = ("Resource usage:\n Common:\n  GLOBAL:0\n Function _Z6kernelPf:\n  REG:218 STACK:0 SHARED:13312 LOCAL:0 "
                "CONSTANT[0]:392 TEXTURE:0\n Function _Z5otherv:\n  REG:40 STACK:16 SHARED:0 LOCAL:24\n")
        self.assertEqual(workbench.resource_usage(text), {
            "_Z6kernelPf": {"registers": 218, "stack": 0, "shared": 13312, "local": 0},
            "_Z5otherv": {"registers": 40, "stack": 16, "shared": 0, "local": 24}})

    def test_the_folders_harness_runs_and_its_result_comes_back(self):
        # A stand-in harness that records its arguments as the result.
        (self.folder / "harness.py").write_text(
            "import json, sys\nargs = sys.argv[1:]\nout = args[args.index('--out') + 1]\n"
            "json.dump({'kind': 'kernels_check', 'args': args[:-2], 'passed': False}, open(out, 'w'))\nsys.exit(1)\n",
            encoding="utf-8")
        code, result = workbench.run_harness(self.folder, "check")
        self.assertEqual((code, result["args"]), (1, ["check"]))
        self.assertTrue((self.folder / "check.json").is_file())
        code, result = workbench.run_harness(self.folder, "bench", shape=2, iters=7, out=self.folder / "b.json")
        self.assertEqual(result["args"], ["bench", "--shape", "2", "--iters", "7"])
        self.assertEqual(main(["workbench", "check", str(self.folder)]), 1)   # a failed check fails the command
        with self.assertRaises(SystemExit):
            workbench.run_harness(Path(self.temp.name), "check")   # no harness: no workbench folder
        with self.assertRaisesRegex(SystemExit, "holds no compiled kernel"):
            workbench.profile(self.folder)

    def test_the_harness_ships_with_the_package(self):
        text = workbench.HARNESS.read_text(encoding="utf-8")
        for part in ("def check(", "def bench(", "copies_for", "repeatable", "column_invariant"):
            self.assertIn(part, text)
        self.assertIn("workbench/harness.py", (ROOT / "pyproject.toml").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
