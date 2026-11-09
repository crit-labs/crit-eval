"""Reference packages (crit_eval.quality.packages): prepared data and reference scores packed once, fetched into a run's data
byte for byte, found by the checkpoint's and the suite's identity, and refused when a file differs from its record."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("", "tests"): sys.path.insert(0, str(ROOT / _folder))
from crit_eval.quality import fixed, packages
from crit_eval.quality import prepare  # noqa: E402
from crit_eval import EVAL_DIR, load_json  # noqa: E402

MODEL, SUITE = "a" * 64, "b" * 64
PROTOCOL = load_json(EVAL_DIR / "protocol.json")["protocol"]
REFERENCE = 3


def scored(data, model=MODEL, suite=SUITE, protocol=PROTOCOL):
    """A prepared data folder with stored reference scores of one text."""
    data = Path(data)
    data.mkdir(parents=True, exist_ok=True)
    (data / "prepared.json").write_text(json.dumps({"protocol": protocol, "suite_sha256": suite,
                                                    "model": {"sha256": model}}))
    fixed.write(data, {"reference": REFERENCE, "model_sha256": model, "top_k": 2},
                {"text": ([-0.5, -1.5], [[3, 4], [5, 6]], [[-0.1, -0.2], [-0.3, -0.4]])})
    return data


class PackageTests(unittest.TestCase):
    def identities(self, model=MODEL, suite=SUITE):
        return (patch.object(prepare, "model_identity", lambda path, chat_template=None: {"sha256": model}),
                patch.object(prepare, "suite_identity", lambda: suite),
                patch("crit_eval.quality.reference.REFERENCE_VERSION", REFERENCE))

    def test_a_packed_checkpoint_is_fetched_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = scored(root / "data")
            first, second, third = self.identities()
            with first, second, third:
                entry = packages.pack(data, root / "packages")
                self.assertEqual(entry["name"], packages.name(MODEL, SUITE))
                self.assertEqual(sorted(entry["files"]), ["prepared.json", "reference/index.json",
                                                          "reference/" + fixed.text_file("text")])
                self.assertEqual(packages.fetch("checkpoint", root / "run", root / "packages", log=lambda m: None),
                                 entry["name"])
            for relative in entry["files"]:
                self.assertEqual((root / "run" / relative).read_bytes(), (data / relative).read_bytes(), relative)

    def test_a_result_names_the_package_its_reference_came_from_until_it_is_scored_again(self):
        from crit_eval.run import quality_checks
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, third = self.identities()
            with first, second, third:
                entry = packages.pack(scored(root / "data"), root / "packages")
                packages.fetch("checkpoint", root / "run", root / "packages", log=lambda m: None)
            origin = quality_checks.reference_origin(root / "run")
            self.assertEqual(origin["reference"], REFERENCE)
            self.assertEqual(origin["package"]["name"], entry["name"])
            self.assertIsNone(quality_checks.reference_origin(root / "data")["package"])   # scored here
            index = root / "run" / "reference" / "index.json"
            index.write_bytes(index.read_bytes() + b" ")
            self.assertIsNone(quality_checks.reference_origin(root / "run")["package"])
            self.assertIsNone(quality_checks.reference_origin(root / "nowhere"))

    def test_a_package_of_another_protocol_series_is_not_fetched(self):
        # A package prepared under an earlier series would replace the run's prepared data with data the run refuses
        # (prepare.check_prepared): the reference is scored locally instead, and the run's data is left as it was.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, third = self.identities()
            with first, second, third:
                packages.pack(scored(root / "data", protocol="crit-eval-v0.1.0"), root / "packages")
                run = scored(root / "run")
                before = (run / "prepared.json").read_bytes()
                self.assertIsNone(packages.find("checkpoint", root / "packages"))
                self.assertIsNone(packages.fetch("checkpoint", run, root / "packages", log=lambda m: None))
            self.assertEqual((run / "prepared.json").read_bytes(), before)

    def test_another_checkpoint_or_suite_has_no_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("crit_eval.quality.reference.REFERENCE_VERSION", REFERENCE):
                packages.pack(scored(root / "data"), root / "packages")
            for model, suite in (("c" * 64, SUITE), (MODEL, "d" * 64)):
                first, second, third = self.identities(model, suite)
                with first, second, third:
                    self.assertIsNone(packages.fetch("checkpoint", root / "run", root / "packages"))
            first, second, third = self.identities()
            with first, second, third:
                self.assertIsNone(packages.fetch("checkpoint", root / "run", root / "nowhere"))
            self.assertFalse((root / "run").exists())

    def test_a_file_that_differs_from_its_record_is_refused_before_anything_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, third = self.identities()
            with first, second, third:
                entry = packages.pack(scored(root / "data"), root / "packages")
                (root / "packages" / entry["name"] / "reference" / fixed.text_file("text")).write_bytes(b"tampered")
                with self.assertRaisesRegex(SystemExit, "differs from the sha256"):
                    packages.fetch("checkpoint", root / "run", root / "packages")
            self.assertFalse((root / "run" / "prepared.json").exists())

    def test_data_prepared_from_another_model_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, third = self.identities()
            with first, second, third:
                packages.pack(scored(root / "data"), root / "packages")
                scored(root / "run", model="c" * 64)
                with self.assertRaisesRegex(SystemExit, "another model or suite"):
                    packages.fetch("checkpoint", root / "run", root / "packages")

    def test_the_scores_carry_their_spread_and_allowance(self):
        # The reference's own spread and the allowance (scored with the activations the checkpoint declares) travel
        # with the scores they were measured against; ones of other scores (an earlier reference, say) stay behind.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = scored(root / "data")
            identity = {"reference": REFERENCE, "model_sha256": MODEL, "top_k": 2}
            spread = {"long-1/prompt": {"kld_mean": 0.03, "kld_p99": 0.7, "ppl_ratio": 1.01}}
            allowance = {"activations": ["nvfp4"], "sets": spread}
            first, second, third = self.identities()
            with first, second, third:
                entry = packages.pack(data, root / "packages")
                self.assertNotIn(f"{fixed.STORE}/{fixed.SPREAD}", entry["files"])   # none stored
                fixed.write_spread(data, {**identity, "reference": REFERENCE - 1, "spread_perturbation": "tf32"}, spread)
                fixed.write_allowance(data, {**identity, "reference": REFERENCE - 1}, allowance)
                entry = packages.pack(data, root / "packages")
                self.assertFalse({f"{fixed.STORE}/{fixed.SPREAD}", f"{fixed.STORE}/{fixed.ALLOWANCE}"} & set(entry["files"]))
                fixed.write_spread(data, {**identity, "spread_perturbation": "tf32", "spread_texts": ["text"]}, spread)
                fixed.write_allowance(data, {**identity, "allowance": "declared_activations"}, allowance)
                entry = packages.pack(data, root / "packages")
                self.assertIn(f"{fixed.STORE}/{fixed.SPREAD}", entry["files"])
                self.assertIn(f"{fixed.STORE}/{fixed.ALLOWANCE}", entry["files"])
                packages.fetch("checkpoint", root / "run", root / "packages", log=lambda m: None)
            self.assertEqual(fixed.load_spread(root / "run", {"reference": REFERENCE}), spread)
            self.assertEqual(fixed.load_allowance(root / "run", identity), allowance)

    def test_scores_of_another_reference_are_not_packed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("crit_eval.quality.reference.REFERENCE_VERSION", REFERENCE + 1):
                with self.assertRaisesRegex(SystemExit, "not this crit's reference"):
                    packages.pack(scored(root / "data"), root / "packages")


if __name__ == "__main__":
    unittest.main()
