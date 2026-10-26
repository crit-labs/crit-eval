"""Reference packages (crit_eval.packages): prepared data and reference scores packed once, fetched into a run's data
byte for byte, found by the checkpoint's and the suite's identity, and refused when a file differs from its record."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("", "tests"): sys.path.insert(0, str(ROOT / _folder))
from crit_eval import fixed, packages, prepare  # noqa: E402

MODEL, SUITE = "a" * 64, "b" * 64
REFERENCE = 3


def scored(data, model=MODEL, suite=SUITE):
    """A prepared data folder with stored reference scores of one text."""
    data = Path(data)
    data.mkdir(parents=True, exist_ok=True)
    (data / "prepared.json").write_text(json.dumps({"protocol": "crit-eval-v7", "suite_sha256": suite,
                                                    "model": {"sha256": model}}))
    fixed.write(data, {"reference": REFERENCE, "model_sha256": model, "top_k": 2},
                {"text": ([-0.5, -1.5], [[3, 4], [5, 6]], [[-0.1, -0.2], [-0.3, -0.4]])})
    return data


class PackageTests(unittest.TestCase):
    def identities(self, model=MODEL, suite=SUITE):
        return (patch.object(prepare, "model_identity", lambda path, chat_template=None: {"sha256": model}),
                patch.object(prepare, "suite_identity", lambda: suite),
                patch("crit_eval.reference.REFERENCE_VERSION", REFERENCE))

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
        from crit_eval import runner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, third = self.identities()
            with first, second, third:
                entry = packages.pack(scored(root / "data"), root / "packages")
                packages.fetch("checkpoint", root / "run", root / "packages", log=lambda m: None)
            origin = runner.reference_origin(root / "run")
            self.assertEqual(origin["reference"], REFERENCE)
            self.assertEqual(origin["package"]["name"], entry["name"])
            self.assertIsNone(runner.reference_origin(root / "data")["package"])   # scored here
            index = root / "run" / "reference" / "index.json"
            index.write_bytes(index.read_bytes() + b" ")
            self.assertIsNone(runner.reference_origin(root / "run")["package"])
            self.assertIsNone(runner.reference_origin(root / "nowhere"))

    def test_another_checkpoint_or_suite_has_no_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("crit_eval.reference.REFERENCE_VERSION", REFERENCE):
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

    def test_scores_of_another_reference_are_not_packed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("crit_eval.reference.REFERENCE_VERSION", REFERENCE + 1):
                with self.assertRaisesRegex(SystemExit, "not this crit's reference"):
                    packages.pack(scored(root / "data"), root / "packages")


if __name__ == "__main__":
    unittest.main()
