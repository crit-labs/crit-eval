"""Machine settings (crit_eval.machine_settings): the settings.toml of the crit project crit names in CRIT_PROJECT, the
one its runs read (an installed crit's too), else of the checkouts crit-eval runs from."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import crit_eval  # noqa: E402
from crit_eval.quality import packages  # noqa: E402

PROJECT_SETTINGS = """vram_headroom_gib = 2
ram_headroom_gib = 8
context_tokens = 65536
eval_packages = "/srv/crit/packages"
wsl_run_root = "/srv/crit/runs"
"""


class MachineSettingsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.project = Path(temp.name).resolve() / "project"
        self.project.mkdir()
        (self.project / "settings.toml").write_text(PROJECT_SETTINGS, encoding="utf-8")
        checkout = Path(temp.name).resolve() / "checkout"
        checkout.mkdir()
        (checkout / "settings.toml").write_text("vram_headroom_gib = 4\ncontext_tokens = 50000\n", encoding="utf-8")
        patched = patch.object(crit_eval, "checkouts", return_value=[checkout])
        patched.start()
        self.addCleanup(patched.stop)

    def named(self, folder):
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        if folder is None:
            os.environ.pop(crit_eval.PROJECT_VARIABLE, None)
        else:
            os.environ[crit_eval.PROJECT_VARIABLE] = str(folder)

    def test_the_project_crit_names_gives_the_settings_over_the_checkouts(self):
        self.named(self.project)
        self.assertEqual(crit_eval.machine_settings(), {"vram_headroom_gib": 2.0, "ram_headroom_gib": 8.0,
                                                        "context_tokens": 65536, "eval_packages": "/srv/crit/packages"})
        self.assertEqual(crit_eval.requested_context(), 65536)
        self.assertEqual(packages.source(), "/srv/crit/packages")

    def test_a_named_project_without_settings_has_none_of_the_checkouts(self):
        self.named(self.project / "elsewhere")
        self.assertEqual(crit_eval.machine_settings(), {})
        self.assertEqual(crit_eval.requested_context(), crit_eval.DEFAULT_CONTEXT_TOKENS)

    def test_without_a_named_project_the_checkouts_settings_stand(self):
        self.named(None)
        self.assertEqual(crit_eval.machine_settings(), {"vram_headroom_gib": 4.0, "context_tokens": 50000})


if __name__ == "__main__":
    unittest.main()
