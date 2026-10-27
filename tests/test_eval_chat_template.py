"""The chat template Eval renders prompts with for a model that carries none: crit's own for its family (DeepSeek V4)
or `crit eval prepare --chat-template FILE`; prepared.json records it, the model's identity includes it, and runs,
references and packages of the checkpoint agree on it (offline)."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for _folder in ("", "tests"): sys.path.insert(0, str(ROOT / _folder))

import core_text_fixtures as fx  # noqa: E402
from crit_core.text import ModelTextError, load_model_text, shipped_template  # noqa: E402
from crit_eval.run import chat
from crit_eval import cli  # noqa: E402
from crit_eval.quality import prepare  # noqa: E402
from crit_eval.quality import fixed, packages
from crit_eval.quality.reference import REFERENCE_VERSION  # noqa: E402

TEMPLATE = ("{{ bos_token }}{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}"
            "{% if add_generation_prompt %}<reply>{% endif %}")


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def without_template(directory, model_type="llama"):
    """A small model folder whose files hold no chat template, and no weights."""
    files = fx.llama_files(**{"config.json": {"model_type": model_type, "eos_token_id": 1}})
    del files["chat_template.jinja"]
    return fx.write_files(directory, files)


def quiet(*argv):
    """`crit eval ARGV`'s output. The speed cases that cut a document to an exact prompt length render without the
    document: their lengths suit real tokenizers, not this fixture's, and the cut is no part of the template."""
    speed_entry = prepare.speed_entry

    def uncut(model, case, output_tokens):
        return speed_entry(model, {k: v for k, v in case.items() if k not in ("document", "prompt_tokens")},
                           output_tokens)
    with contextlib.redirect_stdout(io.StringIO()) as out, patch.object(prepare, "speed_entry", uncut):
        cli.main(list(argv))
    return out.getvalue()


class IdentityTests(unittest.TestCase):
    def test_the_template_a_model_renders_with_is_part_of_its_identity_when_it_is_none_of_its_files(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b, \
                tempfile.TemporaryDirectory() as c:
            deepseek, plain = without_template(a, "deepseek_v4"), without_template(b)
            own = fx.write_files(c, fx.llama_files())
            crit = prepare.model_identity(deepseek)
            self.assertEqual(crit["files"]["chat_template"], sha(shipped_template("deepseek_v4")[1]))
            given = prepare.model_identity(plain, TEMPLATE)
            self.assertEqual(given["files"]["chat_template"], sha(TEMPLATE))
            self.assertNotEqual(given["sha256"], prepare.model_identity(plain)["sha256"])
            self.assertNotIn("chat_template", prepare.model_identity(plain)["files"])
            # A model's own template is identified by its files, as before.
            self.assertNotIn("chat_template", prepare.model_identity(own)["files"])
            with self.assertRaisesRegex(ModelTextError, "carries its own chat template"):
                prepare.model_identity(own, TEMPLATE)


class ChatTemplateOptionTests(unittest.TestCase):
    def test_prepare_renders_with_the_given_template_and_records_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, data = without_template(root / "model"), root / "data"
            template = root / "plain.jinja"
            template.write_text(TEMPLATE, encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, r"carries no chat template .*--chat-template FILE"):
                quiet("prepare", "--model", str(model), "--data", str(data))
            printed = quiet("prepare", "--model", str(model), "--data", str(data), "--chat-template", str(template))
            self.assertIn("chat template: plain.jinja (--chat-template)", printed)
            prepared = json.loads((data / "prepared.json").read_text(encoding="utf-8"))
            self.assertEqual(prepared["chat_template"], {"origin": "file", "file": "plain.jinja",
                                                         "sha256": {"default": sha(TEMPLATE)}, "source": TEMPLATE})
            self.assertEqual(prepared["model"], prepare.model_identity(model, TEMPLATE))
            self.assertEqual(prepare.prepared_template(prepared), TEMPLATE)
            # Every prompt renders with it, as a server loading the data's template renders it.
            text = load_model_text(model, clock=None, chat_template=TEMPLATE)
            case = prepared["speed"][0]
            self.assertEqual(case["prompt_ids"], text.encode_chat(case["messages"], add_generation_prompt=True))
            self.assertEqual(chat.Server("http://127.0.0.1:1/v1", prepared).chat_template, TEMPLATE)
            self.assertEqual(chat.load_text(model, TEMPLATE).templates["default"].source, TEMPLATE)
            # A run renders with the data's template: another --chat-template, or one for a model with its own, fails.
            prepare.check_template(prepared, model, data)
            prepare.check_template(prepared, model, data, TEMPLATE)
            with self.assertRaisesRegex(SystemExit, "another chat template than --chat-template's"):
                prepare.check_template(prepared, model, data, TEMPLATE + " ")
            own = fx.write_files(root / "own", fx.llama_files())
            with self.assertRaisesRegex(SystemExit, "carries its own chat template"):
                quiet("prepare", "--model", str(own), "--data", str(root / "other"), "--chat-template", str(template))

    def test_a_deepseek_v4_checkpoint_renders_crits_template_and_a_package_of_it_is_found_by_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, data = without_template(root / "model", "deepseek_v4"), root / "data"
            self.assertIn("chat template: crit's deepseek-v4.jinja (the model carries none)",
                          quiet("prepare", "--model", str(model), "--data", str(data)))
            prepared = json.loads((data / "prepared.json").read_text(encoding="utf-8"))
            name, source = shipped_template("deepseek_v4")
            self.assertEqual(prepared["chat_template"], {"origin": "crit", "file": name,
                                                         "sha256": {"default": sha(source)}, "source": source})
            self.assertTrue(prepared["heldout"][0]["prompt_ids"])
            self.assertIsNone(prepare.prepared_template(prepared))
            prepare.check_template(prepared, model, data)
            # Data from an earlier crit's template no longer matches what this crit renders.
            stale = {**prepared, "chat_template": {**prepared["chat_template"], "sha256": {"default": "0" * 64}}}
            with self.assertRaisesRegex(SystemExit, "another chat template than crit renders"):
                prepare.check_template(stale, model, data)
            # The package of the data is found by the checkpoint and the template that identify it.
            fixed.write(data, {"reference": REFERENCE_VERSION, "model_sha256": prepared["model"]["sha256"], "top_k": 2},
                        {"text": ([-0.5, -1.5], [[3, 4], [5, 6]], [[-0.1, -0.2], [-0.3, -0.4]])})
            entry = packages.pack(data, root / "packages")
            self.assertEqual(packages.find(model, root / "packages")[0], entry["name"])
            self.assertIsNone(packages.find(model, root / "packages", TEMPLATE))


if __name__ == "__main__":
    unittest.main()
