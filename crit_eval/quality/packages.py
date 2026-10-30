"""Reference packages: a checkpoint's prepared data and its stored reference scores, computed once and fetched by
every crit-eval of that checkpoint, so a run needs only the engine (`crit eval pack`, `crit eval fetch`).

A package source is a folder or an HTTPS base URL (a Hugging Face dataset's `resolve/main/`, say) holding
packages.json and one folder per package, named by the first 16 hex digits of the model's and the suite's sha256:
prepared.json (crit_eval.quality.prepare) and reference/ (crit_eval.quality.fixed), with the reference's own spread
(spread.json) and the as_quantized quality mode's allowance (allowance.json) when the data holds them. packages.json
lists each package with the
identities it was made under (model_sha256, suite_sha256, protocol, reference version) and the sha256 of every file,
which fetch checks before it writes anything into the data. The scores do not depend on the device that computed
them beyond rounding far below the drift budget, and a run reads them under the model, the suite and the reference
version alone (runner.run), so one package serves every GPU. The model is identified as prepare identifies it
(model_identity: its config, tokenizer, template and weight file names and sizes), so a checkpoint is found without
hashing its weights. A package prepared with --chat-template is found with the same template.
"""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from .. import load_json, write_json

INDEX = "packages.json"
PACKAGE = "reference/package.json"   # in a data folder: the package its reference scores were fetched from
SOURCE_SETTING = "eval_packages"
LOCAL = "reference-packages"   # beside crit_eval in a checkout: the packages made there, not in git
# The published packages (a Hugging Face dataset of crit-labs), the source when nothing else names one.
PUBLISHED = "https://huggingface.co/datasets/crit-labs/crit-eval-references/resolve/main"


def name(model_sha, suite_sha):
    return f"{model_sha[:16]}-{suite_sha[:16]}"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def pack(data, packages):
    """Add prepared data folder `data`, scored by the reference, to package folder `packages` (created when missing)
    and list it in its packages.json. Returns the package's entry."""
    from . import fixed
    from .reference import REFERENCE_VERSION
    data, packages = Path(data), Path(packages)
    prepared = load_json(data / "prepared.json")
    index = data / fixed.STORE / "index.json"
    if not index.is_file():
        raise SystemExit(f"{data} holds no reference scores: score them first (crit eval reference --data {data})")
    stored = load_json(index)
    if stored["identity"].get("reference") != REFERENCE_VERSION or stored["identity"].get("model_sha256") != \
            prepared["model"]["sha256"]:
        raise SystemExit(f"{data}: its reference scores are not this crit's reference of the prepared model")
    files = ["prepared.json", f"{fixed.STORE}/index.json"] + [f"{fixed.STORE}/{f}" for f, _ in stored["files"].values()]
    # The scores' own spread (reference.store_spread) and allowance (reference.store_allowance), when they were
    # measured against these scores.
    for extra in (fixed.SPREAD, fixed.ALLOWANCE):
        path = data / fixed.STORE / extra
        if path.is_file() and all((load_json(path).get("identity") or {}).get(key) == value
                                  for key, value in stored["identity"].items()):
            files.append(f"{fixed.STORE}/{extra}")
    key = name(prepared["model"]["sha256"], prepared["suite_sha256"])
    target = packages / key
    shutil.rmtree(target, ignore_errors=True)
    entry = {"model_sha256": prepared["model"]["sha256"], "suite_sha256": prepared["suite_sha256"],
             "protocol": prepared["protocol"], "reference": REFERENCE_VERSION, "files": {}, "bytes": 0}
    for relative in files:
        raw = (data / relative).read_bytes()
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        (target / relative).write_bytes(raw)
        entry["files"][relative] = _sha(raw)
        entry["bytes"] += len(raw)
    listing = load_json(packages / INDEX) if (packages / INDEX).is_file() else {"packages": {}}
    listing["packages"][key] = entry
    write_json(packages / INDEX, listing)
    return {"name": key, **entry}


def _reader(source):
    """read(relative path) -> bytes of a package source: a folder or an http(s) base URL."""
    source = str(source)
    if source.startswith(("https://", "http://")):
        from urllib.request import urlopen
        base = source.rstrip("/") + "/"

        def read(relative):
            with urlopen(base + relative, timeout=120) as response:
                return response.read()
        return read
    folder = Path(source)
    return lambda relative: (folder / relative).read_bytes()


def find(model, source, chat_template=None):
    """(name, entry) of the package for checkpoint `model` under the current suite in `source`, or None.
    chat_template: the --chat-template source the package was prepared with (prepare.model_identity)."""
    from .prepare import model_identity, suite_identity
    from .reference import REFERENCE_VERSION
    model_sha, suite_sha = model_identity(model, chat_template)["sha256"], suite_identity()
    from urllib.error import HTTPError, URLError
    try:
        listing = json.loads(_reader(source)(INDEX))
    except FileNotFoundError:
        return None
    except HTTPError as error:
        if error.code == 404:
            return None
        raise
    except (URLError, TimeoutError):   # offline: the reference is scored here instead
        return None
    key = name(model_sha, suite_sha)
    entry = listing.get("packages", {}).get(key)
    if entry is None or entry["model_sha256"] != model_sha or entry["suite_sha256"] != suite_sha \
            or entry.get("reference") != REFERENCE_VERSION:
        return None
    return key, entry


def fetch(model, data, source, log=print, chat_template=None):
    """Write the package for checkpoint `model` from `source` into data folder `data`: its prepared.json and reference
    scores, every file checked against packages.json first. Returns the package's name, or None when the source has
    none for the checkpoint (prepared with `chat_template`, a --chat-template source, when given). A data folder
    already prepared from another model or suite is refused."""
    from .prepare import PREPARED
    found = find(model, source, chat_template)
    if found is None:
        return None
    key, entry = found
    data = Path(data)
    if (data / PREPARED).is_file():
        present = load_json(data / PREPARED)
        if present["model"]["sha256"] != entry["model_sha256"] or present["suite_sha256"] != entry["suite_sha256"]:
            raise SystemExit(f"{data} is prepared from another model or suite than the package {key}: fetch into a "
                             "new data directory")
    read = _reader(source)
    with tempfile.TemporaryDirectory(dir=data.parent if data.parent.exists() else None) as staging:
        staging = Path(staging)
        for relative, sha in entry["files"].items():
            raw = read(f"{key}/{relative}")
            if _sha(raw) != sha:
                raise SystemExit(f"{source}: {key}/{relative} differs from the sha256 packages.json records")
            (staging / relative).parent.mkdir(parents=True, exist_ok=True)
            (staging / relative).write_bytes(raw)
        data.mkdir(parents=True, exist_ok=True)
        from .fixed import LEGACY_AS_QUANTIZED, STORE
        for store in (STORE, LEGACY_AS_QUANTIZED):
            shutil.rmtree(data / store, ignore_errors=True)
        for relative in entry["files"]:
            (data / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staging / relative), data / relative)
    # Which package the scores came from, for results (crit_eval.run.quality_checks.reference_origin): kept while index.json is its own.
    from .. import portable_path
    index = (data / "reference" / "index.json").read_bytes()
    write_json(data / PACKAGE, {"name": key, "source": source if str(source).startswith(("https://", "http://"))
                                else portable_path(source), "index_sha256": _sha(index)})
    log(f"fetched reference package {key} ({entry['bytes'] / 2**20:.1f} MiB) into {data}")
    return key


def source(given=None):
    """The package source: `given`, else settings.toml's eval_packages, else the checkout's reference-packages
    folder (LOCAL, kept out of git) when it holds a packages.json, else the published packages (PUBLISHED)."""
    from .. import EVAL_DIR, machine_settings
    local = EVAL_DIR.parent / LOCAL
    return (given or machine_settings().get(SOURCE_SETTING) or (str(local) if (local / INDEX).is_file() else None)
            or PUBLISHED)
