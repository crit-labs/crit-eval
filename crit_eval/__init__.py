"""crit eval: one benchmark and quality suite that every engine runs the same way."""
import json
import os
import re
from pathlib import Path

# The package holds its suite, protocol, probes and examples; a checkout of crit-eval is its parent.
EVAL_DIR = Path(__file__).resolve().parent
SUITE_DIR = EVAL_DIR / "suite"
CONTRACT = "eval-v1"
PROTOCOL_ID = re.compile(r"crit-eval-v(\d+)(?:\.(\d+))?")


def protocol_version(identifier):
    """crit-eval-v4 -> (4, 0) and crit-eval-v4.1 -> (4, 1). Scores compare within one major version: a change
    that can move a score bumps the major, anything else the minor (docs/README.md)."""
    match = PROTOCOL_ID.fullmatch(str(identifier))
    if not match:
        raise ValueError(f"not an Eval protocol id: {identifier}")
    return int(match[1]), int(match[2] or 0)


# The context a user wants engines built for, unless settings.toml's context_tokens says otherwise; the model's
# positions and the device's memory may allow less.
DEFAULT_CONTEXT_TOKENS = 262_144


def checkouts():
    """The checkouts Eval runs from: crit-eval's (when the package is not installed into site-packages) and the crit
    checkout of the crit_core it imports, which holds the machine's settings.toml."""
    import crit_core
    found = [EVAL_DIR.parent, Path(crit_core.__file__).resolve().parents[3]]
    return [root for i, root in enumerate(found) if root not in found[:i] and (root / ".git").exists()]


def machine_settings():
    """Memory thresholds from settings.toml (vram_headroom_gib, context_reserve_gib), the wanted context
    (context_tokens) and the reference package source (eval_packages, crit_eval.packages): crit-eval's checkout's,
    else the crit checkout's (checkouts); without one Eval uses the protocol's defaults and DEFAULT_CONTEXT_TOKENS."""
    path = next((root / "settings.toml" for root in checkouts() if (root / "settings.toml").is_file()), None)
    if path is None:
        return {}
    import tomllib
    value = tomllib.loads(path.read_text(encoding="utf-8"))
    found = {key: float(value[key]) for key in ("vram_headroom_gib", "context_reserve_gib")
             if isinstance(value.get(key), (int, float)) and not isinstance(value.get(key), bool)}
    if type(value.get("context_tokens")) is int and value["context_tokens"] > 0:
        found["context_tokens"] = value["context_tokens"]
    if isinstance(value.get("eval_packages"), str) and value["eval_packages"]:
        found["eval_packages"] = value["eval_packages"]
    return found


def requested_context():
    """The context the user wants: settings.toml's context_tokens, else DEFAULT_CONTEXT_TOKENS."""
    return machine_settings().get("context_tokens", DEFAULT_CONTEXT_TOKENS)


def portable_path(path):
    """A path as results store it, naming no user: relative to a checkout Eval runs from (posix) inside it, `~/...` inside
    the home directory, the bare file name where another user directory (/home/NAME, C:/Users/NAME, WSL's
    /mnt/c/Users/NAME) holds it, else the absolute path (toolkits under /usr/local or Program Files). None stays
    None."""
    if path is None:
        return None
    path = Path(os.path.abspath(path))
    for root, prefix in [(root, "") for root in checkouts()] + [(Path.home(), "~/")]:
        try:
            return prefix + path.relative_to(root).as_posix()
        except ValueError:
            pass
    if any(part.lower() in ("users", "home") for part in path.parts[:-2]):
        return path.name
    return path.as_posix()


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=1)
        handle.write("\n")
    os.replace(temporary, path)
