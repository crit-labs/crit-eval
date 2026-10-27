"""The engine under test: an adapter module loaded from its own folder, the engine it opens, and the identity of a
native engine's library."""
import hashlib
import importlib.util
import sys
import types
from pathlib import Path

from .. import CONTRACT, portable_path


_ADAPTER_FOLDERS = set()   # folders of the adapters loaded in this process


def load_adapter(path):
    """The adapter as a module of a package of its own folder (crit_eval_adapter_<sha>), so the relative imports of
    two adapters (`from .engine import Engine`, `crit eval ab` of two packages) each find their own folder's code.
    The folder also goes on sys.path for absolute imports, which Python keeps once per name: an adapter whose folder
    holds a top-level module that another adapter's folder already provided is refused, not run on the other's code."""
    path = Path(path).resolve()
    folder = path.parent
    for name, imported in list(sys.modules.items()):
        origin = getattr(imported, "__file__", None)
        if "." in name or not origin or not ((folder / name).is_dir() or (folder / f"{name}.py").is_file()):
            continue
        owner = next((other for other in _ADAPTER_FOLDERS
                      if other != folder and Path(origin).resolve().is_relative_to(other)), None)
        if owner:
            raise SystemExit(f"{path} and the adapter in {owner} both import a top-level {name!r}, which Python "
                             f"loads once: the second would run the first one's code. Import it relatively in each "
                             f"adapter (from .{name} import ...); each adapter is loaded as a package of its folder")
    namespace = "crit_eval_adapter_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    if namespace not in sys.modules:
        package = types.ModuleType(namespace)
        package.__path__ = [str(folder)]
        sys.modules[namespace] = package
    spec = importlib.util.spec_from_file_location(f"{namespace}.{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(folder))
    _ADAPTER_FOLDERS.add(folder)
    # A retained package is inventoried byte for byte: importing its adapter (and the modules it imports) must
    # not write __pycache__ into it.
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    if not hasattr(module, "create_eval_engine"):
        raise SystemExit(f"{path} does not export create_eval_engine(context); see docs/ADAPTER.md")
    return module


def open_engine(module, model, config, speculative, context_tokens=None, exact_prompts=False):
    """context_tokens: the context (prompt plus output) the engine is built for, when one is required. exact_prompts:
    an engine behind a server must render every frozen prompt to its prepared ids from the first request on, as the
    quality honesty check needs (crit_eval.run.chat), so a wrong template fails at once instead of after the speed parts."""
    context = {"contract": CONTRACT, "model": str(model), "config": dict(config), "speculative": speculative}
    if context_tokens:
        context["context_tokens"] = int(context_tokens)
    engine = module.create_eval_engine(context)
    if exact_prompts and hasattr(engine, "exact_prompts"):
        engine.exact_prompts = True
    info = engine.info()
    for key in ("name", "configuration", "max_context"):
        if key not in info:
            raise SystemExit(f"engine.info() is missing {key!r}")
    return engine, info


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def native_identity(info):
    """sha256 and size of each native library the adapter declares it loaded (`native_libraries`), each path as
    portable_path stores it."""
    out = []
    for path in info.get("native_libraries") or []:
        entry = {"path": portable_path(path)}
        try:
            entry.update(sha256=file_sha256(path), bytes=Path(path).stat().st_size)
        except OSError as error:
            entry["error"] = error.strerror or type(error).__name__   # str(error) would name the path
        out.append(entry)
    return out


def plain_on_engine(engine, info):
    """Whether the speculative engine itself decodes plain one-token greedy (info "plain", generate_plain),
    so exactness needs no second engine."""
    return bool(info.get("speculative") and info.get("plain") and callable(getattr(engine, "generate_plain", None)))
