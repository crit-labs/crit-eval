"""Tool-call validity on the agent speed cases (protocol speed.tool_calls): the calls a reply makes up to the model's
first end token, in the model's chat-template tool-call format, each valid, invalid or truncated. Reported, not judged."""
import json
import re

from ..limits.context import token_text

# The speed case kinds whose prompts define tools (suite/speed.json).
KINDS = ("tool-call", "agent")
OPEN, CLOSE = "<tool_call>", "</tool_call>"
FUNCTION = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.S)
PARAMETER = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)
JSON_TYPES = {"string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,), "array": (list,),
              "object": (dict,)}


def template_format(prompt_text):
    """The tool-call format the chat template renders and asks for, read from a prompt that defines tools: "json"
    (<tool_call> holding {"name": ..., "arguments": {...}}, Qwen3 and Hermes) or "xml" (<tool_call> holding
    <function=NAME> with a <parameter=KEY> block per argument, Qwen3.5 on and Qwen3-Coder); None for any other."""
    at = prompt_text.find(OPEN)
    if at < 0:
        return None
    rest = prompt_text[at + len(OPEN):].lstrip()
    return "xml" if rest.startswith("<function=") else "json" if rest.startswith("{") else None


def _typed(value, kind):
    return isinstance(value, JSON_TYPES.get(kind, (object,))) and not (kind in ("integer", "number")
                                                                        and isinstance(value, bool))


def _xml_value(text, kind):
    """A <parameter> block's text as its declared type, as a Qwen3-Coder parser reads it; ValueError when it is none."""
    if kind in (None, "string"):
        return text
    if kind == "boolean":
        if text.strip() not in ("true", "false"):
            raise ValueError(f"{text.strip()[:40]!r} is no boolean")
        return text.strip() == "true"
    value = json.loads(text)
    if not _typed(value, kind):
        raise ValueError(f"{text.strip()[:40]!r} is no {kind}")
    return value


def judge(name, arguments, tools):
    """Why a call is invalid (None when valid): it names a tool defined in tools, and its arguments are an object of
    that tool's declared properties, of their declared types, with every required one."""
    tool = tools.get(name)
    if tool is None:
        return f"names no defined tool ({name!r})"
    if not isinstance(arguments, dict):
        return "its arguments are no object"
    properties = (tool.get("parameters") or {}).get("properties") or {}
    unknown = [key for key in arguments if key not in properties]
    if unknown:
        return f"argument {unknown[0]!r} is not declared"
    wrong = [key for key, value in arguments.items() if not _typed(value, properties[key].get("type"))]
    if wrong:
        return f"argument {wrong[0]!r} is not of its declared type {properties[wrong[0]].get('type')}"
    missing = [key for key in (tool.get("parameters") or {}).get("required") or [] if key not in arguments]
    return f"required argument {missing[0]!r} is missing" if missing else None


def _parse(body, fmt, tools):
    """(name, reason or None) of one <tool_call> block's body."""
    if fmt == "json":
        try:
            call = json.loads(body)
        except ValueError:
            return None, "its JSON does not parse"
        if not isinstance(call, dict) or not isinstance(call.get("name"), str):
            return None, "its JSON names no tool"
        arguments = call.get("arguments", call.get("parameters", {}))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                return call["name"], "its arguments do not parse"
        return call["name"], judge(call["name"], arguments, tools)
    found = FUNCTION.fullmatch(body.strip())
    if not found:
        return None, "it holds no single <function=...> block"
    name = found[1].strip()
    properties = ((tools.get(name) or {}).get("parameters") or {}).get("properties") or {}
    arguments = {}
    for key, text in PARAMETER.findall(found[2]):
        try:
            arguments[key.strip()] = _xml_value(text, (properties.get(key.strip()) or {}).get("type"))
        except ValueError as error:
            return name, f"argument {key.strip()!r}: {error}"
    if PARAMETER.sub("", found[2]).strip():
        return name, "text outside its <parameter> blocks"
    return name, judge(name, arguments, tools)


def count(text, fmt, tools, finished, served=()):
    """The calls in `text` (the reply up to the model's first end token) and those a server parsed itself (served:
    [{"name", "arguments" (a JSON string)}]): valid, invalid, and truncated, a call the output-token limit cut
    (finished False: the reply reached no end token), which is not counted as invalid."""
    calls, at = [], 0
    while True:
        start = text.find(OPEN, at)
        if start < 0:
            break
        end = text.find(CLOSE, start)
        if end < 0:
            calls.append({"name": None, "state": "invalid" if finished else "truncated",
                          "reason": "no </tool_call> before the end token" if finished else "cut by the output limit"})
            break
        name, reason = _parse(text[start + len(OPEN):end], fmt, tools)
        calls.append({"name": name, "state": "invalid" if reason else "valid", **({"reason": reason} if reason else {})})
        at = end + len(CLOSE)
    for call in served:
        try:
            arguments = json.loads(call.get("arguments") or "{}")
        except ValueError:
            calls.append({"name": call.get("name"), "state": "truncated" if not finished else "invalid",
                          "reason": "its arguments do not parse"})
            continue
        reason = judge(call.get("name"), arguments, tools)
        calls.append({"name": call.get("name"), "state": "invalid" if reason else "valid",
                      **({"reason": reason} if reason else {})})
    return {"format": fmt, **{state: sum(c["state"] == state for c in calls) for state in ("valid", "invalid",
                                                                                           "truncated")},
            "calls": calls}


def case_tool_calls(case, tokens, sample, prepared):
    """protocol speed.tool_calls for one measured speed case of an agent kind: its output's calls (tokens, the case's
    first measured output; sample, the request that produced it, a served one carrying the calls its server parsed
    and its finish reason). None for other kinds; unmeasured without the prepared token strings or a format crit-eval
    parses."""
    if case["kind"] not in KINDS:
        return None
    strings = (prepared.get("retrieval") or {}).get("token_text")
    if not strings:
        return {"status": "unmeasured", "reason": "the prepared data has no token strings"}
    fmt = template_format(token_text(case["prompt_ids"], strings))
    if fmt is None:
        return {"status": "unmeasured",
                "reason": "the chat template's tool-call format is not one crit-eval parses (<tool_call> holding JSON "
                          "or a <function=...> block)"}
    stops = set(prepared.get("stop_ids") or [])
    served = (sample or {}).get("served")
    finished = (served.get("finish_reason") != "length" if served is not None
                else any(t in stops for t in tokens))
    tools = {t["function"]["name"]: t["function"] for t in case.get("tools") or [] if t.get("function")}
    return {"status": "measured", **count(token_text(tokens, strings, stops), fmt, tools, finished,
                                          (sample or {}).get("tool_calls") or ())}
