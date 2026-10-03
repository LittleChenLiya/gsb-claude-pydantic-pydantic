"""Static selection interpreter for discriminated unions."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal
from pydantic import TypeAdapter

__all__ = ("StaticDiscriminatorResult", "static_discriminate")

Status = Literal["matched", "missing_tag", "unknown_tag", "unsupported_path", "unknown_discriminator"]

@dataclass
class StaticDiscriminatorResult:
    status: Status
    path: list[str | int] | None = None
    tag: Any = None
    branch: Any = None
    branch_name: str | None = None
    choices: dict[Any, Any] = field(default_factory=dict)
    discriminator: Any = None
    tried_paths: list[list[str | int]] = field(default_factory=list)

def _find_tagged_union(schema):
    if not isinstance(schema, dict): return None
    t = schema.get("type")
    if t == "tagged-union": return schema
    if t in ("nullable", "definitions", "missing-sentinel"):
        inner = schema.get("schema")
        if inner is not None:
            r = _find_tagged_union(inner)
            if r is not None: return r
    if t == "definitions":
        for ds in schema.get("definitions", []):
            r = _find_tagged_union(ds)
            if r is not None: return r
    return None

def _is_path_supported(path):
    return all(isinstance(e, (str, int)) and not isinstance(e, bool) for e in path)

def _resolve_path(value, path):
    """Resolve a path in a read-only structure.

    Returns (status, value) where status is one of:
    - "found": the path exists (value is the tag, may be None)
    - "missing": the path could exist but a key/index is absent
    - "unsupported": the path cannot be resolved because the input structure
      does not support key/index access at some level
    """
    current = value
    for key in path:
        if isinstance(current, dict):
            if key in current:
                current = current[key]
            else:
                return "missing", None
        elif isinstance(current, (list, tuple)) and isinstance(key, int) and not isinstance(key, bool):
            try:
                current = current[key]
            except IndexError:
                return "missing", None
        else:
            return "unsupported", None
    return "found", current

def _get_branch_name(choice):
    if not isinstance(choice, dict): return None
    t = choice.get("type")
    if t in ("model", "dataclass"):
        cls = choice.get("cls")
        if cls is not None: return getattr(cls, "__name__", str(cls))
    elif t == "typed-dict": return choice.get("name", "TypedDict")
    elif t == "definition-ref": return choice.get("schema_ref")
    elif t == "tagged-union": return "tagged-union"
    elif t == "lax-or-strict": return "lax-or-strict"
    return None

def static_discriminate(adapter, value):
    schema = _find_tagged_union(adapter.core_schema)
    if schema is None:
        return StaticDiscriminatorResult(status="unsupported_path", discriminator=None)
    discriminator = schema.get("discriminator")
    choices = schema.get("choices", {})
    if callable(discriminator):
        return StaticDiscriminatorResult(status="unknown_discriminator", choices=choices, discriminator=discriminator)
    if isinstance(discriminator, str):
        paths = [[discriminator]]
    elif isinstance(discriminator, list):
        if len(discriminator) == 0: paths = []
        elif isinstance(discriminator[0], list): paths = discriminator
        else: paths = [discriminator]
    else:
        return StaticDiscriminatorResult(status="unsupported_path", choices=choices, discriminator=discriminator)
    for path in paths:
        if not _is_path_supported(path):
            return StaticDiscriminatorResult(status="unsupported_path", choices=choices, discriminator=discriminator, tried_paths=paths)
    any_unsupported = False
    for path in paths:
        resolve_status, tag = _resolve_path(value, path)
        if resolve_status == "found":
            if tag in choices:
                branch = choices[tag]
                return StaticDiscriminatorResult(status="matched", path=path, tag=tag, branch=branch, branch_name=_get_branch_name(branch), choices=choices, discriminator=discriminator, tried_paths=paths)
            else:
                return StaticDiscriminatorResult(status="unknown_tag", path=path, tag=tag, choices=choices, discriminator=discriminator, tried_paths=paths)
        elif resolve_status == "unsupported":
            any_unsupported = True
    if any_unsupported:
        return StaticDiscriminatorResult(status="unsupported_path", path=paths[0] if paths else None, choices=choices, discriminator=discriminator, tried_paths=paths)
    return StaticDiscriminatorResult(status="missing_tag", path=paths[0] if paths else None, choices=choices, discriminator=discriminator, tried_paths=paths)
