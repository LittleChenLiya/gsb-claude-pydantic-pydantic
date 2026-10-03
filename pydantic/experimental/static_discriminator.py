"""A static discriminator interpreter for tagged (discriminated) unions.

!!! warning "Experimental"
    This module is experimental and its API may change without notice.

Validating a discriminated union can fail with errors that are hard to
troubleshoot ("which branch was picked, and why?"). This module answers that
question *statically*: given a [`TypeAdapter`][pydantic.TypeAdapter] (or a raw
core schema) and an input value, it walks the already-compiled core schema,
reads the discriminator tag from the input, and reports which branch the tag
maps to — **without running validation**.

Nothing is executed while interpreting:

* the input is never validated, coerced or serialized;
* field/model validators and serializers are never called;
* a *callable* discriminator is never called (it cannot be evaluated
  statically, so the outcome is reported as `UNKNOWN`);
* object attributes are never read — only `Mapping` keys and `list`/`tuple`
  indices are consulted, so property getters and `__getattr__` hooks on
  arbitrary objects are never triggered.

## Supported scope

* `tagged-union` core schemas, as produced by
  `Annotated[Union[...], Field(discriminator='...')]` or
  [`Discriminator`][pydantic.Discriminator], possibly wrapped in
  `definitions` / `definition-ref` schemas (e.g. for recursive models).
* String discriminators (`discriminator='kind'`) and alias candidate paths
  (`discriminator=[['kind'], ['Kind']]`, produced when the discriminator
  field has an alias). Candidates are tried in schema order; the first path
  present in the input wins.
* Nested tag paths (e.g. `[['pet', 'pet_type']]`) and integer path segments
  for `list`/`tuple` inputs, as accepted by
  `core_schema.tagged_union_schema`.
* Explicit `None` tags: a key that is present with value `None` is a *found*
  tag (and `None` may itself be a valid choice) — presence is decided with
  `in`, never by truthiness.

Not supported (reported as `UNSUPPORTED`, never raised for ordinary input):

* schemas that are not tagged unions (e.g. a plain smart `Union`);
* tag paths that cannot be walked for the given input shape (e.g. a string
  segment on a non-mapping input, or an out-of-range structure type);
* callable discriminators are reported as `UNKNOWN` — executing them to
  guess a branch would violate the "static only" guarantee.

## Short demo

```python
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter
from pydantic.experimental.static_discriminator import StaticDiscriminatorInterpreter

class A(BaseModel):
    kind: Literal['a']
    x: int

class B(BaseModel):
    kind: Literal['b']
    y: str

adapter = TypeAdapter(Annotated[Union[A, B], Field(discriminator='kind')])
interpreter = StaticDiscriminatorInterpreter(adapter)

print(interpreter.inspect({'kind': 'a'}).explain())
#> read path ('kind',) -> tag 'a' -> branch #1 of 2 (model 'A')
print(interpreter.inspect({'x': 1}).explain())
#> no discriminator tag found: none of the candidate paths [('kind',)] is present in the input
print(interpreter.inspect({'kind': 'z'}).explain())
#> read path ('kind',) -> tag 'z' does not match any known tag ['a', 'b']
```

Note that `{'kind': 'a'}` matches branch `A` even though the required field
`x` is missing: the interpreter reports the *choice*, not a validation
result.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic_core import CoreSchema

if TYPE_CHECKING:
    from ..type_adapter import TypeAdapter

__all__ = (
    'DiscriminatorStatus',
    'DiscriminatorReport',
    'StaticDiscriminatorInterpreter',
    'UNREAD',
    'inspect_discriminator',
)


class _Unread:
    """Sentinel type for `DiscriminatorReport.tag` when no tag was read."""

    _instance: _Unread | None = None

    def __new__(cls) -> _Unread:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return 'UNREAD'

    def __bool__(self) -> bool:
        return False


UNREAD: Any = _Unread()
"""Sentinel for `DiscriminatorReport.tag` when no tag value was read from the input.

Distinct from `None`, which is a legitimate tag value.
"""


class DiscriminatorStatus(str, Enum):
    """The possible outcomes of a static discriminator inspection."""

    MATCHED = 'matched'
    """A tag was read and it maps to a known branch."""
    MISSING_TAG = 'missing_tag'
    """Every candidate tag path was walkable, but none is present in the input."""
    UNKNOWN_TAG = 'unknown_tag'
    """A tag was read, but it does not match any of the known choices."""
    UNSUPPORTED = 'unsupported'
    """The schema or the tag path cannot be interpreted statically for this input."""
    UNKNOWN = 'unknown'
    """The discriminator is a callable; it was not executed, so the branch is unknown."""


@dataclass
class DiscriminatorReport:
    """The result of statically interpreting a discriminated union choice."""

    status: DiscriminatorStatus
    """The outcome of the inspection."""
    reason: str
    """A human-readable explanation of the outcome."""
    paths: list[tuple[str | int, ...]] = field(default_factory=list)
    """All candidate tag paths, in schema (alias) order."""
    path: tuple[str | int, ...] | None = None
    """The path that was actually read to obtain the tag, if any."""
    tag: Any = UNREAD
    """The raw tag value read from the input, or `UNREAD` if no tag was read.

    Note that an explicit `None` in the input is a *read* tag, not a missing one.
    """
    choices: list[Any] = field(default_factory=list)
    """The known tags, in schema order."""
    choice: CoreSchema | None = None
    """The core schema of the matched branch, if `status` is `MATCHED`."""
    choice_index: int | None = None
    """The 0-based index of the matched branch within `choices`, if matched."""

    @property
    def matched(self) -> bool:
        """Whether a branch was selected."""
        return self.status is DiscriminatorStatus.MATCHED

    @property
    def choice_label(self) -> str | None:
        """A best-effort human-readable label for the matched branch (e.g. the model class name)."""
        if self.choice is None:
            return None
        return _choice_label(self.choice)

    def explain(self) -> str:
        """A short, one-line explanation of how the outcome was reached."""
        return self.reason


def _choice_label(choice: CoreSchema) -> str:
    schema_type = choice.get('type', '?')
    if schema_type == 'model':
        cls = choice.get('cls')
        if cls is not None:
            return f'model {cls.__name__!r}'
    if schema_type == 'definition-ref':
        return f'ref {choice.get("schema_ref")!r}'
    inner = choice.get('schema')
    if isinstance(inner, dict):
        inner_type = inner.get('type')
        if inner_type == 'model-fields':
            return f'model {inner.get("model_name")!r}'
    return f'schema of type {schema_type!r}'


class StaticDiscriminatorInterpreter:
    """Statically interprets the discriminator of a tagged union core schema.

    Args:
        adapter_or_schema: A [`TypeAdapter`][pydantic.TypeAdapter] whose compiled core schema
            contains a tagged union, or a raw core schema dictionary.
    """

    def __init__(self, adapter_or_schema: TypeAdapter[Any] | CoreSchema) -> None:
        schema: CoreSchema
        if isinstance(adapter_or_schema, dict):
            schema = adapter_or_schema
        elif hasattr(adapter_or_schema, 'core_schema'):
            schema = adapter_or_schema.core_schema
        else:
            raise TypeError(f'expected a TypeAdapter or a core schema dict, got {type(adapter_or_schema).__name__!r}')
        self._definitions: dict[str, CoreSchema] = {}
        self._schema = self._resolve(schema)

    def _resolve(self, schema: CoreSchema) -> CoreSchema:
        """Unwrap `definitions`/`definition-ref` wrappers, collecting definition tables."""
        seen: set[str] = set()
        while True:
            schema_type = schema.get('type')
            if schema_type == 'definitions':
                for definition in schema.get('definitions', []):
                    ref = definition.get('ref')
                    if ref is not None:
                        self._definitions[ref] = definition
                schema = schema['schema']
            elif schema_type == 'definition-ref':
                ref = schema['schema_ref']
                if ref in seen:
                    break
                seen.add(ref)
                target = self._definitions.get(ref)
                if target is None:
                    break
                schema = target
            else:
                break
        return schema

    def inspect(self, data: Any) -> DiscriminatorReport:
        """Statically determine which branch of the tagged union `data` would select.

        This never validates `data`, never calls validators or callable
        discriminators, and never reads object attributes.
        """
        schema = self._schema
        if schema.get('type') != 'tagged-union':
            return DiscriminatorReport(
                status=DiscriminatorStatus.UNSUPPORTED,
                reason=(
                    f'schema of type {schema.get("type")!r} is not a tagged union; '
                    'only discriminated unions can be interpreted statically'
                ),
            )

        discriminator = schema.get('discriminator')
        tags, choice_schemas = _normalize_choices(schema.get('choices', {}))

        if callable(discriminator):
            name = getattr(discriminator, '__name__', repr(discriminator))
            return DiscriminatorReport(
                status=DiscriminatorStatus.UNKNOWN,
                reason=(
                    f'discriminator is the callable {name!r}; it was not executed, so the matching branch is unknown'
                ),
                choices=tags,
            )

        paths = _normalize_paths(discriminator)
        if paths is None:
            return DiscriminatorReport(
                status=DiscriminatorStatus.UNSUPPORTED,
                reason=f'discriminator {discriminator!r} is neither a string nor a list of paths',
                choices=tags,
            )

        saw_unsupported = False
        unsupported_notes: list[str] = []
        for path in paths:
            outcome, value, note = _walk_path(data, path)
            if outcome == 'unsupported':
                saw_unsupported = True
                unsupported_notes.append(note)
                continue
            if outcome == 'missing':
                continue
            # outcome == 'found': the first present path wins (schema order)
            return self._match_tag(path, value, tags, choice_schemas)

        if saw_unsupported:
            return DiscriminatorReport(
                status=DiscriminatorStatus.UNSUPPORTED,
                reason='; '.join(unsupported_notes),
                paths=paths,
                choices=tags,
            )
        return DiscriminatorReport(
            status=DiscriminatorStatus.MISSING_TAG,
            reason=f'no discriminator tag found: none of the candidate paths {paths!r} is present in the input',
            paths=paths,
            choices=tags,
        )

    def _match_tag(
        self,
        path: tuple[str | int, ...],
        tag: Any,
        tags: list[Any],
        choice_schemas: list[CoreSchema],
    ) -> DiscriminatorReport:
        try:
            index = _index_of_tag(tags, tag)
        except TypeError:
            index = None
        if index is None:
            return DiscriminatorReport(
                status=DiscriminatorStatus.UNKNOWN_TAG,
                reason=f'read path {path!r} -> tag {tag!r} does not match any known tag {tags!r}',
                paths=[path],
                path=path,
                tag=tag,
                choices=tags,
            )
        choice = self._resolve(choice_schemas[index])
        label = _choice_label(choice)
        return DiscriminatorReport(
            status=DiscriminatorStatus.MATCHED,
            reason=f'read path {path!r} -> tag {tag!r} -> branch #{index + 1} of {len(tags)} ({label})',
            paths=[path],
            path=path,
            tag=tag,
            choices=tags,
            choice=choice,
            choice_index=index,
        )


def _index_of_tag(tags: list[Any], tag: Any) -> int | None:
    """Find `tag` among the known `tags`, mirroring dict lookup semantics.

    Raises `TypeError` if `tag` is unhashable (tags in a tagged union are always
    hashable, so an unhashable tag can never match).
    """
    hash(tag)  # raises TypeError for unhashable tags, e.g. a dict or a list
    for i, known in enumerate(tags):
        if known is tag or (type(known) is type(tag) and known == tag):
            return i
    return None


def _normalize_choices(raw_choices: Any) -> tuple[list[Any], list[CoreSchema]]:
    """Normalize the `choices` of a tagged union to parallel lists of tags and schemas."""
    if isinstance(raw_choices, Mapping):
        items: Iterable[tuple[Any, CoreSchema]] = raw_choices.items()
    else:
        # list of (tag, schema) pairs; a tag of `None` marks an untagged choice and is skipped by core
        items = [(tag, choice) for tag, choice in raw_choices if tag is not None]
    tags: list[Any] = []
    choice_schemas: list[CoreSchema] = []
    for tag, choice in items:
        tags.append(tag)
        choice_schemas.append(choice)
    return tags, choice_schemas


def _normalize_paths(discriminator: Any) -> list[tuple[str | int, ...]] | None:
    """Normalize a static discriminator to a list of candidate paths, in schema order."""
    if isinstance(discriminator, str):
        return [(discriminator,)]
    if isinstance(discriminator, list):
        paths: list[tuple[str | int, ...]] = []
        for path in discriminator:
            if not isinstance(path, (list, tuple)) or not path:
                return None
            if not all(isinstance(seg, (str, int)) and not isinstance(seg, bool) for seg in path):
                return None
            paths.append(tuple(path))
        return paths or None
    return None


def _walk_path(data: Any, path: tuple[str | int, ...]) -> tuple[str, Any, str]:
    """Walk `path` into `data` using only mapping keys and sequence indices.

    Returns `(outcome, value, note)` where `outcome` is one of `'found'`, `'missing'`
    or `'unsupported'`. Never uses `getattr`, so object attribute getters are untouched.
    """
    current = data
    for segment in path:
        if isinstance(segment, str):
            if isinstance(current, Mapping):
                if segment in current:
                    current = current[segment]
                    continue
                return 'missing', UNREAD, ''
            return (
                'unsupported',
                UNREAD,
                f'cannot read path {path!r}: segment {segment!r} requires a mapping, '
                f'got {type(current).__name__!r} (object attributes are never read)',
            )
        # integer segment: sequence index (or an integer mapping key)
        if isinstance(current, (list, tuple)):
            if 0 <= segment < len(current):
                current = current[segment]
                continue
            return 'missing', UNREAD, ''
        if isinstance(current, Mapping):
            if segment in current:
                current = current[segment]
                continue
            return 'missing', UNREAD, ''
        return (
            'unsupported',
            UNREAD,
            f'cannot read path {path!r}: segment {segment!r} requires a list/tuple or mapping, '
            f'got {type(current).__name__!r}',
        )
    return 'found', current, ''


def inspect_discriminator(adapter_or_schema: TypeAdapter[Any] | CoreSchema, data: Any) -> DiscriminatorReport:
    """Convenience one-shot wrapper around `StaticDiscriminatorInterpreter`."""
    return StaticDiscriminatorInterpreter(adapter_or_schema).inspect(data)
