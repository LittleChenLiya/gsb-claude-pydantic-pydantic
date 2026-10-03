"""Read-only serialization exposure auditor for Pydantic models.

A model's ``repr()`` output is not a reliable guide to what a model exports:
``repr=False`` only hides a field from the human-readable representation, it
never removes the field from ``model_dump()`` / ``model_dump_json()`` output.
This module provides a **static, read-only** auditor that answers the question
"which fields would be exported, under which keys, and with which sensitivity?"
*without* creating a model instance and *without* executing any user code.

The auditor works purely from:

* model field definitions (:attr:`BaseModel.model_fields` /
  :attr:`BaseModel.model_computed_fields`), and
* the compiled core schema (:attr:`BaseModel.__pydantic_core_schema__`).

It reports, per field:

* the **serialization alias** (the key actually used on export),
* **exclusion** (``exclude=True``),
* the **inheritance source** (which class in the MRO defined the field),
* **explicit sensitive types** (``SecretStr`` & friends, which pydantic
  protects by default at dump time),
* ``repr`` visibility - reported for information only; ``repr=False`` is
  *never* treated as a serialization exclusion.

Custom serializers (``@field_serializer``, ``PlainSerializer`` /
``WrapSerializer`` annotations) and ``@computed_field`` properties are
**opaque**: the auditor never calls them. They are marked
:data:`Sensitivity.UNKNOWN` unless an explicit
:class:`SerializerAuditAdapter` declares how to expand them. Only adapters
registered by the caller are consulted, and every consultation is counted in
the report's ``callback_counts``.

Recursive models are handled with a **per-path stack**: expansion stops at the
reference that would re-enter a model already on the current path, so
recursion terminates - but the same nested model reached via two different
paths is expanded on *both* paths (no global visited set).

Example:
    ```python
    from pydantic import BaseModel, SecretStr
    from pydantic.experimental.serialization_audit import audit_model

    class User(BaseModel):
        name: str
        password: SecretStr

    report = audit_model(User)
    print(report.render_text())
    ```
"""

from __future__ import annotations

import enum
import inspect
import types as _types
import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from dataclasses import field as _dc_field
from typing import Any

from pydantic import BaseModel
from pydantic._internal._repr import display_as_type
from pydantic.types import _SecretBase

__all__ = (
    'Sensitivity',
    'FieldAuditContext',
    'AdapterVerdict',
    'SerializerAuditAdapter',
    'FieldAuditEntry',
    'SerializationAuditReport',
    'audit_model',
)


class Sensitivity(str, enum.Enum):
    """How sensitive a field's serialized output is considered to be."""

    PLAIN = 'plain'
    """An ordinary exportable field with no custom serialization logic."""

    SECRET = 'secret'
    """A pydantic secret type (e.g. ``SecretStr``); pydantic masks it by default at dump time."""

    UNKNOWN = 'unknown'
    """Custom serializer or computed field with no audit adapter; output shape is not statically known."""

    ADAPTER_DECLARED = 'adapter-declared'
    """A registered audit adapter explicitly declared how this opaque field serializes."""


@dataclass(frozen=True)
class FieldAuditContext:
    """Context handed to a :class:`SerializerAuditAdapter` for an opaque field.

    The ``serializer`` callable is provided for *identification only*; the
    auditor guarantees it is never called.
    """

    model: type[BaseModel]
    """The model class that owns the field."""

    field_name: str
    """The field (or computed field) name."""

    kind: str
    """One of ``'field_serializer'``, ``'computed_field'`` or ``'model_serializer'``."""

    serializer: Callable[..., Any] | None
    """The user serializer callable, if any. Never invoked by the auditor."""

    annotation: Any
    """The declared (return) annotation of the field."""

    export_key: str
    """The key under which the field is exported (serialization alias aware)."""


@dataclass(frozen=True)
class AdapterVerdict:
    """An adapter's explicit, static declaration for an opaque field."""

    sensitivity: Sensitivity = Sensitivity.ADAPTER_DECLARED
    """Declared sensitivity of the serialized output."""

    annotation: Any = None
    """Declared output annotation. If it contains ``BaseModel`` subclasses they are expanded statically."""

    note: str = ''
    """Free-form note recorded in the report."""


class SerializerAuditAdapter(typing.Protocol):
    """Explicit safe-audit adapter for custom serializers / computed fields.

    Adapters are the *only* way opaque fields get expanded. An adapter must be
    a pure, side-effect-free declaration: it receives a
    :class:`FieldAuditContext` and returns an :class:`AdapterVerdict`, or
    ``None`` to decline (the field then stays :data:`Sensitivity.UNKNOWN`).
    """

    @property
    def name(self) -> str:  # pragma: no cover - protocol declaration
        """Stable identifier used in the report's callback counts."""
        ...

    def audit(self, context: FieldAuditContext) -> AdapterVerdict | None:  # pragma: no cover
        """Return a verdict for the opaque field, or ``None`` to decline."""
        ...


@dataclass
class FieldAuditEntry:
    """Static audit result for one exported (or excluded) field."""

    path: str
    """Dotted path from the audited root model, e.g. ``'address.city'``."""

    field_name: str
    """The attribute name on the model."""

    annotation: str
    """Display string of the declared annotation."""

    export_key: str
    """Key used on serialization (serialization alias, falling back to alias, then name)."""

    serialization_alias: str | None
    """The explicit serialization alias, if any."""

    excluded: bool
    """``True`` when the field is excluded from serialization (``exclude=True``)."""

    repr_visible: bool
    """Whether the field appears in ``repr()``. Informational only: ``repr=False``
    does **not** exclude a field from serialization."""

    defined_in: str
    """Qualified name of the class in the MRO that defined the field."""

    sensitivity: Sensitivity
    """Static sensitivity classification."""

    computed: bool = False
    """Whether this is a ``@computed_field``."""

    custom_serializer: bool = False
    """Whether a user-defined serializer governs this field's output."""

    recursive_ref: bool = False
    """Whether expansion stopped here because the model already appears on the current path."""

    adapter: str | None = None
    """Name of the audit adapter that declared this entry, if any."""

    note: str = ''
    """Extra explanation (adapter notes, recursion stop reason, ...)."""

    children: list[FieldAuditEntry] = _dc_field(default_factory=list)
    """Statically expanded nested-model fields."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary of this entry (metadata only, never values)."""
        return {
            'path': self.path,
            'field_name': self.field_name,
            'annotation': self.annotation,
            'export_key': self.export_key,
            'serialization_alias': self.serialization_alias,
            'excluded': self.excluded,
            'repr_visible': self.repr_visible,
            'defined_in': self.defined_in,
            'sensitivity': self.sensitivity.value,
            'computed': self.computed,
            'custom_serializer': self.custom_serializer,
            'recursive_ref': self.recursive_ref,
            'adapter': self.adapter,
            'note': self.note,
            'children': [child.to_dict() for child in self.children],
        }

    def _flags(self) -> list[str]:
        flags = [f'sensitivity={self.sensitivity.value}']
        if self.excluded:
            flags.append('excluded')
        if not self.repr_visible:
            flags.append('repr=False (still exported)')
        if self.computed:
            flags.append('computed')
        if self.custom_serializer:
            flags.append('custom-serializer')
        if self.recursive_ref:
            flags.append('recursive-ref')
        if self.adapter:
            flags.append(f'adapter={self.adapter}')
        return flags

    def _render(self, indent: int, lines: list[str]) -> None:
        pad = '  ' * indent
        alias_note = ''
        if self.serialization_alias is not None and self.serialization_alias != self.field_name:
            alias_note = f" (serialization alias of '{self.field_name}')"
        lines.append(
            f'{pad}- {self.path}: {self.annotation} -> export key {self.export_key!r}{alias_note} '
            f'[{", ".join(self._flags())}] (defined in {self.defined_in})'
        )
        if self.note:
            lines.append(f'{pad}    note: {self.note}')
        for child in self.children:
            child._render(indent + 1, lines)


@dataclass
class SerializationAuditReport:
    """Static serialization exposure report for one model.

    The report contains metadata only - field names, types, aliases, flags -
    and never any field values.
    """

    model: str
    """Qualified name of the audited model."""

    entries: list[FieldAuditEntry]
    """Top-level field entries (nested models appear as children)."""

    custom_model_serializer: bool
    """Whether the model defines a custom ``@model_serializer`` (whole-dump shape unknown)."""

    callback_counts: dict[str, int]
    """How many times each registered adapter was consulted."""

    adapter_matches: dict[str, int]
    """How many times each adapter returned a verdict."""

    notes: list[str] = _dc_field(default_factory=list)
    """Report-level notes and guarantees."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary of the whole report (metadata only)."""
        return {
            'model': self.model,
            'custom_model_serializer': self.custom_model_serializer,
            'callback_counts': dict(self.callback_counts),
            'adapter_matches': dict(self.adapter_matches),
            'notes': list(self.notes),
            'entries': [entry.to_dict() for entry in self.entries],
        }

    def render_text(self) -> str:
        """Render a static, human-readable report. Contains no field values."""
        lines = [
            f'Serialization audit report for {self.model}',
            'Guarantees: static read-only audit; no model instance created; '
            'no user serializer or computed field executed; no field values inspected.',
        ]
        if self.custom_model_serializer:
            lines.append(
                'WARNING: custom @model_serializer detected; the exported shape is unknown '
                'unless declared by an audit adapter.'
            )
        lines.append('fields:')
        for entry in self.entries:
            entry._render(1, lines)
        if self.callback_counts:
            counts = ', '.join(
                f'{name}: {count} consulted / {self.adapter_matches.get(name, 0)} matched'
                for name, count in self.callback_counts.items()
            )
            lines.append(f'adapter callbacks: {counts}')
        lines.extend(f'note: {note}' for note in self.notes)
        return '\n'.join(lines)


def _is_secret_class(obj: Any) -> bool:
    try:
        return inspect.isclass(obj) and issubclass(obj, _SecretBase)
    except TypeError:  # pragma: no cover - defensive
        return False


def _analyze_annotation(annotation: Any, models: list[type[BaseModel]], seen: set) -> bool:
    """Collect nested ``BaseModel`` subclasses and detect secret types in an annotation.

    ``models`` is deduplicated per-annotation (via ``seen``) so ``Union[X, X]``
    expands once - this is deliberately *not* a global visited set.
    """
    if annotation is None or annotation is type(None):
        return False
    if id(annotation) in seen:
        return False
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return _analyze_annotation(typing.get_args(annotation)[0], models, seen)
    if origin is typing.Union or origin is getattr(_types, 'UnionType', None):
        secret = False
        for arg in typing.get_args(annotation):
            secret = _analyze_annotation(arg, models, seen) or secret
        return secret
    if origin in (list, set, frozenset, tuple, dict):
        secret = False
        for arg in typing.get_args(annotation):
            secret = _analyze_annotation(arg, models, seen) or secret
        return secret
    if _is_secret_class(annotation):
        return True
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        seen.add(id(annotation))
        models.append(annotation)
        return False
    return False


def _find_custom_serializer(node: Any, visited: set) -> Callable[..., Any] | None:
    """Find a user-defined serialization function in a core-schema fragment.

    Pydantic's own serializers (e.g. the ``SecretStr`` masking function) are
    recognized by their module and are *not* treated as custom. Nothing is
    ever called.
    """
    node_id = id(node)
    if node_id in visited:
        return None
    visited.add(node_id)
    if isinstance(node, dict):
        serialization = node.get('serialization')
        if isinstance(serialization, dict) and serialization.get('type') in ('function-plain', 'function-wrap'):
            func = serialization.get('function')
            module = getattr(func, '__module__', '') or ''
            if not module.startswith('pydantic'):
                return func
        for value in node.values():
            found = _find_custom_serializer(value, visited)
            if found is not None:
                return found
    elif isinstance(node, (list, tuple)):
        for item in node:
            found = _find_custom_serializer(item, visited)
            if found is not None:
                return found
    return None


def _model_fields_schema(model: type[BaseModel]) -> dict[str, Any]:
    core_schema = model.__pydantic_core_schema__
    if core_schema.get('type') == 'definitions':
        # Recursive models wrap the model schema in a definitions block.
        core_schema = next(
            (d for d in core_schema.get('definitions', []) if isinstance(d, dict) and d.get('type') == 'model'),
            core_schema.get('schema', {}),
        )
    inner = core_schema.get('schema', {})
    if isinstance(inner, dict) and inner.get('type') == 'model-fields':
        return inner.get('fields', {})
    return {}


def _defined_in(model: type[BaseModel], name: str, *, computed: bool = False) -> str:
    """Return the qualified name of the class in the MRO that defined ``name``."""
    for klass in model.__mro__:
        if klass in (BaseModel, object):
            continue
        if name in (klass.__dict__.get('__annotations__') or {}):
            return f'{klass.__module__}.{klass.__qualname__}'
        if computed and name in klass.__dict__:
            return f'{klass.__module__}.{klass.__qualname__}'
    return f'{model.__module__}.{model.__qualname__}'


class _Auditor:
    def __init__(self, adapters: Sequence[SerializerAuditAdapter]):
        self.adapters = list(adapters)
        self.callback_counts: dict[str, int] = {}
        self.adapter_matches: dict[str, int] = {}

    def consult_adapters(self, context: FieldAuditContext) -> tuple[SerializerAuditAdapter, AdapterVerdict] | None:
        for adapter in self.adapters:
            self.callback_counts[adapter.name] = self.callback_counts.get(adapter.name, 0) + 1
            verdict = adapter.audit(context)
            if verdict is not None:
                self.adapter_matches[adapter.name] = self.adapter_matches.get(adapter.name, 0) + 1
                return adapter, verdict
        return None

    def audit_model_fields(
        self, model: type[BaseModel], path_prefix: str, path_stack: tuple[type[BaseModel], ...]
    ) -> list[FieldAuditEntry]:
        entries: list[FieldAuditEntry] = []
        fields_schema = _model_fields_schema(model)
        for name, field_info in model.model_fields.items():
            entries.append(self._audit_field(model, name, field_info, fields_schema.get(name), path_prefix, path_stack))
        for name, computed_info in model.model_computed_fields.items():
            entries.append(self._audit_computed_field(model, name, computed_info, path_prefix, path_stack))
        return entries

    def _audit_field(
        self,
        model: type[BaseModel],
        name: str,
        field_info: Any,
        field_schema: Any,
        path_prefix: str,
        path_stack: tuple[type[BaseModel], ...],
    ) -> FieldAuditEntry:
        path = f'{path_prefix}{name}'
        serialization_alias = field_info.serialization_alias
        export_key = serialization_alias or field_info.alias or name
        excluded = field_info.exclude is True
        annotation = field_info.annotation

        nested_models: list[type[BaseModel]] = []
        is_secret = _analyze_annotation(annotation, nested_models, set())

        serializer_func = _find_custom_serializer(field_schema, set()) if field_schema is not None else None
        custom_serializer = serializer_func is not None

        entry = FieldAuditEntry(
            path=path,
            field_name=name,
            annotation=display_as_type(annotation),
            export_key=export_key,
            serialization_alias=serialization_alias,
            excluded=excluded,
            repr_visible=field_info.repr,
            defined_in=_defined_in(model, name),
            sensitivity=Sensitivity.PLAIN,
            custom_serializer=custom_serializer,
        )

        if is_secret:
            entry.sensitivity = Sensitivity.SECRET
            entry.note = 'pydantic secret type; masked by default at dump time'
        if custom_serializer:
            entry.sensitivity = Sensitivity.UNKNOWN
            entry.note = 'custom serializer present; not executed, output shape unknown'
            self._apply_adapters(
                entry,
                FieldAuditContext(
                    model=model,
                    field_name=name,
                    kind='field_serializer',
                    serializer=serializer_func,
                    annotation=annotation,
                    export_key=export_key,
                ),
                path,
                path_stack,
            )
        for sub_model in nested_models:
            entry.children.extend(self._expand_nested(sub_model, path, path_stack))
        return entry

    def _audit_computed_field(
        self,
        model: type[BaseModel],
        name: str,
        computed_info: Any,
        path_prefix: str,
        path_stack: tuple[type[BaseModel], ...],
    ) -> FieldAuditEntry:
        path = f'{path_prefix}{name}'
        alias = computed_info.alias
        export_key = alias or name
        return_type = computed_info.return_type
        annotation = None if return_type is typing.Any or return_type is None else return_type

        entry = FieldAuditEntry(
            path=path,
            field_name=name,
            annotation=display_as_type(annotation) if annotation is not None else 'Any',
            export_key=export_key,
            serialization_alias=alias,
            excluded=False,
            repr_visible=computed_info.repr,
            defined_in=_defined_in(model, name, computed=True),
            sensitivity=Sensitivity.UNKNOWN,
            computed=True,
            note='computed field; not executed, output shape unknown',
        )
        self._apply_adapters(
            entry,
            FieldAuditContext(
                model=model,
                field_name=name,
                kind='computed_field',
                serializer=getattr(computed_info.wrapped_property, 'fget', None),
                annotation=annotation,
                export_key=export_key,
            ),
            path,
            path_stack,
        )
        return entry

    def _apply_adapters(
        self,
        entry: FieldAuditEntry,
        context: FieldAuditContext,
        path: str,
        path_stack: tuple[type[BaseModel], ...],
    ) -> None:
        result = self.consult_adapters(context)
        if result is None:
            return
        adapter, verdict = result
        entry.sensitivity = verdict.sensitivity
        entry.adapter = adapter.name
        entry.note = verdict.note or f'declared by audit adapter {adapter.name!r}'
        if verdict.annotation is not None:
            entry.annotation = display_as_type(verdict.annotation)
            nested_models: list[type[BaseModel]] = []
            _analyze_annotation(verdict.annotation, nested_models, set())
            for sub_model in nested_models:
                entry.children.extend(self._expand_nested(sub_model, path, path_stack))

    def _expand_nested(
        self, sub_model: type[BaseModel], path: str, path_stack: tuple[type[BaseModel], ...]
    ) -> list[FieldAuditEntry]:
        if sub_model in path_stack:
            return [
                FieldAuditEntry(
                    path=f'{path}.<{sub_model.__name__}>',
                    field_name=f'<{sub_model.__name__}>',
                    annotation=sub_model.__qualname__,
                    export_key='',
                    serialization_alias=None,
                    excluded=False,
                    repr_visible=True,
                    defined_in=f'{sub_model.__module__}.{sub_model.__qualname__}',
                    sensitivity=Sensitivity.PLAIN,
                    recursive_ref=True,
                    note=f'recursion stopped: {sub_model.__qualname__} already appears on this path',
                )
            ]
        return self.audit_model_fields(sub_model, f'{path}.', path_stack + (sub_model,))


def audit_model(model: type[BaseModel], *, adapters: Sequence[SerializerAuditAdapter] = ()) -> SerializationAuditReport:
    """Statically audit which fields ``model`` would export on serialization.

    Args:
        model: The ``BaseModel`` subclass to audit. No instance is created.
        adapters: Explicit safe-audit adapters used to expand custom
            serializers and computed fields. Without a matching adapter those
            fields are reported as :data:`Sensitivity.UNKNOWN`. Adapters are
            the only callbacks invoked, and every consultation is counted.

    Returns:
        A :class:`SerializationAuditReport` containing metadata only - never
        field values.

    The auditor never executes user code: field serializers, computed field
    properties and model serializers are inspected via the compiled schema,
    not called.
    """
    if not (inspect.isclass(model) and issubclass(model, BaseModel)):
        raise TypeError(f'audit_model() expects a BaseModel subclass, got {model!r}')

    auditor = _Auditor(adapters)
    entries = auditor.audit_model_fields(model, '', (model,))

    notes = [
        'repr=False only hides a field from repr(); it never excludes it from serialization.',
        'No user code was executed and no model instance was created during this audit.',
    ]

    custom_model_serializer = False
    for decorator in model.__pydantic_decorators__.model_serializers.values():
        module = getattr(decorator.func, '__module__', '') or ''
        if not module.startswith('pydantic'):
            custom_model_serializer = True
    if custom_model_serializer:
        context = FieldAuditContext(
            model=model,
            field_name='',
            kind='model_serializer',
            serializer=next(iter(model.__pydantic_decorators__.model_serializers.values())).func,
            annotation=None,
            export_key='',
        )
        result = auditor.consult_adapters(context)
        if result is not None:
            adapter, verdict = result
            notes.append(f'model serializer declared by audit adapter {adapter.name!r}: {verdict.note}')
        else:
            notes.append('custom @model_serializer present; exported shape is unknown without an audit adapter')

    return SerializationAuditReport(
        model=f'{model.__module__}.{model.__qualname__}',
        entries=entries,
        custom_model_serializer=custom_model_serializer,
        callback_counts=auditor.callback_counts,
        adapter_matches=auditor.adapter_matches,
        notes=notes,
    )
