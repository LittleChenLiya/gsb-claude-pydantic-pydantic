"""Read-only serialization exposure auditor for Pydantic models.

This module provides a read-only auditor that determines which fields of a
Pydantic model will be exposed during serialization, based on the model's
fields and compiled core schema -- *without* dumping real instances.

The auditor reports:

- **Serialization alias** -- the effective alias used when serializing.
- **Field exclusion** -- whether a field is excluded from serialization
  (``exclude=True``). ``repr=False`` is *not* treated as exclusion.
- **Inheritance source** -- which class in the MRO defined each field.
- **Sensitive types** -- fields annotated with ``SecretStr`` / ``SecretBytes``
  are flagged as default-protected.
- **Custom serializers** -- fields with a custom serializer (``PlainSerializer``,
  ``WrapSerializer``, ``@field_serializer``) are marked ``unknown`` unless an
  explicit safe audit adapter expands them.
- **Computed fields** -- marked ``unknown`` unless an explicit safe audit
  adapter expands them.

Recursive models are handled via a *path stack*: when a ``definition-ref``
points to a model that is already on the current path, the auditor stops and
marks the reference as recursive. Two different paths to the same model are
both preserved (no global visited set).

No user code is executed during the audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias

from pydantic_core import core_schema

if TYPE_CHECKING:
    from pydantic import BaseModel

__all__ = [
    'AuditAdapter',
    'FieldAudit',
    'ModelAuditReport',
    'SensitiveKind',
    'SerializationAuditor',
    'SerializerKind',
    'audit_model',
]

SensitiveKind: TypeAlias = Literal['secret-str', 'secret-bytes']
SerializerKind: TypeAlias = Literal['none', 'default-protected', 'unknown', 'expanded']


@dataclass(frozen=True, slots=True)
class FieldAudit:
    """Audit result for a single model field.

    Attributes:
        name: The field name.
        path: The full path from the root model (tuple of field names).
        serialization_alias: The effective alias used for serialization.
        excluded: Whether the field is excluded from serialization
            (``exclude=True``). ``repr=False`` does *not* set this.
        repr: Whether the field appears in ``repr()``. Reported for reference;
            *not* used to determine serialization exclusion.
        source: The name of the class that defined this field (inheritance
            source).
        sensitive: The sensitive kind, or ``None`` if not sensitive.
        serializer: The serializer kind.
        computed: Whether this is a computed field.
        recursive: Whether this field is a recursive reference (the auditor
            stopped here to avoid infinite expansion).
        recursion_ref: The ``schema_ref`` that caused the recursion stop, if
            any.
        children: Nested fields (for nested models, lists, dicts, etc.).
    """

    name: str
    path: tuple[str, ...]
    serialization_alias: str
    excluded: bool
    repr: bool
    source: str
    sensitive: SensitiveKind | None
    serializer: SerializerKind
    computed: bool
    recursive: bool
    recursion_ref: str | None
    children: tuple[FieldAudit, ...]


@dataclass(frozen=True, slots=True)
class ModelAuditReport:
    """Audit result for a model.

    Attributes:
        model_name: The name of the audited model.
        model_cls: The fully-qualified class name.
        fields: The top-level fields of the model.
        adapter_calls: The number of times the audit adapter was called.
        recursive_refs: The schema refs that were stopped due to recursion.
    """

    model_name: str
    model_cls: str
    fields: tuple[FieldAudit, ...]
    adapter_calls: int
    recursive_refs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a plain dictionary (JSON-serializable)."""
        return {
            'model_name': self.model_name,
            'model_cls': self.model_cls,
            'adapter_calls': self.adapter_calls,
            'recursive_refs': list(self.recursive_refs),
            'fields': [_field_to_dict(f) for f in self.fields],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Return the report as a JSON string."""
        import json

        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_tree(self) -> str:
        """Return the report as a human-readable tree string."""
        lines: list[str] = []
        lines.append(f'{self.model_name} ({self.model_cls})')
        lines.append(f'  adapter_calls: {self.adapter_calls}')
        if self.recursive_refs:
            lines.append(f'  recursive_refs: {list(self.recursive_refs)}')
        for f in self.fields:
            _field_to_lines(f, lines, indent=1)
        return '\n'.join(lines)


def _field_to_dict(f: FieldAudit) -> dict[str, Any]:
    return {
        'name': f.name,
        'path': list(f.path),
        'serialization_alias': f.serialization_alias,
        'excluded': f.excluded,
        'repr': f.repr,
        'source': f.source,
        'sensitive': f.sensitive,
        'serializer': f.serializer,
        'computed': f.computed,
        'recursive': f.recursive,
        'recursion_ref': f.recursion_ref,
        'children': [_field_to_dict(c) for c in f.children],
    }


def _field_to_lines(f: FieldAudit, lines: list[str], *, indent: int) -> None:
    prefix = '  ' * indent
    flags: list[str] = []
    if f.excluded:
        flags.append('EXCLUDED')
    if not f.repr:
        flags.append('repr=False')
    if f.sensitive:
        flags.append(f'sensitive={f.sensitive}')
    if f.serializer != 'none':
        flags.append(f'serializer={f.serializer}')
    if f.computed:
        flags.append('computed')
    if f.recursive:
        flags.append(f'recursive -> {f.recursion_ref}')
    flag_str = f'  [{", ".join(flags)}]' if flags else ''
    lines.append(
        f'{prefix}- {f.name} (alias={f.serialization_alias}, source={f.source}){flag_str}'
    )
    for c in f.children:
        _field_to_lines(c, lines, indent=indent + 1)


class AuditAdapter(Protocol):
    """Adapter for expanding custom serializers and computed fields.

    The adapter is called for each custom serializer and computed field
    encountered during the audit. It returns a core schema to expand, or
    ``None`` to mark the field as ``unknown``.

    The adapter is *explicit*: it must actively opt in to expanding a
    serializer or computed field by returning a schema. Without an adapter,
    all custom serializers and computed fields are marked ``unknown``.

    The adapter is *safe*: it declares the serialized output schema without
    executing user code.
    """

    def expand_serializer(
        self,
        schema: core_schema.CoreSchema,
        *,
        field_name: str,
        path: tuple[str, ...],
    ) -> core_schema.CoreSchema | None:
        """Return a core schema to expand the serializer, or ``None`` to mark unknown.

        The returned schema describes the structure of the serialized output.
        The auditor walks it (with path-stack recursion protection) to report
        nested fields.
        """
        ...

    def expand_computed_field(
        self,
        property_name: str,
        *,
        path: tuple[str, ...],
    ) -> core_schema.CoreSchema | None:
        """Return a core schema to expand the computed field, or ``None`` to mark unknown.

        The returned schema describes the structure of the computed field's
        output. The auditor walks it (with path-stack recursion protection) to
        report nested fields.
        """
        ...


_SENSITIVE_TYPE_NAMES: frozenset[str] = frozenset({'SecretStr', 'SecretBytes'})

_SENSITIVE_KIND_MAP: dict[str, SensitiveKind] = {
    'SecretStr': 'secret-str',
    'SecretBytes': 'secret-bytes',
}


def _detect_sensitive(model_cls: type[BaseModel], field_name: str) -> SensitiveKind | None:
    """Detect whether a field is a sensitive type (SecretStr / SecretBytes).

    Uses the model's FieldInfo annotation (not repr, not instance dumping).
    """
    field_info = model_cls.model_fields.get(field_name)
    if field_info is None:
        return None
    annotation = field_info.annotation
    if annotation is None:
        return None
    origin = _unwrap_annotation(annotation)
    if origin is None:
        return None
    name = getattr(origin, '__name__', None)
    if name is None:
        return None
    return _SENSITIVE_KIND_MAP.get(name)


def _unwrap_annotation(annotation: Any) -> Any:
    """Unwrap Annotated / Optional / Union to find the underlying type."""
    from typing import Union, get_args, get_origin

    origin = get_origin(annotation)
    if origin is None:
        return annotation
    # Annotated[X, ...] -> X
    if hasattr(origin, '__mro__') and 'Annotated' in str(origin):
        args = get_args(annotation)
        if args:
            return _unwrap_annotation(args[0])
    # Union[X, None] -> X (Optional)
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _unwrap_annotation(args[0])
    return annotation


def _find_source(model_cls: type[BaseModel], field_name: str, *, computed: bool = False) -> str:
    """Find the class in the MRO that defined a field (inheritance source).

    Walks the MRO in reverse (base-most first) and returns the first class
    that has the field. ``BaseModel`` itself is skipped (it has no fields).
    """
    attr = '__pydantic_computed_fields__' if computed else '__pydantic_fields__'
    for base in reversed(model_cls.__mro__):
        if base is object:
            continue
        fields = getattr(base, attr, None)
        if fields and field_name in fields:
            return base.__name__
    return model_cls.__name__


_SCHEMA_WITH_INNER: frozenset[str] = frozenset(
    {
        'nullable',
        'function-after',
        'function-before',
        'function-wrap',
        'default',
        'lax-or-strict',
        'json-or-python',
        'list',
        'set',
        'frozenset',
        'deque',
        'dict',
        'tuple',
        'chain',
    }
)


class SerializationAuditor:
    """Read-only serialization exposure auditor for a Pydantic model.

    Based on the model's fields and compiled core schema. Does *not* dump
    real instances or execute user code.

    Args:
        model: The Pydantic model class to audit.
        adapter: An optional audit adapter for expanding custom serializers
            and computed fields. If ``None``, all custom serializers and
            computed fields are marked ``unknown``.
    """

    def __init__(self, model: type[BaseModel], *, adapter: AuditAdapter | None = None) -> None:
        self.model = model
        self.adapter = adapter
        self._definitions: dict[str, dict[str, Any]] = {}
        self._adapter_calls = 0
        self._recursive_refs: list[str] = []

    def audit(self) -> ModelAuditReport:
        """Run the audit and return the report."""
        schema = self.model.__pydantic_core_schema__

        self._build_definitions_map(schema)

        fields = self._walk_root(schema)

        return ModelAuditReport(
            model_name=self.model.__name__,
            model_cls=f'{self.model.__module__}.{self.model.__qualname__}',
            fields=tuple(fields),
            adapter_calls=self._adapter_calls,
            recursive_refs=tuple(self._recursive_refs),
        )

    def _build_definitions_map(self, schema: core_schema.CoreSchema) -> None:
        """Build a map from ``ref`` to model schema from the compiled schema."""
        if schema['type'] == 'definitions':
            for definition in schema.get('definitions', []):
                ref = definition.get('ref')
                if ref is not None:
                    self._definitions[ref] = definition
        elif schema['type'] == 'model':
            ref = schema.get('ref')
            if ref is not None:
                self._definitions[ref] = schema

    def _walk_root(self, schema: core_schema.CoreSchema) -> list[FieldAudit]:
        """Walk the root model schema."""
        if schema['type'] == 'definitions':
            root_ref = schema['schema'].get('schema_ref')
            if root_ref is not None and root_ref in self._definitions:
                return self._walk_model(root_ref, path=(), ref_stack=())
            return self._walk_schema(schema['schema'], path=(), ref_stack=())
        if schema['type'] == 'model':
            ref = schema.get('ref')
            if ref is not None:
                return self._walk_model(ref, path=(), ref_stack=())
            return self._walk_model_schema(schema, path=(), ref_stack=())
        return self._walk_schema(schema, path=(), ref_stack=())

    def _walk_model(
        self, ref: str, *, path: tuple[str, ...], ref_stack: tuple[str, ...]
    ) -> list[FieldAudit]:
        """Walk a model definition by ref."""
        model_schema = self._definitions.get(ref)
        if model_schema is None:
            return []
        return self._walk_model_schema(model_schema, path=path, ref_stack=ref_stack)

    def _walk_model_schema(
        self,
        model_schema: dict[str, Any],
        *,
        path: tuple[str, ...],
        ref_stack: tuple[str, ...],
    ) -> list[FieldAudit]:
        """Walk a model schema (type ``model``)."""
        inner = model_schema.get('schema', {})
        if inner.get('type') != 'model-fields':
            return []
        fields = inner.get('fields', {})
        computed = inner.get('computed_fields', [])

        result: list[FieldAudit] = []
        result.extend(
            self._walk_field(name, field_schema, path=path, ref_stack=ref_stack)
            for name, field_schema in fields.items()
        )
        result.extend(
            self._walk_computed_field(cf, path=path, ref_stack=ref_stack) for cf in computed
        )
        return result

    def _walk_field(
        self,
        name: str,
        field_schema: dict[str, Any],
        *,
        path: tuple[str, ...],
        ref_stack: tuple[str, ...],
    ) -> FieldAudit:
        """Walk a model-field schema."""
        field_path = path + (name,)
        schema = field_schema.get('schema', {})
        serialization_alias = field_schema.get('serialization_alias', name)
        excluded = bool(field_schema.get('serialization_exclude', False))

        field_info = self.model.model_fields.get(name)
        repr_flag = field_info.repr if field_info is not None else True

        sensitive = _detect_sensitive(self.model, name)

        has_custom_serializer = 'serialization' in schema
        if has_custom_serializer:
            serializer: SerializerKind = 'unknown'
            children: tuple[FieldAudit, ...] = ()
            if self.adapter is not None:
                self._adapter_calls += 1
                expanded = self.adapter.expand_serializer(
                    schema.get('serialization'), field_name=name, path=field_path
                )
                if expanded is not None:
                    serializer = 'expanded'
                    children = tuple(
                        self._walk_schema(expanded, path=field_path, ref_stack=ref_stack)
                    )
        elif sensitive is not None:
            serializer = 'default-protected'
            children = ()
        else:
            serializer = 'none'
            children = tuple(self._walk_schema(schema, path=field_path, ref_stack=ref_stack))

        source = _find_source(self.model, name)

        return FieldAudit(
            name=name,
            path=field_path,
            serialization_alias=serialization_alias,
            excluded=excluded,
            repr=repr_flag,
            source=source,
            sensitive=sensitive,
            serializer=serializer,
            computed=False,
            recursive=False,
            recursion_ref=None,
            children=children,
        )

    def _walk_computed_field(
        self,
        cf: dict[str, Any],
        *,
        path: tuple[str, ...],
        ref_stack: tuple[str, ...],
    ) -> FieldAudit:
        """Walk a computed-field schema."""
        property_name = cf.get('property_name', '')
        field_path = path + (property_name,)

        cfi = self.model.model_computed_fields.get(property_name)
        repr_flag = cfi.repr if cfi is not None else True
        alias = cfi.alias if cfi is not None and cfi.alias is not None else property_name

        serializer: SerializerKind = 'unknown'
        children: tuple[FieldAudit, ...] = ()
        if self.adapter is not None:
            self._adapter_calls += 1
            expanded = self.adapter.expand_computed_field(property_name, path=field_path)
            if expanded is not None:
                serializer = 'expanded'
                children = tuple(
                    self._walk_schema(expanded, path=field_path, ref_stack=ref_stack)
                )

        source = _find_source(self.model, property_name, computed=True)

        return FieldAudit(
            name=property_name,
            path=field_path,
            serialization_alias=alias,
            excluded=False,
            repr=repr_flag,
            source=source,
            sensitive=None,
            serializer=serializer,
            computed=True,
            recursive=False,
            recursion_ref=None,
            children=children,
        )

    def _walk_schema(
        self,
        schema: dict[str, Any],
        *,
        path: tuple[str, ...],
        ref_stack: tuple[str, ...],
    ) -> list[FieldAudit]:
        """Walk a core schema, collecting nested model fields.

        Handles recursion via the path stack: when a ``definition-ref``
        points to a ref already on the stack, the auditor stops and marks
        the reference as recursive.
        """
        if not isinstance(schema, dict):
            return []

        schema_type = schema.get('type')

        if schema_type == 'definition-ref':
            ref = schema.get('schema_ref')
            if ref is None:
                return []
            if ref in ref_stack:
                self._recursive_refs.append(ref)
                # Extract a clean model name from the ref (module.Name:id).
                ref_name = ref.split(':')[0] if ':' in ref else ref
                short_name = ref_name.rsplit('.', 1)[-1]
                return [
                    FieldAudit(
                        name=short_name,
                        path=path,
                        serialization_alias=short_name,
                        excluded=False,
                        repr=True,
                        source='recursive',
                        sensitive=None,
                        serializer='none',
                        computed=False,
                        recursive=True,
                        recursion_ref=ref,
                        children=(),
                    )
                ]
            new_stack = ref_stack + (ref,)
            return self._walk_model(ref, path=path, ref_stack=new_stack)

        if schema_type == 'model':
            ref = schema.get('ref')
            if ref is not None:
                new_stack = ref_stack + (ref,)
                return self._walk_model_schema(schema, path=path, ref_stack=new_stack)
            return self._walk_model_schema(schema, path=path, ref_stack=ref_stack)

        if schema_type in _SCHEMA_WITH_INNER:
            inner = schema.get('schema')
            if inner is not None:
                return self._walk_schema(inner, path=path, ref_stack=ref_stack)
            items = schema.get('items_schema')
            if items is not None:
                return self._walk_schema(items, path=path, ref_stack=ref_stack)
            values = schema.get('values_schema')
            if values is not None:
                result: list[FieldAudit] = []
                keys = schema.get('keys_schema')
                if keys is not None:
                    result.extend(self._walk_schema(keys, path=path, ref_stack=ref_stack))
                result.extend(self._walk_schema(values, path=path, ref_stack=ref_stack))
                return result
            items_list = schema.get('items_schema')
            if isinstance(items_list, list):
                result = []
                for item in items_list:
                    result.extend(self._walk_schema(item, path=path, ref_stack=ref_stack))
                return result
            return []

        if schema_type in ('union', 'tagged-union'):
            choices = schema.get('choices', [])
            result = []
            for choice in choices:
                result.extend(self._walk_schema(choice, path=path, ref_stack=ref_stack))
            return result

        if schema_type == 'chain':
            steps = schema.get('steps', [])
            result = []
            for step in steps:
                result.extend(self._walk_schema(step, path=path, ref_stack=ref_stack))
            return result

        if schema_type == 'json-or-python':
            result = []
            json_schema = schema.get('json_schema')
            if json_schema is not None:
                result.extend(self._walk_schema(json_schema, path=path, ref_stack=ref_stack))
            python_schema = schema.get('python_schema')
            if python_schema is not None:
                result.extend(self._walk_schema(python_schema, path=path, ref_stack=ref_stack))
            return result

        if schema_type in ('model-field', 'dataclass-field', 'typed-dict-field'):
            inner = schema.get('schema')
            if inner is not None:
                return self._walk_schema(inner, path=path, ref_stack=ref_stack)
            return []

        if schema_type == 'computed-field':
            return_schema = schema.get('return_schema')
            if return_schema is not None:
                return self._walk_schema(return_schema, path=path, ref_stack=ref_stack)
            return []

        # Primitive types (int, str, bool, float, bytes, date, etc.) have no
        # nested models. Nothing to do.
        return []


def audit_model(
    model: type[BaseModel], *, adapter: AuditAdapter | None = None
) -> ModelAuditReport:
    """Audit a model's serialization exposure.

    Read-only: based on the model's fields and compiled core schema. Does
    not dump real instances or execute user code.

    Args:
        model: The Pydantic model class to audit.
        adapter: An optional audit adapter for expanding custom serializers
            and computed fields.

    Returns:
        The audit report.
    """
    return SerializationAuditor(model, adapter=adapter).audit()
