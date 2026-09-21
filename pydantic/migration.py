"""Recursive discriminated-union schema migration utilities.

This module provides :class:`SchemaMigrator`, a reusable migrator that rewrites
data produced by an older schema version into instances of new
:class:`~pydantic.BaseModel` models. It supports:

* renaming fields via old-name -> new-name aliases,
* discriminated unions via an explicit old-tag -> new-model mapping,
* nested models, lists, dicts, `Optional` branches and generic models,
* structured :class:`~pydantic.ValidationError` failures for unknown
  discriminator tags, duplicate mappings and circular references,
* a per-(model, config) migration-plan cache that is safe for concurrent use.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic_core import InitErrorDetails, PydanticCustomError, ValidationError

from .main import BaseModel
from .type_adapter import TypeAdapter

__all__ = ('MigrationConfig', 'SchemaMigrator')

_NoneType = type(None)


def _migration_error(error_type: str, message: str, loc: tuple[Any, ...], input_value: Any) -> ValidationError:
    """Build a structured :class:`ValidationError` for a migration failure."""
    return ValidationError.from_exception_data(
        'SchemaMigration',
        [
            InitErrorDetails(
                type=PydanticCustomError(error_type, message),
                loc=loc,
                input=input_value,
            )
        ],
    )


def _is_model_type(annotation: Any) -> bool:
    """Whether the annotation is a (possibly parametrized generic) Pydantic model."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return True
    origin = get_origin(annotation)
    return isinstance(origin, type) and issubclass(origin, BaseModel)


def _normalize_pairs(pairs, kind):
    """Normalize a mapping or iterable of pairs, rejecting duplicate keys."""
    if pairs is None:
        return ()
    items = list(pairs.items()) if isinstance(pairs, Mapping) else list(pairs)
    seen: set[str] = set()
    normalized: list[tuple[str, Any]] = []
    for key, value in items:
        if key in seen:
            raise _migration_error(
                'duplicate_mapping',
                f'Duplicate {kind} mapping for {key!r}',
                (kind, key),
                key,
            )
        seen.add(key)
        normalized.append((key, value))
    return tuple(normalized)


class MigrationConfig:
    """Configuration describing how old-schema data maps onto new models.

    Args:
        field_aliases: Mapping (or iterable of pairs) of old field name -> new
            field name. Applied at every model level during migration.
        discriminator: Name of the discriminator field in the *old* data.
        discriminator_map: Mapping (or iterable of pairs) of old discriminator
            tag -> new model class. Tags not present in the mapping are
            rejected with a structured `ValidationError`.

    Raises:
        ValidationError: If a mapping contains duplicate keys, or two field
            aliases target the same new field name.
    """

    def __init__(
        self,
        field_aliases=None,
        *,
        discriminator: str = 'type',
        discriminator_map=None,
    ) -> None:
        alias_items = _normalize_pairs(field_aliases, 'field_aliases')
        new_names: set[str] = set()
        for _old, new in alias_items:
            if new in new_names:
                raise _migration_error(
                    'duplicate_mapping',
                    f'Multiple old fields map to the same new field {new!r}',
                    ('field_aliases', new),
                    new,
                )
            new_names.add(new)
        self.field_aliases: dict[str, str] = dict(alias_items)
        self.discriminator = discriminator
        self.discriminator_map: dict[str, type[BaseModel]] = dict(
            _normalize_pairs(discriminator_map, 'discriminator_map')
        )
        self._key = (
            frozenset(self.field_aliases.items()),
            discriminator,
            frozenset(self.discriminator_map.items()),
        )

    def __repr__(self) -> str:
        return (
            f'MigrationConfig(field_aliases={self.field_aliases!r}, '
            f'discriminator={self.discriminator!r}, discriminator_map={self.discriminator_map!r})'
        )


class _ModelPlan:
    """Precomputed per-model migration plan."""

    __slots__ = ('fields',)

    def __init__(self, fields: list) -> None:
        # (new field name, old data key, field annotation)
        self.fields = fields


class SchemaMigrator:
    """Recursively migrate old-schema data into new Pydantic models.

    The migrator compiles a migration plan per `(model, config)` pair and
    caches it; the cache is guarded by a lock so a single migrator instance
    can be shared across threads. Plans are never shared across different
    configs.
    """

    def __init__(self, config: MigrationConfig | None = None) -> None:
        self.config = config if config is not None else MigrationConfig()
        self._cache: dict[Any, _ModelPlan] = {}
        self._lock = threading.Lock()

    def migrate(self, model_cls: Any, data: Any) -> Any:
        """Migrate `data` into an instance of `model_cls`.

        `model_cls` may be a model class, a parametrized generic model, or a
        union/`Optional` annotation. Raises a structured
        :class:`~pydantic.ValidationError` for unknown discriminator tags,
        circular references and invalid input.
        """
        migrated = self._migrate_value(model_cls, data, (), set())
        if _is_model_type(model_cls):
            return model_cls.model_validate(migrated, by_name=True)
        return TypeAdapter(model_cls).validate_python(migrated)

    def dump(self, instance: BaseModel, *, by_alias: bool = True) -> dict[str, Any]:
        """Serialize a migrated model, preserving aliases and field order."""
        return instance.model_dump(by_alias=by_alias)

    def json_schema(self, model_cls: type[BaseModel]) -> dict[str, Any]:
        """Return the JSON Schema of the target model.

        The schema is generated from the same model used for runtime
        validation, so the two are consistent by construction.
        """
        return model_cls.model_json_schema()

    def _plan(self, model_cls: Any) -> _ModelPlan:
        key = (model_cls, self.config._key)
        plan = self._cache.get(key)
        if plan is None:
            with self._lock:
                plan = self._cache.get(key)
                if plan is None:
                    reverse_aliases = {new: old for old, new in self.config.field_aliases.items()}
                    fields = [
                        (name, reverse_aliases.get(name, name), field.annotation)
                        for name, field in model_cls.model_fields.items()
                    ]
                    plan = _ModelPlan(fields)
                    self._cache[key] = plan
        return plan

    def _migrate_value(self, annotation: Any, value: Any, path: tuple, seen: set) -> Any:
        if get_origin(annotation) is Annotated:
            return self._migrate_value(get_args(annotation)[0], value, path, seen)

        origin = get_origin(annotation)

        if _is_model_type(annotation):
            return self._migrate_model(annotation, value, path, seen)

        if origin in (list, tuple, set, frozenset):
            args = get_args(annotation)
            item_type = args[0] if args else Any
            return self._migrate_sequence(origin, item_type, value, path, seen)

        if origin is dict:
            args = get_args(annotation)
            value_type = args[1] if len(args) == 2 else Any
            return self._migrate_mapping(value_type, value, path, seen)

        if origin in (Union, UnionType):
            return self._migrate_union(get_args(annotation), value, path, seen)

        # Primitives and anything else: left as-is for Pydantic to validate.
        return value

    def _migrate_sequence(self, origin: Any, item_type: Any, value: Any, path: tuple, seen: set) -> Any:
        if not isinstance(value, (list, tuple)):
            return value  # let Pydantic report the type error
        self._check_cycle(value, path, seen)
        try:
            items = [self._migrate_value(item_type, item, path + (index,), seen) for index, item in enumerate(value)]
        finally:
            seen.discard(id(value))
        if origin is tuple:
            return tuple(items)
        if origin in (set, frozenset):
            return origin(items)
        return items

    def _migrate_mapping(self, value_type: Any, value: Any, path: tuple, seen: set) -> Any:
        if not isinstance(value, Mapping):
            return value  # let Pydantic report the type error
        self._check_cycle(value, path, seen)
        try:
            return {key: self._migrate_value(value_type, item, path + (key,), seen) for key, item in value.items()}
        finally:
            seen.discard(id(value))

    def _migrate_union(self, args: tuple, value: Any, path: tuple, seen: set) -> Any:
        branches = tuple(arg for arg in args if arg is not _NoneType)
        if value is None:
            if len(branches) != len(args):
                return None
            return value  # let Pydantic report the error
        if len(branches) == 1:
            return self._migrate_value(branches[0], value, path, seen)

        model_branches = tuple(branch for branch in branches if _is_model_type(branch))
        if model_branches:
            # Discriminated union: never silently pick a branch.
            if not isinstance(value, Mapping):
                raise _migration_error(
                    'mapping_expected',
                    'Input should be a mapping to resolve the discriminated union',
                    path,
                    value,
                )
            if self.config.discriminator not in value:
                raise _migration_error(
                    'missing_discriminator',
                    f'Unable to resolve union: discriminator field {self.config.discriminator!r} is missing',
                    path,
                    value,
                )
            tag = value[self.config.discriminator]
            target = self.config.discriminator_map.get(tag)
            if target is None:
                raise _migration_error(
                    'unknown_discriminator',
                    f'No migration mapping for discriminator tag {tag!r}',
                    path,
                    tag,
                )
            if not any(target is branch or get_origin(branch) is target for branch in model_branches):
                raise _migration_error(
                    'unknown_discriminator',
                    f'Discriminator tag {tag!r} maps to {target.__name__}, which is not a branch of this union',
                    path,
                    tag,
                )
            return self._migrate_model(target, value, path, seen)

        # Non-model union (e.g. int | str): nothing structural to migrate.
        return value

    def _rewrite_discriminator(self, model_cls: Any, data: Mapping, kwargs: dict, path: tuple) -> None:
        """Check the old discriminator tag and rewrite it to the new literal tag."""
        if self.config.discriminator not in data or not self.config.discriminator_map:
            return
        tag = data[self.config.discriminator]
        target = self.config.discriminator_map.get(tag)
        if target is None:
            raise _migration_error(
                'unknown_discriminator',
                f'No migration mapping for discriminator tag {tag!r}',
                path,
                tag,
            )
        if target is not model_cls and get_origin(model_cls) is not target:
            raise _migration_error(
                'unknown_discriminator',
                f'Discriminator tag {tag!r} maps to {target.__name__}, not {model_cls!r}',
                path,
                tag,
            )
        new_name = self.config.field_aliases.get(self.config.discriminator, self.config.discriminator)
        field = model_cls.model_fields.get(new_name)
        if field is None:
            return
        if get_origin(field.annotation) is Literal:
            kwargs[new_name] = get_args(field.annotation)[0]

    def _migrate_model(self, model_cls: Any, data: Any, path: tuple, seen: set) -> Any:
        if not isinstance(data, Mapping):
            return data  # let Pydantic report the type error
        self._check_cycle(data, path, seen)
        try:
            plan = self._plan(model_cls)
            kwargs: dict[str, Any] = {}
            consumed: set[str] = set()
            for new_name, old_key, annotation in plan.fields:
                if old_key in data:
                    consumed.add(old_key)
                    kwargs[new_name] = self._migrate_value(annotation, data[old_key], path + (new_name,), seen)
            # Pass through unknown keys so Pydantic's extra config decides.
            for key, item in data.items():
                if key not in consumed and key not in kwargs:
                    kwargs[key] = item
            self._rewrite_discriminator(model_cls, data, kwargs, path)
            return kwargs
        finally:
            seen.discard(id(data))

    def _check_cycle(self, value: Any, path: tuple, seen: set) -> None:
        if id(value) in seen:
            raise _migration_error(
                'circular_reference',
                'Circular reference detected during migration',
                path,
                None,
            )
        seen.add(id(value))
