"""Experimental recursive discriminated-union schema migrator.

This module provides `UnionMigrator`, a reusable utility to migrate data produced
for an *old* version of a model schema into a *new* set of models. It supports:

* renaming fields via old-version aliases,
* remapping discriminated union members via an explicit discriminator mapping,
* recursive traversal of nested models, lists, dicts and `Optional`/union branches,
* structured `ValidationError` failures for unknown discriminator values,
  duplicate mappings and circular references,
* a per-(model, config) isolated, thread-safe plan cache.

!!! warning "Experimental"
    This feature is experimental and subject to change or removal.
"""

from __future__ import annotations

import threading
import types
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Union, get_args, get_origin

from pydantic_core import InitErrorDetails, PydanticCustomError, ValidationError

if TYPE_CHECKING:
    from ..main import BaseModel

__all__ = ('UnionMigrator',)

_NoneType = type(None)


def _custom_error(error_type: str, message: str, ctx: dict[str, Any]) -> PydanticCustomError:
    return PydanticCustomError(error_type, message, ctx)


def _raise_validation_error(title: str, error: PydanticCustomError, loc: tuple[Any, ...], input_value: Any) -> None:
    raise ValidationError.from_exception_data(
        title,
        [InitErrorDetails(type=error, loc=loc, input=input_value)],
    )


@dataclass(frozen=True)
class _FieldPlan:
    """How to move one field of the old data into the new model."""

    new_key: str
    old_key: str
    annotation: Any


@dataclass(frozen=True)
class _ModelPlan:
    model: type[BaseModel]
    fields: tuple[_FieldPlan, ...]
    # Literal tag of the discriminator field if this model is a union member, else None.
    new_tag: Any


class UnionMigrator:
    """Migrate old-version data into new discriminated-union aware models.

    Args:
        discriminator: Name of the discriminator field on the *new* models.
        mapping: Mapping of *old* discriminator values to the *new* model classes.
        aliases: Mapping of *old* field names (aliases) to *new* field names. Aliases
            are applied at every nesting level of the migrated data.

    Raises:
        ValidationError: If the configuration contains duplicate mappings (two old
            discriminator values pointing to the same model, two old aliases pointing
            to the same new field, or an alias colliding with the discriminator).
    """

    def __init__(
        self,
        *,
        discriminator: str,
        mapping: dict[Any, type[BaseModel]],
        aliases: dict[str, str] | None = None,
    ) -> None:
        from ..main import BaseModel

        aliases = dict(aliases or {})
        errors: list[InitErrorDetails] = []

        seen_models: dict[type[BaseModel], Any] = {}
        for old_tag, model in mapping.items():
            if not (isinstance(model, type) and issubclass(model, BaseModel)):
                errors.append(
                    InitErrorDetails(
                        type=_custom_error(
                            'invalid_mapping',
                            'Mapping values must be Pydantic model classes, got {model}',
                            {'tag': old_tag, 'model': repr(model)},
                        ),
                        loc=('mapping', old_tag),
                        input=model,
                    )
                )
            elif model in seen_models:
                errors.append(
                    InitErrorDetails(
                        type=_custom_error(
                            'duplicate_mapping',
                            'Duplicate discriminator mapping: tags {first} and {tag} both map to model {model}',
                            {'tag': old_tag, 'first': seen_models[model], 'model': model.__qualname__},
                        ),
                        loc=('mapping', old_tag),
                        input=old_tag,
                    )
                )
            else:
                seen_models[model] = old_tag

        seen_new_names: dict[str, str] = {}
        for old_name, new_name in aliases.items():
            if old_name == discriminator and new_name != discriminator:
                errors.append(
                    InitErrorDetails(
                        type=_custom_error(
                            'duplicate_mapping',
                            'Alias {old} shadows the discriminator field {discriminator}',
                            {'old': old_name, 'discriminator': discriminator},
                        ),
                        loc=('aliases', old_name),
                        input=old_name,
                    )
                )
            elif new_name in seen_new_names:
                errors.append(
                    InitErrorDetails(
                        type=_custom_error(
                            'duplicate_mapping',
                            'Duplicate alias mapping: old fields {first} and {old} both map to new field {new}',
                            {'old': old_name, 'first': seen_new_names[new_name], 'new': new_name},
                        ),
                        loc=('aliases', old_name),
                        input=old_name,
                    )
                )
            else:
                seen_new_names[new_name] = old_name

        if errors:
            raise ValidationError.from_exception_data('UnionMigrator', errors)

        self.discriminator = discriminator
        self.mapping = dict(mapping)
        self.aliases = aliases
        # new field name -> old field name, used to read old data.
        self._old_keys = {new: old for old, new in aliases.items()}
        # The old key under which the discriminator value is stored.
        self._old_discriminator_key = self._old_keys.get(discriminator, discriminator)
        self._config_fingerprint = (
            discriminator,
            tuple(sorted((repr(tag), m.__module__ + '.' + m.__qualname__) for tag, m in mapping.items())),
            tuple(sorted(aliases.items())),
        )
        self._plans: dict[Any, _ModelPlan] = {}
        self._lock = threading.Lock()

    def migrate(self, model: type[BaseModel], data: Any) -> BaseModel:
        """Migrate old-schema `data` into an instance of `model`.

        Raises:
            ValidationError: If a discriminator value is unknown, a circular
                reference is found in the data, or the migrated data does not
                validate against the new model.
        """
        plan = self._get_plan(model)
        transformed = self._transform_model(plan, data, loc=(), active=set())
        return model.model_validate(transformed)

    def json_schema(self, model: type[BaseModel], *, mode: str = 'validation') -> dict[str, Any]:
        """Return the JSON Schema of `model`, consistent with runtime validation."""
        return model.model_json_schema(mode=mode)

    def _get_plan(self, model: type[BaseModel]) -> _ModelPlan:
        key = (self._config_fingerprint, model)
        plan = self._plans.get(key)
        if plan is None:
            with self._lock:
                plan = self._plans.get(key)
                if plan is None:
                    plan = self._build_plan(model)
                    self._plans[key] = plan
        return plan

    def _build_plan(self, model: type[BaseModel]) -> _ModelPlan:
        fields: list[_FieldPlan] = []
        new_tag: Any = None
        for name, info in model.model_fields.items():
            if isinstance(info.validation_alias, str):
                new_key = info.validation_alias
            elif isinstance(info.alias, str):
                new_key = info.alias
            else:
                new_key = name
            old_key = self._old_keys.get(name, name)
            if name == self.discriminator:
                literal_args = list(get_args(info.annotation))
                new_tag = literal_args[0] if literal_args else None
            fields.append(_FieldPlan(new_key=new_key, old_key=old_key, annotation=info.annotation))
        return _ModelPlan(model=model, fields=tuple(fields), new_tag=new_tag)

    # -- transformation -------------------------------------------------

    def _transform(self, annotation: Any, value: Any, loc: tuple[Any, ...], active: set[int]) -> Any:
        if value is None:
            return None

        discriminator: Any = None
        while get_origin(annotation) is Annotated:
            for meta in get_args(annotation)[1:]:
                meta_disc = getattr(meta, 'discriminator', None)
                if meta_disc is not None:
                    discriminator = getattr(meta_disc, 'discriminator', meta_disc)
            annotation = get_args(annotation)[0]

        origin = get_origin(annotation)
        if origin in (Union, types.UnionType):
            members = [arg for arg in get_args(annotation) if arg is not _NoneType]
            if len(members) == 1:
                return self._transform(members[0], value, loc, active)
            if discriminator is not None:
                return self._transform_union(discriminator, value, loc, active)
            return value

        if self._is_model(annotation):
            return self._transform_model(self._get_plan(annotation), value, loc, active)

        if origin in (list, tuple, set, frozenset):
            args = get_args(annotation)
            item_annotation = args[0] if args else Any
            if isinstance(value, (list, tuple, set, frozenset)):
                return [
                    self._transform(item_annotation, item, loc + (index,), active) for index, item in enumerate(value)
                ]
            return value

        if origin is dict:
            args = get_args(annotation)
            value_annotation = args[1] if len(args) == 2 else Any
            if isinstance(value, Mapping):
                return {
                    key: self._transform(value_annotation, item, loc + (key,), active) for key, item in value.items()
                }
            return value

        return value

    def _transform_union(self, discriminator: str, value: Any, loc: tuple[Any, ...], active: set[int]) -> Any:
        if not isinstance(value, Mapping):
            return value
        old_disc_key = self._old_keys.get(discriminator, discriminator)
        old_tag = value.get(old_disc_key)
        model = self.mapping.get(old_tag)
        if model is None:
            _raise_validation_error(
                'UnionMigrator',
                _custom_error(
                    'unknown_discriminator',
                    'Unknown discriminator value {tag} for discriminator {discriminator}; expected one of {expected}',
                    {
                        'tag': old_tag,
                        'discriminator': discriminator,
                        'expected': ', '.join(repr(tag) for tag in self.mapping),
                    },
                ),
                loc + (old_disc_key,),
                old_tag,
            )
        return self._transform_model(self._get_plan(model), value, loc, active)

    def _transform_model(self, plan: _ModelPlan, data: Any, loc: tuple[Any, ...], active: set[int]) -> Any:
        if not isinstance(data, Mapping):
            return data
        data_id = id(data)
        if data_id in active:
            _raise_validation_error(
                'UnionMigrator',
                _custom_error(
                    'circular_reference',
                    'Circular reference detected while migrating model {model}',
                    {'model': plan.model.__qualname__},
                ),
                loc,
                None,
            )
        active.add(data_id)
        try:
            out: dict[str, Any] = {}
            consumed: set[Any] = set()
            for field_plan in plan.fields:
                old_key = field_plan.old_key
                if old_key not in data:
                    continue
                consumed.add(old_key)
                value = data[old_key]
                if old_key == self._old_discriminator_key and plan.new_tag is not None:
                    out[field_plan.new_key] = plan.new_tag
                else:
                    out[field_plan.new_key] = self._transform(field_plan.annotation, value, loc + (old_key,), active)
            # Carry over extra keys so the new model's `extra` config decides.
            return {**out, **{key: item for key, item in data.items() if key not in consumed}}
        finally:
            active.discard(data_id)

    @staticmethod
    def _is_model(annotation: Any) -> bool:
        from ..main import BaseModel

        return isinstance(annotation, type) and issubclass(annotation, BaseModel)
