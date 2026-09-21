# Schema Migration

Pydantic ships a reusable, recursive discriminated-union schema migrator in
`pydantic.migration`. Given the field aliases and the discriminator mapping of
an *old* schema version, `SchemaMigrator` rewrites old data into instances of
your *new* models — including nested models, lists, dicts, `Optional` branches
and generic models.

## Configuration

Migration behavior is described by `MigrationConfig`:

* `field_aliases`: old field name -> new field name. Applied at every model
  level. Accepts a mapping or an iterable of pairs.
* `discriminator`: the name of the discriminator field in the old data
  (defaults to `'type'`).
* `discriminator_map`: old discriminator tag -> new model class. The tag in
  the data is validated against this mapping and rewritten to the new model's
  `Literal` tag.

Duplicate keys in either mapping — or two aliases targeting the same new
field — are rejected immediately with a structured
`ValidationError` of type `'duplicate_mapping'`.

## Usage

```python
from typing import Literal

from pydantic import BaseModel
from pydantic.migration import MigrationConfig, SchemaMigrator


class CatV2(BaseModel):
    kind: Literal['cat'] = 'cat'
    name: str


class DogV2(BaseModel):
    kind: Literal['dog'] = 'dog'
    name: str


class OwnerV2(BaseModel):
    full_name: str
    pets: list[CatV2 | DogV2]


config = MigrationConfig(
    field_aliases={'name': 'full_name', 'pet_list': 'pets'},
    discriminator='kind',
    discriminator_map={'feline': CatV2, 'canine': DogV2},
)
migrator = SchemaMigrator(config)

old_data = {
    'name': 'Ada',
    'pet_list': [
        {'kind': 'feline', 'name': 'Tom'},
        {'kind': 'canine', 'name': 'Rex'},
    ],
}
owner = migrator.migrate(OwnerV2, old_data)
print(owner.pets)
#> [CatV2(kind='cat', name='Tom'), DogV2(kind='dog', name='Rex')]
```

## Serialization and JSON Schema

`migrator.dump(instance)` serializes with `by_alias=True`, preserving field
aliases and the model's field definition order. `migrator.json_schema(Model)`
returns the model's JSON Schema; because the schema is generated from the same
model used for runtime validation, the two are consistent by construction —
serialized output of a migrated model always validates against the schema.

## Error handling

All structural migration failures raise a structured
`pydantic.ValidationError` (inspectable via `.errors()`); the migrator never
silently picks a union branch:

| Error type | Cause |
| ---------- | ----- |
| `duplicate_mapping` | A mapping contains duplicate keys or conflicting targets. |
| `unknown_discriminator` | A tag is missing from `discriminator_map` or maps to a non-branch model. |
| `missing_discriminator` | A union value lacks the discriminator field. |
| `circular_reference` | The input data contains a cycle. |
| `mapping_expected` | A union value is not a mapping. |

Ordinary type mismatches (missing fields, wrong value types) surface as
standard Pydantic validation errors.

## Caching and concurrency

Each `SchemaMigrator` compiles a migration plan per `(model, config)` pair
and caches it under a lock, so a single migrator instance can be shared safely
across threads. Plans are keyed by the full config — different configs never
share cache entries.

## Limitations

* Discriminated unions are resolved exclusively through
  `discriminator_map`; unions of models without a configured mapping fail
  loudly instead of guessing.
* Recursive *models* are supported; recursive (cyclic) *data* is rejected.
* Unknown keys are passed through to the target model, so the model
  `extra` config decides whether they are kept or rejected.
