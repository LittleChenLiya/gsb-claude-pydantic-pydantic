"""Tests for the experimental `pydantic.experimental.union_migration` module."""

from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Generic, Literal, TypeVar

import pytest

from pydantic import BaseModel, Field, ValidationError
from pydantic.experimental.union_migration import UnionMigrator


class CatV2(BaseModel):
    kind: Literal['cat']
    name: str
    lives: int


class DogV2(BaseModel):
    kind: Literal['dog']
    name: str
    good_boy: bool = True


PetV2 = Annotated[CatV2 | DogV2, Field(discriminator='kind')]


class OwnerV2(BaseModel):
    owner_name: str = Field(alias='ownerName')
    pets: list[PetV2]
    favorite: PetV2 | None
    notes: dict[str, PetV2] = {}


@pytest.fixture(name='migrator')
def migrator_fixture() -> UnionMigrator:
    return UnionMigrator(
        discriminator='kind',
        mapping={'cat_v1': CatV2, 'dog_v1': DogV2},
        aliases={'pet_kind': 'kind', 'pet_name': 'name', 'owner': 'owner_name'},
    )


def old_owner_data() -> dict:
    return {
        'owner': 'alice',
        'pets': [
            {'pet_kind': 'cat_v1', 'pet_name': 'tom', 'lives': 9},
            {'pet_kind': 'dog_v1', 'pet_name': 'rex', 'good_boy': False},
        ],
        'favorite': {'pet_kind': 'cat_v1', 'pet_name': 'tom', 'lives': 9},
        'notes': {'stray': {'pet_kind': 'dog_v1', 'pet_name': 'buddy'}},
    }


def test_migrate_nested_models_collections_and_optional(migrator: UnionMigrator) -> None:
    owner = migrator.migrate(OwnerV2, old_owner_data())

    assert owner.owner_name == 'alice'
    assert [type(pet) for pet in owner.pets] == [CatV2, DogV2]
    assert owner.pets[0].name == 'tom'
    assert owner.pets[1].good_boy is False
    assert isinstance(owner.favorite, CatV2)
    assert isinstance(owner.notes['stray'], DogV2)
    assert owner.notes['stray'].good_boy is True


def test_migrate_optional_none_branch(migrator: UnionMigrator) -> None:
    data = old_owner_data()
    data['favorite'] = None
    owner = migrator.migrate(OwnerV2, data)
    assert owner.favorite is None


def test_serialization_preserves_aliases_and_field_order(migrator: UnionMigrator) -> None:
    owner = migrator.migrate(OwnerV2, old_owner_data())

    dumped = owner.model_dump(by_alias=True, mode='json')
    assert list(dumped) == ['ownerName', 'pets', 'favorite', 'notes']
    assert list(dumped['pets'][0]) == ['kind', 'name', 'lives']
    assert dumped['pets'][0]['kind'] == 'cat'
    assert dumped['pets'][1]['kind'] == 'dog'

    # Round trip: serialized output validates against the new schema as-is.
    assert OwnerV2.model_validate(dumped) == owner


def test_unknown_discriminator_raises_structured_error(migrator: UnionMigrator) -> None:
    data = old_owner_data()
    data['pets'][1]['pet_kind'] = 'fish_v1'

    with pytest.raises(ValidationError) as exc_info:
        migrator.migrate(OwnerV2, data)

    (error,) = exc_info.value.errors()
    assert error['type'] == 'unknown_discriminator'
    assert error['loc'] == ('pets', 1, 'pet_kind')
    assert error['ctx']['tag'] == 'fish_v1'
    assert 'cat_v1' in error['ctx']['expected']


def test_missing_discriminator_raises_structured_error(migrator: UnionMigrator) -> None:
    data = old_owner_data()
    del data['favorite']['pet_kind']

    with pytest.raises(ValidationError) as exc_info:
        migrator.migrate(OwnerV2, data)

    (error,) = exc_info.value.errors()
    assert error['type'] == 'unknown_discriminator'
    assert error['loc'] == ('favorite', 'pet_kind')
    assert error['ctx']['tag'] is None


def test_duplicate_discriminator_mapping_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        UnionMigrator(discriminator='kind', mapping={'cat_v1': CatV2, 'cat_v2': CatV2})

    (error,) = exc_info.value.errors()
    assert error['type'] == 'duplicate_mapping'
    assert error['loc'] == ('mapping', 'cat_v2')


def test_duplicate_alias_mapping_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        UnionMigrator(
            discriminator='kind',
            mapping={'cat_v1': CatV2},
            aliases={'pet_name': 'name', 'old_name': 'name'},
        )

    (error,) = exc_info.value.errors()
    assert error['type'] == 'duplicate_mapping'
    assert error['loc'] == ('aliases', 'old_name')


def test_alias_shadowing_discriminator_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        UnionMigrator(
            discriminator='kind',
            mapping={'cat_v1': CatV2},
            aliases={'kind': 'other'},
        )

    (error,) = exc_info.value.errors()
    assert error['type'] == 'duplicate_mapping'


def test_invalid_mapping_value_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        UnionMigrator(discriminator='kind', mapping={'cat_v1': int})  # type: ignore[dict-item]

    (error,) = exc_info.value.errors()
    assert error['type'] == 'invalid_mapping'


def test_recursive_model_migration() -> None:
    class NodeV2(BaseModel):
        node_id: int
        children: list['NodeV2'] = []

    NodeV2.model_rebuild()

    migrator = UnionMigrator(
        discriminator='kind',
        mapping={},
        aliases={'id': 'node_id', 'kids': 'children'},
    )
    data = {'id': 1, 'kids': [{'id': 2, 'kids': [{'id': 3}]}]}

    node = migrator.migrate(NodeV2, data)

    assert node.node_id == 1
    assert node.children[0].node_id == 2
    assert node.children[0].children[0].node_id == 3
    assert node.model_dump() == {
        'node_id': 1,
        'children': [{'node_id': 2, 'children': [{'node_id': 3, 'children': []}]}],
    }


def test_circular_reference_raises_structured_error() -> None:
    class NodeV2(BaseModel):
        node_id: int
        children: list['NodeV2'] = []

    NodeV2.model_rebuild()

    migrator = UnionMigrator(discriminator='kind', mapping={}, aliases={'id': 'node_id', 'kids': 'children'})
    data: dict = {'id': 1, 'kids': []}
    data['kids'].append(data)

    with pytest.raises(ValidationError) as exc_info:
        migrator.migrate(NodeV2, data)

    (error,) = exc_info.value.errors()
    assert error['type'] == 'circular_reference'
    assert error['loc'] == ('kids', 0)


def test_generic_model_migration(migrator: UnionMigrator) -> None:
    T = TypeVar('T')

    class EnvelopeV2(BaseModel, Generic[T]):
        seq: int
        payload: T

    data = {'seq': 7, 'payload': {'pet_kind': 'cat_v1', 'pet_name': 'tom', 'lives': 9}}
    envelope = migrator.migrate(EnvelopeV2[CatV2], data)

    assert envelope.seq == 7
    assert isinstance(envelope.payload, CatV2)
    assert envelope.payload.kind == 'cat'
    assert envelope.payload.name == 'tom'


def test_json_schema_consistent_with_runtime(migrator: UnionMigrator) -> None:
    owner = migrator.migrate(OwnerV2, old_owner_data())
    dumped = owner.model_dump(by_alias=True, mode='json')

    schema = migrator.json_schema(OwnerV2)

    # Schema properties use the same aliases and order as runtime serialization.
    assert list(schema['properties']) == list(dumped)
    assert schema['required'] == ['ownerName', 'pets', 'favorite']

    # The discriminated union schema references exactly the mapped models.
    pet_schema = schema['$defs']['CatV2']['properties']['kind']
    assert pet_schema['const'] == 'cat'
    assert schema['$defs']['DogV2']['properties']['kind']['const'] == 'dog'

    # Every discriminator value accepted at runtime is declared in the schema.
    assert {member['const'] for member in (pet_schema, {'const': 'dog'})} == {'cat', 'dog'}

    # The serialized output of a migration satisfies the schema's declared shape.
    for key in schema['required']:
        assert key in dumped


def test_cache_isolated_per_model_and_config() -> None:
    migrator_a = UnionMigrator(
        discriminator='kind',
        mapping={'cat_v1': CatV2},
        aliases={'pet_kind': 'kind', 'pet_name': 'name'},
    )
    migrator_b = UnionMigrator(
        discriminator='kind',
        mapping={'cat_v1': CatV2},
        aliases={'kind': 'kind', 'name': 'name'},
    )

    cat_a = migrator_a.migrate(CatV2, {'pet_kind': 'cat_v1', 'pet_name': 'tom', 'lives': 9})
    cat_b = migrator_b.migrate(CatV2, {'kind': 'cat_v1', 'name': 'tom', 'lives': 9})
    assert cat_a == cat_b

    # Plans are cached per (model, config): same model, different configs -> distinct entries.
    assert len(migrator_a._plans) == 1
    assert len(migrator_b._plans) == 1
    key_a = next(iter(migrator_a._plans))
    key_b = next(iter(migrator_b._plans))
    assert key_a[0] != key_b[0]
    assert key_a[1] is CatV2 and key_b[1] is CatV2


def test_concurrent_migration_and_cache_recovery(migrator: UnionMigrator) -> None:
    def work(index: int) -> object:
        if index % 10 == 0:
            # Interleave failures: they must not corrupt the shared plan cache.
            bad = old_owner_data()
            bad['pets'][0]['pet_kind'] = 'fish_v1'
            with pytest.raises(ValidationError):
                migrator.migrate(OwnerV2, bad)
            return None
        data = old_owner_data()
        data['owner'] = f'owner_{index}'
        return migrator.migrate(OwnerV2, data)

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(work, range(200)))

    for index, result in enumerate(results):
        if index % 10 == 0:
            assert result is None
        else:
            assert isinstance(result, OwnerV2)
            assert result.owner_name == f'owner_{index}'

    # The cache holds a single plan per (model, config), regardless of call count.
    assert len(migrator._plans) == len({key for key in migrator._plans}) <= 4

    # Recovery: valid migrations still succeed after the concurrent failures.
    owner = migrator.migrate(OwnerV2, old_owner_data())
    assert owner.owner_name == 'alice'
