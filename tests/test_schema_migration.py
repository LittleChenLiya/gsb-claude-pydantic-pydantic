"""Tests for pydantic.migration: recursive discriminated-union schema migration."""

from concurrent.futures import ThreadPoolExecutor
from typing import Generic, Literal, TypeVar

import pytest

from pydantic import BaseModel, Field, ValidationError
from pydantic.migration import MigrationConfig, SchemaMigrator


class CatV2(BaseModel):
    kind: Literal['cat'] = 'cat'
    name: str
    lives: int


class DogV2(BaseModel):
    kind: Literal['dog'] = 'dog'
    name: str
    good_boy: bool


PetV2 = CatV2 | DogV2


class OwnerV2(BaseModel):
    full_name: str = Field(alias='fullName')
    pets: list[PetV2]
    nickname: str | None = None
    vet: CatV2 | None = None


class NodeV2(BaseModel):
    label: str
    children: list['NodeV2'] = []


T = TypeVar('T')


class Box(BaseModel, Generic[T]):
    item: T
    count: int


CONFIG = MigrationConfig(
    field_aliases={'name': 'full_name', 'pet_list': 'pets', 'title': 'label', 'kids': 'children', 'entry': 'item'},
    discriminator='kind',
    discriminator_map={'feline': CatV2, 'canine': DogV2},
)

OLD_OWNER = {
    'name': 'Ada',
    'pet_list': [
        {'kind': 'feline', 'name': 'Tom', 'lives': 9},
        {'kind': 'canine', 'name': 'Rex', 'good_boy': True},
    ],
    'nickname': None,
    'vet': {'kind': 'feline', 'name': 'Tom', 'lives': 9},
}


def migrator():
    return SchemaMigrator(CONFIG)


def test_migrate_nested_models_lists_and_optional():
    owner = migrator().migrate(OwnerV2, OLD_OWNER)
    assert owner.full_name == 'Ada'
    assert owner.pets == [CatV2(name='Tom', lives=9), DogV2(name='Rex', good_boy=True)]
    assert owner.nickname is None
    assert owner.vet == CatV2(name='Tom', lives=9)


def test_migrate_rewrites_discriminator_tag():
    owner = migrator().migrate(OwnerV2, OLD_OWNER)
    assert owner.pets[0].kind == 'cat'
    assert owner.pets[1].kind == 'dog'


def test_migrate_top_level_union():
    pet = migrator().migrate(PetV2, {'kind': 'canine', 'name': 'Rex', 'good_boy': False})
    assert pet == DogV2(name='Rex', good_boy=False)


def test_migrate_dict_values():
    class Registry(BaseModel):
        cats: dict[str, CatV2]

    config = MigrationConfig(discriminator='kind', discriminator_map={'feline': CatV2})
    reg = SchemaMigrator(config).migrate(Registry, {'cats': {'a': {'kind': 'feline', 'name': 'Tom', 'lives': 9}}})
    assert reg.cats == {'a': CatV2(name='Tom', lives=9)}


def test_migrate_recursive_model():
    data = {
        'title': 'root',
        'kids': [
            {'title': 'a', 'kids': [{'title': 'a1', 'kids': []}]},
            {'title': 'b'},
        ],
    }
    node = migrator().migrate(NodeV2, data)
    assert node.label == 'root'
    assert node.children[0].children[0].label == 'a1'
    assert node.children[1].label == 'b'


def test_migrate_generic_model():
    box = migrator().migrate(Box[CatV2], {'entry': {'kind': 'feline', 'name': 'Tom', 'lives': 9}, 'count': 1})
    assert box.item == CatV2(name='Tom', lives=9)
    assert box.count == 1


def test_dump_preserves_alias_and_field_order():
    owner = migrator().migrate(OwnerV2, OLD_OWNER)
    dumped = migrator().dump(owner)
    assert list(dumped.keys()) == ['fullName', 'pets', 'nickname', 'vet']
    assert dumped['fullName'] == 'Ada'
    assert list(dumped['pets'][0].keys()) == ['kind', 'name', 'lives']


def test_json_schema_consistent_with_runtime():
    m = migrator()
    owner = m.migrate(OwnerV2, OLD_OWNER)
    dumped = m.dump(owner)
    schema = m.json_schema(OwnerV2)
    # Field order and aliases in the schema match the serialized output.
    assert list(schema['properties'].keys()) == list(dumped.keys())
    # The serialized output round-trips through runtime validation.
    assert OwnerV2.model_validate(dumped) == owner
    # Discriminator tags in the schema match the runtime values.
    cat_schema = schema['$defs']['CatV2']
    assert cat_schema['properties']['kind']['const'] == 'cat'
    assert dumped['pets'][0]['kind'] == 'cat'


def test_unknown_discriminator_raises_structured_error():
    data = {'name': 'Ada', 'pet_list': [{'kind': 'reptile', 'name': 'Rex'}]}
    with pytest.raises(ValidationError) as exc_info:
        migrator().migrate(OwnerV2, data)
    (error,) = exc_info.value.errors()
    assert error['type'] == 'unknown_discriminator'
    assert error['loc'] == ('pets', 0)


def test_missing_discriminator_raises_structured_error():
    data = {'name': 'Ada', 'pet_list': [{'name': 'Tom', 'lives': 9}]}
    with pytest.raises(ValidationError) as exc_info:
        migrator().migrate(OwnerV2, data)
    (error,) = exc_info.value.errors()
    assert error['type'] == 'missing_discriminator'


def test_discriminator_mapped_to_non_branch_model():
    class Other(BaseModel):
        kind: Literal['other'] = 'other'

    config = MigrationConfig(discriminator='kind', discriminator_map={'feline': Other})
    with pytest.raises(ValidationError) as exc_info:
        SchemaMigrator(config).migrate(PetV2, {'kind': 'feline', 'name': 'Tom', 'lives': 9})
    (error,) = exc_info.value.errors()
    assert error['type'] == 'unknown_discriminator'


def test_duplicate_field_alias_mapping_rejected():
    with pytest.raises(ValidationError) as exc_info:
        MigrationConfig(field_aliases=[('a', 'x'), ('a', 'y')])
    (error,) = exc_info.value.errors()
    assert error['type'] == 'duplicate_mapping'


def test_conflicting_field_alias_targets_rejected():
    with pytest.raises(ValidationError) as exc_info:
        MigrationConfig(field_aliases={'a': 'x', 'b': 'x'})
    (error,) = exc_info.value.errors()
    assert error['type'] == 'duplicate_mapping'


def test_duplicate_discriminator_mapping_rejected():
    with pytest.raises(ValidationError) as exc_info:
        MigrationConfig(discriminator_map=[('feline', CatV2), ('feline', DogV2)])
    (error,) = exc_info.value.errors()
    assert error['type'] == 'duplicate_mapping'


def test_circular_reference_raises_structured_error():
    data = {'title': 'root', 'kids': []}
    data['kids'].append(data)
    with pytest.raises(ValidationError) as exc_info:
        migrator().migrate(NodeV2, data)
    (error,) = exc_info.value.errors()
    assert error['type'] == 'circular_reference'
    assert error['loc'] == ('children', 0)


def test_invalid_field_type_raises_validation_error():
    data = {'name': 'Ada', 'pet_list': [{'kind': 'feline', 'name': 'Tom', 'lives': 'not-an-int'}]}
    with pytest.raises(ValidationError):
        migrator().migrate(OwnerV2, data)


def test_missing_required_field_raises_validation_error():
    with pytest.raises(ValidationError):
        migrator().migrate(OwnerV2, {'pet_list': []})


def test_cache_isolated_per_model_config():
    config_b = MigrationConfig(field_aliases={'name': 'full_name', 'pet_list': 'pets'})
    m1 = SchemaMigrator(MigrationConfig(field_aliases={'name': 'full_name', 'pet_list': 'pets'}))
    m2 = SchemaMigrator(config_b)
    assert m1._plan(OwnerV2) is m1._plan(OwnerV2)
    assert m1._plan(OwnerV2) is not m2._plan(OwnerV2)
    # Same model under different configs produces distinct cache entries.
    other = MigrationConfig(field_aliases={'name': 'full_name', 'pet_list': 'pets', 'x': 'y'})
    m3 = SchemaMigrator(other)
    m3._plan(OwnerV2)
    assert len(m3._cache) == 1
    m3_cfg2 = SchemaMigrator(MigrationConfig(field_aliases={'name': 'full_name'}))
    m3_cfg2._plan(OwnerV2)
    assert m3._plan(OwnerV2) is not m3_cfg2._plan(OwnerV2)


def test_concurrent_migration_shares_cache_safely():
    m = migrator()
    expected = m.migrate(OwnerV2, OLD_OWNER)
    cache_size_after_warmup = len(m._cache)

    def work(index):
        data = {**OLD_OWNER, 'name': f'user-{index}'}
        return m.migrate(OwnerV2, data)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(work, range(64)))

    assert [r.full_name for r in results] == [f'user-{i}' for i in range(64)]
    assert all(r.pets == expected.pets for r in results)
    # Concurrent first-use did not create duplicate or corrupted cache entries.
    assert len(m._cache) == cache_size_after_warmup


def test_concurrent_first_use_builds_cache_once():
    m = migrator()

    def work(_):
        return m._plan(OwnerV2)

    with ThreadPoolExecutor(max_workers=16) as pool:
        plans = list(pool.map(work, range(64)))

    assert all(plan is plans[0] for plan in plans)
    assert len(m._cache) == 1
