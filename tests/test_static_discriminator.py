"""Tests for `pydantic.experimental.static_discriminator`."""

from typing import Annotated, Literal

import pytest
from pydantic_core import core_schema

from pydantic import BaseModel, Discriminator, Field, Tag, TypeAdapter, field_validator, model_validator
from pydantic.experimental.static_discriminator import (
    UNREAD,
    DiscriminatorStatus,
    StaticDiscriminatorInterpreter,
    inspect_discriminator,
)


class A(BaseModel):
    kind: Literal['a']
    x: int


class B(BaseModel):
    kind: Literal['b']
    y: str


@pytest.fixture
def interpreter() -> StaticDiscriminatorInterpreter:
    adapter = TypeAdapter(Annotated[A | B, Field(discriminator='kind')])
    return StaticDiscriminatorInterpreter(adapter)


def test_kind_a_reports_branch_a(interpreter):
    report = interpreter.inspect({'kind': 'a', 'x': 1})
    assert report.status is DiscriminatorStatus.MATCHED
    assert report.matched
    assert report.path == ('kind',)
    assert report.tag == 'a'
    assert report.choice_index == 0
    assert report.choice_label == "model 'A'"
    assert report.choice is not None and report.choice['type'] == 'model'


def test_kind_b_reports_branch_b(interpreter):
    report = interpreter.inspect({'kind': 'b', 'y': 's'})
    assert report.status is DiscriminatorStatus.MATCHED
    assert report.tag == 'b'
    assert report.choice_index == 1
    assert report.choice_label == "model 'B'"


def test_missing_tag_reports_reason(interpreter):
    report = interpreter.inspect({'x': 1})
    assert report.status is DiscriminatorStatus.MISSING_TAG
    assert not report.matched
    assert report.tag is UNREAD
    assert report.path is None
    assert report.paths == [('kind',)]
    assert 'no discriminator tag found' in report.reason


def test_unknown_tag_reports_reason(interpreter):
    report = interpreter.inspect({'kind': 'z'})
    assert report.status is DiscriminatorStatus.UNKNOWN_TAG
    assert report.tag == 'z'
    assert report.path == ('kind',)
    assert "'z' does not match any known tag ['a', 'b']" in report.reason


def test_no_validation_is_performed(interpreter):
    # missing required field `x` and wrong type for `x` do not matter statically
    assert interpreter.inspect({'kind': 'a'}).status is DiscriminatorStatus.MATCHED
    assert interpreter.inspect({'kind': 'a', 'x': 'not-an-int'}).status is DiscriminatorStatus.MATCHED


def test_callable_discriminator_is_unknown_and_never_called():
    calls = []

    def disc(v):
        calls.append(v)
        return 'a'

    adapter = TypeAdapter(Annotated[Annotated[A, Tag('a')] | Annotated[B, Tag('b')], Discriminator(disc)])
    report = StaticDiscriminatorInterpreter(adapter).inspect({'kind': 'a', 'x': 1})
    assert report.status is DiscriminatorStatus.UNKNOWN
    assert 'not executed' in report.reason
    assert report.choices == ['a', 'b']
    assert calls == []


def test_alias_candidates_resolved_in_schema_order():
    class AliasedA(BaseModel):
        kind: Literal['a'] = Field(alias='Kind')
        x: int

    class AliasedB(BaseModel):
        kind: Literal['b'] = Field(alias='Kind')
        y: str

    adapter = TypeAdapter(Annotated[AliasedA | AliasedB, Field(discriminator='kind')])
    interpreter = StaticDiscriminatorInterpreter(adapter)

    # the field name is tried first, then the alias
    report = interpreter.inspect({'Kind': 'a'})
    assert report.status is DiscriminatorStatus.MATCHED
    assert report.path == ('Kind',)
    assert report.tag == 'a'
    assert report.choice_label == "model 'AliasedA'"

    # when both are present, the first candidate in schema order wins
    report = interpreter.inspect({'kind': 'b', 'Kind': 'a'})
    assert report.status is DiscriminatorStatus.MATCHED
    assert report.path == ('kind',)
    assert report.tag == 'b'
    assert report.choice_label == "model 'AliasedB'"

    report = interpreter.inspect({})
    assert report.status is DiscriminatorStatus.MISSING_TAG
    assert report.paths == [('kind',), ('Kind',)]


def test_explicit_none_is_not_a_missing_tag(interpreter):
    report = interpreter.inspect({'kind': None})
    assert report.status is DiscriminatorStatus.UNKNOWN_TAG  # read, but not a known tag
    assert report.tag is None
    assert report.tag is not UNREAD
    assert report.path == ('kind',)


def test_none_can_be_a_valid_tag():
    class NoneTagged(BaseModel):
        kind: None = None
        z: int = 0

    adapter = TypeAdapter(Annotated[A | NoneTagged, Field(discriminator='kind')])
    interpreter = StaticDiscriminatorInterpreter(adapter)

    report = interpreter.inspect({'kind': None})
    assert report.status is DiscriminatorStatus.MATCHED
    assert report.tag is None
    assert report.choice_label == "model 'NoneTagged'"

    # ...while a genuinely absent key is still reported as missing
    assert interpreter.inspect({}).status is DiscriminatorStatus.MISSING_TAG


def test_nested_tag_path():
    schema = core_schema.tagged_union_schema(
        choices={'cat': core_schema.str_schema(), 'dog': core_schema.int_schema()},
        discriminator=[['pet', 'pet_type']],
    )
    interpreter = StaticDiscriminatorInterpreter(schema)

    report = interpreter.inspect({'pet': {'pet_type': 'cat', 'name': 'whiskers'}})
    assert report.status is DiscriminatorStatus.MATCHED
    assert report.path == ('pet', 'pet_type')
    assert report.tag == 'cat'
    assert report.choice_index == 0

    assert interpreter.inspect({'pet': {}}).status is DiscriminatorStatus.MISSING_TAG
    assert interpreter.inspect({'pet': {'pet_type': 'fish'}}).status is DiscriminatorStatus.UNKNOWN_TAG
    # an intermediate value of the wrong shape makes the path unsupported
    report = interpreter.inspect({'pet': 5})
    assert report.status is DiscriminatorStatus.UNSUPPORTED
    assert "'pet_type' requires a mapping" in report.reason


def test_nested_path_with_integer_segment():
    schema = core_schema.tagged_union_schema(
        choices=[('x', core_schema.str_schema()), ('y', core_schema.int_schema())],
        discriminator=[['items', 0, 'kind']],
    )
    interpreter = StaticDiscriminatorInterpreter(schema)

    report = interpreter.inspect({'items': [{'kind': 'y'}]})
    assert report.status is DiscriminatorStatus.MATCHED
    assert report.path == ('items', 0, 'kind')
    assert report.tag == 'y'
    assert report.choice_index == 1

    # an empty list means the tag is absent, not unsupported
    assert interpreter.inspect({'items': []}).status is DiscriminatorStatus.MISSING_TAG


def test_validators_are_never_called():
    calls = {'field': 0, 'model': 0}

    class ValidatedA(BaseModel):
        kind: Literal['a']
        x: int

        @field_validator('x')
        @classmethod
        def _check_x(cls, v):
            calls['field'] += 1
            return v

        @model_validator(mode='after')
        def _check_model(self):
            calls['model'] += 1
            return self

    class ValidatedB(BaseModel):
        kind: Literal['b']

    adapter = TypeAdapter(Annotated[ValidatedA | ValidatedB, Field(discriminator='kind')])
    interpreter = StaticDiscriminatorInterpreter(adapter)

    assert interpreter.inspect({'kind': 'a', 'x': 1}).status is DiscriminatorStatus.MATCHED
    assert interpreter.inspect({'kind': 'b'}).status is DiscriminatorStatus.MATCHED
    assert interpreter.inspect({}).status is DiscriminatorStatus.MISSING_TAG
    assert calls == {'field': 0, 'model': 0}


def test_object_attribute_getters_are_never_read(interpreter):
    class Evil:
        @property
        def kind(self):
            raise AssertionError('attribute getter was called')

        def __getattr__(self, name):
            raise AssertionError(f'__getattr__ was called for {name!r}')

    report = interpreter.inspect(Evil())
    assert report.status is DiscriminatorStatus.UNSUPPORTED
    assert 'object attributes are never read' in report.reason

    # a model instance is not a mapping either: unsupported, no attribute reads
    report = interpreter.inspect(A(kind='a', x=1))
    assert report.status is DiscriminatorStatus.UNSUPPORTED


def test_non_tagged_union_schema_is_unsupported():
    assert StaticDiscriminatorInterpreter(TypeAdapter(int)).inspect(1).status is DiscriminatorStatus.UNSUPPORTED
    # a plain smart union has no static discriminator
    report = StaticDiscriminatorInterpreter(TypeAdapter(A | B)).inspect({'kind': 'a'})
    assert report.status is DiscriminatorStatus.UNSUPPORTED
    assert 'not a tagged union' in report.reason


def test_definitions_wrapped_schema_is_resolved():
    # recursive models wrap the tagged union in `definitions`/`definition-ref`
    class Node(BaseModel):
        kind: Literal['node']
        children: 'list[RecursiveUnion]' = []

    RecursiveUnion = Annotated[Node | A, Field(discriminator='kind')]
    adapter = TypeAdapter(RecursiveUnion)
    assert adapter.core_schema['type'] == 'definitions'

    report = StaticDiscriminatorInterpreter(adapter).inspect({'kind': 'node', 'children': []})
    assert report.status is DiscriminatorStatus.MATCHED
    assert report.choice_label == "model 'Node'"


def test_explain_demo(interpreter):
    assert interpreter.inspect({'kind': 'a'}).explain() == (
        "read path ('kind',) -> tag 'a' -> branch #1 of 2 (model 'A')"
    )
    assert interpreter.inspect({'x': 1}).explain() == (
        "no discriminator tag found: none of the candidate paths [('kind',)] is present in the input"
    )
    assert interpreter.inspect({'kind': 'z'}).explain() == (
        "read path ('kind',) -> tag 'z' does not match any known tag ['a', 'b']"
    )


def test_inspect_discriminator_convenience_function():
    adapter = TypeAdapter(Annotated[A | B, Field(discriminator='kind')])
    report = inspect_discriminator(adapter, {'kind': 'b', 'y': 's'})
    assert report.status is DiscriminatorStatus.MATCHED
    assert report.choice_label == "model 'B'"


def test_invalid_adapter_type_raises():
    with pytest.raises(TypeError, match='expected a TypeAdapter or a core schema dict'):
        StaticDiscriminatorInterpreter(42)
