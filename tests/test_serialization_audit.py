"""Tests for the read-only serialization exposure auditor."""

from __future__ import annotations

from typing import Annotated

from pydantic_core import core_schema

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, SecretStr, computed_field, field_serializer
from pydantic.experimental.serialization_audit import (
    ModelAuditReport,
    audit_model,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class Address(BaseModel):
    street: str
    city: str


class User(BaseModel):
    name: str = Field(serialization_alias='userName')
    age: int = Field(exclude=True)
    password: SecretStr
    hidden: int = Field(repr=False)
    address: Address
    tags: list[str] = []


# ---------------------------------------------------------------------------
# Basic audit tests
# ---------------------------------------------------------------------------


def test_audit_returns_report() -> None:
    report = audit_model(User)
    assert isinstance(report, ModelAuditReport)
    assert report.model_name == 'User'
    assert len(report.fields) == 6


def test_serialization_alias() -> None:
    report = audit_model(User)
    name_field = next(f for f in report.fields if f.name == 'name')
    assert name_field.serialization_alias == 'userName'


def test_repr_false_not_excluded() -> None:
    """repr=False must NOT be treated as serialization exclusion."""
    report = audit_model(User)
    hidden_field = next(f for f in report.fields if f.name == 'hidden')
    assert hidden_field.repr is False
    assert hidden_field.excluded is False
    assert hidden_field.serializer == 'none'


def test_exclude_true_excluded() -> None:
    """exclude=True must be marked as excluded."""
    report = audit_model(User)
    age_field = next(f for f in report.fields if f.name == 'age')
    assert age_field.excluded is True


def test_secret_str_default_protection() -> None:
    """SecretStr must show default-protected, not unknown."""
    report = audit_model(User)
    pw_field = next(f for f in report.fields if f.name == 'password')
    assert pw_field.sensitive == 'secret-str'
    assert pw_field.serializer == 'default-protected'
    assert pw_field.excluded is False


def test_nested_model_expanded() -> None:
    """Nested model fields must be expanded (not unknown)."""
    report = audit_model(User)
    addr_field = next(f for f in report.fields if f.name == 'address')
    assert addr_field.serializer == 'none'
    assert len(addr_field.children) == 2
    child_names = {c.name for c in addr_field.children}
    assert child_names == {'street', 'city'}


def test_list_of_primitives() -> None:
    """A list of primitives has no nested model children."""
    report = audit_model(User)
    tags_field = next(f for f in report.fields if f.name == 'tags')
    assert tags_field.children == ()


# ---------------------------------------------------------------------------
# Alias tests
# ---------------------------------------------------------------------------


def test_alias_generator() -> None:
    class Model(BaseModel):
        model_config = ConfigDict(alias_generator=lambda x: x.upper())
        foo: str
        bar: int

    report = audit_model(Model)
    foo_field = next(f for f in report.fields if f.name == 'foo')
    bar_field = next(f for f in report.fields if f.name == 'bar')
    assert foo_field.serialization_alias == 'FOO'
    assert bar_field.serialization_alias == 'BAR'


def test_alias_vs_serialization_alias() -> None:
    class Model(BaseModel):
        a: str = Field(alias='aAlias', serialization_alias='aSer')
        b: str = Field(alias='bAlias')

    report = audit_model(Model)
    a_field = next(f for f in report.fields if f.name == 'a')
    b_field = next(f for f in report.fields if f.name == 'b')
    assert a_field.serialization_alias == 'aSer'
    assert b_field.serialization_alias == 'bAlias'


def test_default_alias_is_field_name() -> None:
    class Model(BaseModel):
        plain: str

    report = audit_model(Model)
    plain_field = next(f for f in report.fields if f.name == 'plain')
    assert plain_field.serialization_alias == 'plain'


# ---------------------------------------------------------------------------
# Inheritance tests
# ---------------------------------------------------------------------------


class BaseModel2(BaseModel):
    base_field: str
    base_secret: SecretStr


class MidModel(BaseModel2):
    mid_field: int


class ChildModel(MidModel):
    child_field: float
    base_field: str = 'overridden'


def test_inheritance_source() -> None:
    report = audit_model(ChildModel)
    base_field = next(f for f in report.fields if f.name == 'base_field')
    mid_field = next(f for f in report.fields if f.name == 'mid_field')
    child_field = next(f for f in report.fields if f.name == 'child_field')
    assert base_field.source == 'BaseModel2'
    assert mid_field.source == 'MidModel'
    assert child_field.source == 'ChildModel'


def test_inherited_secret_still_protected() -> None:
    report = audit_model(ChildModel)
    secret_field = next(f for f in report.fields if f.name == 'base_secret')
    assert secret_field.source == 'BaseModel2'
    assert secret_field.sensitive == 'secret-str'
    assert secret_field.serializer == 'default-protected'


def test_inherited_nested_model() -> None:
    class Inner(BaseModel):
        x: int

    class Parent(BaseModel):
        inner: Inner

    class Child(Parent):
        y: str

    report = audit_model(Child)
    inner_field = next(f for f in report.fields if f.name == 'inner')
    assert inner_field.source == 'Parent'
    assert len(inner_field.children) == 1
    assert inner_field.children[0].name == 'x'


# ---------------------------------------------------------------------------
# Custom serializer tests (unknown without adapter)
# ---------------------------------------------------------------------------


def test_custom_serializer_unknown_without_adapter() -> None:
    class Model(BaseModel):
        custom: Annotated[int, PlainSerializer(lambda v: str(v), return_type=str)]

    report = audit_model(Model)
    custom_field = next(f for f in report.fields if f.name == 'custom')
    assert custom_field.serializer == 'unknown'
    assert custom_field.children == ()


def test_field_serializer_unknown_without_adapter() -> None:
    class Model(BaseModel):
        a: int

        @field_serializer('a')
        def ser_a(self, v: int) -> str:
            return str(v)

    report = audit_model(Model)
    a_field = next(f for f in report.fields if f.name == 'a')
    assert a_field.serializer == 'unknown'


def test_computed_field_unknown_without_adapter() -> None:
    class Model(BaseModel):
        x: int

        @computed_field
        @property
        def doubled(self) -> int:
            return self.x * 2

    report = audit_model(Model)
    doubled_field = next(f for f in report.fields if f.name == 'doubled')
    assert doubled_field.computed is True
    assert doubled_field.serializer == 'unknown'
    assert doubled_field.children == ()


# ---------------------------------------------------------------------------
# Adapter tests (callback count)
# ---------------------------------------------------------------------------


class CountingAdapter:
    """Test adapter that counts callback invocations."""

    def __init__(self, *, expand_serializers: bool = True, expand_computed: bool = False) -> None:
        self.serializer_calls = 0
        self.computed_calls = 0
        self._expand_serializers = expand_serializers
        self._expand_computed = expand_computed

    def expand_serializer(self, schema, *, field_name, path):
        self.serializer_calls += 1
        if self._expand_serializers and isinstance(schema, dict) and 'return_schema' in schema:
            return schema['return_schema']
        return None

    def expand_computed_field(self, property_name, *, path):
        self.computed_calls += 1
        if self._expand_computed:
            return core_schema.int_schema()
        return None


def test_adapter_callback_count() -> None:
    class Model(BaseModel):
        a: Annotated[int, PlainSerializer(lambda v: str(v), return_type=str)]
        b: Annotated[int, PlainSerializer(lambda v: v, return_type=int)]
        c: int

        @computed_field
        @property
        def comp(self) -> str:
            return 'x'

    adapter = CountingAdapter()
    report = audit_model(Model, adapter=adapter)

    # 2 serializer calls + 1 computed call = 3 total
    assert report.adapter_calls == 3
    assert adapter.serializer_calls == 2
    assert adapter.computed_calls == 1


def test_adapter_expand_serializer() -> None:
    class Model(BaseModel):
        a: Annotated[int, PlainSerializer(lambda v: str(v), return_type=str)]

    adapter = CountingAdapter()
    report = audit_model(Model, adapter=adapter)
    a_field = next(f for f in report.fields if f.name == 'a')
    assert a_field.serializer == 'expanded'
    assert adapter.serializer_calls == 1


def test_adapter_expand_computed_field() -> None:
    class Model(BaseModel):
        x: int

        @computed_field
        @property
        def comp(self) -> int:
            return 1

    adapter = CountingAdapter(expand_computed=True)
    report = audit_model(Model, adapter=adapter)
    comp_field = next(f for f in report.fields if f.name == 'comp')
    assert comp_field.serializer == 'expanded'
    assert adapter.computed_calls == 1


def test_adapter_none_keeps_unknown() -> None:
    """When adapter returns None, the field stays unknown."""
    class Model(BaseModel):
        a: Annotated[int, PlainSerializer(lambda v: str(v), return_type=str)]

    adapter = CountingAdapter(expand_serializers=False)
    report = audit_model(Model, adapter=adapter)
    a_field = next(f for f in report.fields if f.name == 'a')
    assert a_field.serializer == 'unknown'
    assert adapter.serializer_calls == 1


# ---------------------------------------------------------------------------
# Recursion tests (path stack)
# ---------------------------------------------------------------------------


class Node(BaseModel):
    value: int
    children: list[Node] = []


class Container(BaseModel):
    name: str
    node: Node
    backup: Node


def test_recursion_stops_at_reference() -> None:
    """Recursive models must stop at the reference (path stack)."""
    report = audit_model(Node)
    value_field = next(f for f in report.fields if f.name == 'value')
    children_field = next(f for f in report.fields if f.name == 'children')
    assert value_field.recursive is False
    assert children_field.recursive is False
    # The list is transparent: its children are the Node model's fields.
    assert len(children_field.children) == 2
    child_names = {c.name for c in children_field.children}
    assert child_names == {'value', 'children'}
    # The nested 'children' field contains a recursive reference to Node.
    nested_children = next(c for c in children_field.children if c.name == 'children')
    assert nested_children.recursive is False
    assert len(nested_children.children) == 1
    recursive_ref = nested_children.children[0]
    assert recursive_ref.recursive is True
    assert recursive_ref.recursion_ref is not None
    assert 'Node' in recursive_ref.recursion_ref


def test_recursion_does_not_infinite_loop() -> None:
    """Auditing a recursive model must terminate (no infinite loop)."""
    report = audit_model(Node)
    # If we get here, the audit terminated.
    assert report is not None


def test_two_paths_to_same_model_both_preserved() -> None:
    """Two different paths to the same model must both be preserved."""
    report = audit_model(Container)
    node_field = next(f for f in report.fields if f.name == 'node')
    backup_field = next(f for f in report.fields if f.name == 'backup')

    # Both paths must have children (not deleted by a global visited set).
    assert len(node_field.children) == 2
    assert len(backup_field.children) == 2
    node_child_names = {c.name for c in node_field.children}
    backup_child_names = {c.name for c in backup_field.children}
    assert node_child_names == {'value', 'children'}
    assert backup_child_names == {'value', 'children'}


def test_recursive_refs_recorded() -> None:
    """Recursive refs must be recorded in the report."""
    report = audit_model(Container)
    assert len(report.recursive_refs) >= 1
    for ref in report.recursive_refs:
        assert 'Node' in ref


# ---------------------------------------------------------------------------
# Report output tests
# ---------------------------------------------------------------------------


def test_report_to_dict() -> None:
    report = audit_model(User)
    d = report.to_dict()
    assert d['model_name'] == 'User'
    assert isinstance(d['fields'], list)
    assert len(d['fields']) == 6


def test_report_to_json() -> None:
    report = audit_model(User)
    j = report.to_json()
    assert isinstance(j, str)
    # Must be valid JSON.
    import json

    parsed = json.loads(j)
    assert parsed['model_name'] == 'User'


def test_report_to_tree() -> None:
    report = audit_model(User)
    tree = report.to_tree()
    assert isinstance(tree, str)
    assert 'User' in tree
    assert 'userName' in tree


def test_report_contains_no_sensitive_values() -> None:
    """The report must not contain real sensitive values (no instance dumping)."""
    report = audit_model(User)
    tree = report.to_tree()
    json_str = report.to_json()
    # The report should only contain field metadata, not actual values.
    assert 'secret-str' in tree or 'default-protected' in tree
    # No actual secret values should appear (we never dumped an instance).
    assert '**********' not in tree
    assert '**********' not in json_str


# ---------------------------------------------------------------------------
# Path tests
# ---------------------------------------------------------------------------


def test_field_paths() -> None:
    class Inner(BaseModel):
        x: int

    class Outer(BaseModel):
        inner: Inner

    report = audit_model(Outer)
    inner_field = next(f for f in report.fields if f.name == 'inner')
    assert inner_field.path == ('inner',)
    x_field = inner_field.children[0]
    assert x_field.path == ('inner', 'x')


def test_nested_list_paths() -> None:
    class Item(BaseModel):
        v: int

    class Box(BaseModel):
        items: list[Item]

    report = audit_model(Box)
    items_field = next(f for f in report.fields if f.name == 'items')
    assert items_field.path == ('items',)
    assert len(items_field.children) == 1
    item_child = items_field.children[0]
    assert item_child.path == ('items', 'v')
