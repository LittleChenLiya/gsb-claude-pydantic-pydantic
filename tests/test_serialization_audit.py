"""Tests for the read-only serialization exposure auditor."""

from typing import Optional

import pytest

from pydantic import BaseModel, Field, SecretStr, computed_field, field_serializer, model_serializer
from pydantic.experimental.serialization_audit import (
    AdapterVerdict,
    Sensitivity,
    audit_model,
)


def entries_by_path(report):
    out = {}

    def walk(entries):
        for entry in entries:
            out[entry.path] = entry
            walk(entry.children)

    walk(report.entries)
    return out


def test_repr_false_field_remains_exportable():
    class M(BaseModel):
        visible: str
        hidden_from_repr: str = Field(repr=False)

    report = audit_model(M)
    entries = entries_by_path(report)

    entry = entries['hidden_from_repr']
    assert entry.repr_visible is False
    # repr=False must NOT be mistaken for a serialization exclusion:
    assert entry.excluded is False
    assert entry.sensitivity is Sensitivity.PLAIN
    assert entries['visible'].repr_visible is True


def test_exclude_true_is_marked_excluded():
    class M(BaseModel):
        kept: str
        dropped: str = Field(exclude=True)

    report = audit_model(M)
    entries = entries_by_path(report)

    assert entries['dropped'].excluded is True
    assert entries['kept'].excluded is False


def test_secret_str_reported_as_default_protection_not_unknown():
    class M(BaseModel):
        password: SecretStr
        maybe_password: SecretStr | None = None

    report = audit_model(M)
    entries = entries_by_path(report)

    for path in ('password', 'maybe_password'):
        entry = entries[path]
        assert entry.sensitivity is Sensitivity.SECRET
        assert entry.custom_serializer is False
        assert 'masked by default' in entry.note


def test_custom_serializer_is_unknown_and_never_executed():
    calls = []

    class M(BaseModel):
        secret: SecretStr
        token: str

        @field_serializer('token')
        def serialize_token(self, value):
            calls.append(value)
            return value

    report = audit_model(M)
    entries = entries_by_path(report)

    # user serializer was not executed:
    assert calls == []
    # custom serializer is reported separately from the SecretStr default protection:
    assert entries['token'].sensitivity is Sensitivity.UNKNOWN
    assert entries['token'].custom_serializer is True
    assert entries['secret'].sensitivity is Sensitivity.SECRET
    assert entries['secret'].custom_serializer is False


def test_computed_field_unknown_without_adapter_and_never_executed():
    calls = []

    class M(BaseModel):
        x: int

        @computed_field
        @property
        def double(self) -> int:
            calls.append('double')
            return 2

    report = audit_model(M)
    entries = entries_by_path(report)

    assert calls == []
    entry = entries['double']
    assert entry.computed is True
    assert entry.sensitivity is Sensitivity.UNKNOWN


def test_adapter_expands_custom_serializer_and_computed_field():
    class Public(BaseModel):
        shown: str

    class M(BaseModel):
        token: str

        @field_serializer('token')
        def serialize_token(self, value):
            raise AssertionError('must not be executed')

        @computed_field
        @property
        def profile(self) -> Public:
            raise AssertionError('must not be executed')

    class TokenAdapter:
        name = 'token-adapter'

        def audit(self, context):
            if context.kind == 'field_serializer' and context.field_name == 'token':
                return AdapterVerdict(annotation=str, note='declared: hashes the token')
            return None

    class ProfileAdapter:
        name = 'profile-adapter'

        def audit(self, context):
            if context.kind == 'computed_field' and context.field_name == 'profile':
                return AdapterVerdict(annotation=Public)
            return None

    report = audit_model(M, adapters=[TokenAdapter(), ProfileAdapter()])
    entries = entries_by_path(report)

    token = entries['token']
    assert token.sensitivity is Sensitivity.ADAPTER_DECLARED
    assert token.adapter == 'token-adapter'
    assert token.note == 'declared: hashes the token'

    profile = entries['profile']
    assert profile.sensitivity is Sensitivity.ADAPTER_DECLARED
    assert profile.adapter == 'profile-adapter'
    # adapter-declared return type is expanded statically:
    assert entries['profile.shown'].export_key == 'shown'

    # callback counts: token consulted both adapters (first matched),
    # profile consulted token-adapter (declined) then profile-adapter (matched):
    assert report.callback_counts == {'token-adapter': 2, 'profile-adapter': 1}
    assert report.adapter_matches == {'token-adapter': 1, 'profile-adapter': 1}


def test_serialization_alias_is_reported():
    class M(BaseModel):
        name: str = Field(serialization_alias='userName')
        other: str = Field(alias='aliased')

    report = audit_model(M)
    entries = entries_by_path(report)

    assert entries['name'].serialization_alias == 'userName'
    assert entries['name'].export_key == 'userName'
    assert entries['other'].export_key == 'aliased'


def test_inheritance_source_is_reported():
    class GrandParent(BaseModel):
        inherited: str

    class Parent(GrandParent):
        overridden: str = 'a'

    class Child(Parent):
        own: str
        overridden: str = 'b'

    report = audit_model(Child)
    entries = entries_by_path(report)

    assert entries['inherited'].defined_in.endswith('GrandParent')
    assert entries['own'].defined_in.endswith('Child')
    assert entries['overridden'].defined_in.endswith('Child')

    parent_report = audit_model(Parent)
    parent_entries = entries_by_path(parent_report)
    assert parent_entries['overridden'].defined_in.endswith('Parent')


def test_recursive_model_stops_at_reference():
    class Node(BaseModel):
        value: int
        child: Optional['Node'] = None

    Node.model_rebuild()
    report = audit_model(Node)
    entries = entries_by_path(report)

    # recursion stops at the reference instead of expanding forever:
    ref = entries['child.<Node>']
    assert ref.recursive_ref is True
    assert 'recursion stopped' in ref.note


def test_same_nested_model_on_two_paths_both_expanded():
    class Leaf(BaseModel):
        label: str

    class Branch(BaseModel):
        left: Leaf
        right: Leaf

    report = audit_model(Branch)
    entries = entries_by_path(report)

    # no global visited set: both paths keep their expansion:
    assert entries['left.label'].path == 'left.label'
    assert entries['right.label'].path == 'right.label'


def test_mutually_recursive_models_terminate():
    class A(BaseModel):
        b: Optional['B'] = None

    class B(BaseModel):
        a: A | None = None

    A.model_rebuild()
    report = audit_model(A)
    entries = entries_by_path(report)

    assert entries['b.a'].recursive_ref is False
    assert entries['b.a.<A>'].recursive_ref is True


def test_model_serializer_flagged_and_adapter_counted():
    class M(BaseModel):
        x: int

        @model_serializer
        def serialize(self):
            raise AssertionError('must not be executed')

    report = audit_model(M)
    assert report.custom_model_serializer is True
    assert any('model_serializer' in note for note in report.notes)

    class WholeDumpAdapter:
        name = 'whole-dump'

        def audit(self, context):
            if context.kind == 'model_serializer':
                return AdapterVerdict(note='declared: exports only public keys')
            return None

    adapted = audit_model(M, adapters=[WholeDumpAdapter()])
    assert adapted.callback_counts == {'whole-dump': 1}
    assert adapted.adapter_matches == {'whole-dump': 1}
    assert any('whole-dump' in note for note in adapted.notes)


def test_report_contains_metadata_only_no_values():
    secret_value = 's3cr3t-value-that-must-not-leak'

    class M(BaseModel):
        password: SecretStr
        name: str = Field(serialization_alias='userName')

    report = audit_model(M)
    rendered = report.render_text()
    as_dict = report.to_dict()

    assert secret_value not in rendered
    assert secret_value not in str(as_dict)
    assert 'userName' in rendered
    assert 'sensitivity=secret' in rendered
    assert 'no user serializer or computed field executed' in rendered
    assert as_dict['entries'][0]['field_name'] == 'password'
    assert as_dict['entries'][0]['sensitivity'] == 'secret'


def test_audit_model_rejects_non_models():
    with pytest.raises(TypeError):
        audit_model(int)
