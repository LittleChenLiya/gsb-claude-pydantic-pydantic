"""Tests for the static selection interpreter."""
from __future__ import annotations
from typing import Annotated, Literal
import pytest
from pydantic import BaseModel, Discriminator, Field, Tag, TypeAdapter, field_validator
from pydantic._internal._static_discriminator import static_discriminate
from pydantic_core import core_schema

class Cat(BaseModel):
    kind: Literal["cat"]
    meow: str

class Dog(BaseModel):
    kind: Literal["dog"]
    bark: str

@pytest.fixture
def cat_dog_adapter():
    return TypeAdapter(Annotated[Cat | Dog, Field(discriminator="kind")])

def test_kind_cat(cat_dog_adapter):
    result = static_discriminate(cat_dog_adapter, {"kind": "cat", "meow": "purr"})
    assert result.status == "matched"
    assert result.tag == "cat"
    assert result.branch_name == "Cat"

def test_kind_dog(cat_dog_adapter):
    result = static_discriminate(cat_dog_adapter, {"kind": "dog", "bark": "woof"})
    assert result.status == "matched"
    assert result.tag == "dog"
    assert result.branch_name == "Dog"

def test_missing_kind(cat_dog_adapter):
    result = static_discriminate(cat_dog_adapter, {"meow": "purr"})
    assert result.status == "missing_tag"

def test_unknown_kind(cat_dog_adapter):
    result = static_discriminate(cat_dog_adapter, {"kind": "z"})
    assert result.status == "unknown_tag"
    assert result.tag == "z"

def test_callable_count_zero():
    count = 0
    def disc(v):
        nonlocal count
        count += 1
        return v.get("kind")
    adapter = TypeAdapter(Annotated[Annotated[Cat, Tag("cat")] | Annotated[Dog, Tag("dog")], Discriminator(disc)])
    result = static_discriminate(adapter, {"kind": "cat"})
    assert result.status == "unknown_discriminator"
    assert count == 0

def test_alias_none_not_missing():
    class ACat(BaseModel):
        kind: Literal["cat"] = Field(alias="type")
        meow: str
    class ADog(BaseModel):
        kind: Literal["dog"] = Field(alias="type")
        bark: str
    adapter = TypeAdapter(Annotated[ACat | ADog, Field(discriminator="kind")])
    result = static_discriminate(adapter, {"type": None, "meow": "purr"})
    assert result.status == "unknown_tag"
    assert result.tag is None

def test_alias_order():
    class ACat(BaseModel):
        kind: Literal["cat"] = Field(alias="type")
        meow: str
    class ADog(BaseModel):
        kind: Literal["dog"] = Field(alias="type")
        bark: str
    adapter = TypeAdapter(Annotated[ACat | ADog, Field(discriminator="kind")])
    result = static_discriminate(adapter, {"kind": "cat", "type": "dog"})
    assert result.status == "matched"
    assert result.path == ["kind"]
    result = static_discriminate(adapter, {"type": "dog"})
    assert result.status == "matched"
    assert result.path == ["type"]

def test_nested_path():
    adapter = TypeAdapter(Annotated[Cat | Dog, Field(discriminator="kind")])
    choices = adapter.core_schema["choices"]
    adapter.core_schema = core_schema.tagged_union_schema(choices=choices, discriminator=["outer", "inner"])
    result = static_discriminate(adapter, {"outer": {"inner": "cat"}})
    assert result.status == "matched"
    assert result.path == ["outer", "inner"]

def test_nested_missing():
    adapter = TypeAdapter(Annotated[Cat | Dog, Field(discriminator="kind")])
    choices = adapter.core_schema["choices"]
    adapter.core_schema = core_schema.tagged_union_schema(choices=choices, discriminator=["outer", "inner"])
    result = static_discriminate(adapter, {"outer": {"meow": "purr"}})
    assert result.status == "missing_tag"

def test_nested_unknown():
    adapter = TypeAdapter(Annotated[Cat | Dog, Field(discriminator="kind")])
    choices = adapter.core_schema["choices"]
    adapter.core_schema = core_schema.tagged_union_schema(choices=choices, discriminator=["outer", "inner"])
    result = static_discriminate(adapter, {"outer": {"inner": "z"}})
    assert result.status == "unknown_tag"
    assert result.tag == "z"

def test_validator_zero():
    calls = 0
    class VCat(BaseModel):
        kind: Literal["cat"]
        meow: str
        @field_validator("meow")
        @classmethod
        def val(cls, v):
            nonlocal calls
            calls += 1
            return v
    adapter = TypeAdapter(Annotated[VCat | Dog, Field(discriminator="kind")])
    result = static_discriminate(adapter, {"kind": "cat", "meow": "purr"})
    assert result.status == "matched"
    assert calls == 0

def test_unsupported_scalar():
    adapter = TypeAdapter(Annotated[Cat | Dog, Field(discriminator="kind")])
    result = static_discriminate(adapter, "scalar")
    assert result.status == "unsupported_path"

def test_unsupported_list_str_key():
    adapter = TypeAdapter(Annotated[Cat | Dog, Field(discriminator="kind")])
    result = static_discriminate(adapter, ["cat", 1])
    assert result.status == "unsupported_path"

def test_list_index_path():
    adapter = TypeAdapter(Annotated[Cat | Dog, Field(discriminator="kind")])
    choices = adapter.core_schema["choices"]
    adapter.core_schema = core_schema.tagged_union_schema(choices=choices, discriminator=[0])
    result = static_discriminate(adapter, ["cat", 1])
    assert result.status == "matched"
    assert result.path == [0]

def test_no_union():
    adapter = TypeAdapter(Cat)
    result = static_discriminate(adapter, {"kind": "cat"})
    assert result.status == "unsupported_path"
