"""Demo: Static selection interpreter for discriminated unions.

This script demonstrates the static discriminator interpreter, which determines
which branch of a discriminated union would be selected for a given input, without
actually validating the input.

Run with:
    python demo_static_discriminator.py
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Discriminator, Field, Tag, TypeAdapter
from pydantic._internal._static_discriminator import static_discriminate


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Cat(BaseModel):
    kind: Literal["cat"]
    meow: str


class Dog(BaseModel):
    kind: Literal["dog"]
    bark: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def show(title: str, result) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"  status      : {result.status}")
    if result.path is not None:
        print(f"  path        : {result.path}")
    if result.tag is not None:
        print(f"  tag         : {result.tag!r}")
    if result.branch_name is not None:
        print(f"  branch      : {result.branch_name}")
    if result.tried_paths:
        print(f"  tried_paths : {result.tried_paths}")
    print()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    # 1. Basic discriminated union: kind distinguishes Cat vs Dog
    adapter = TypeAdapter(Annotated[Cat | Dog, Field(discriminator="kind")])

    show("1. dict kind='cat' reports Cat branch", static_discriminate(adapter, {"kind": "cat", "meow": "purr"}))

    show("2. missing kind reports 'missing_tag'", static_discriminate(adapter, {"meow": "purr"}))

    show("3. kind='z' reports 'unknown_tag'", static_discriminate(adapter, {"kind": "z"}))

    # 4. Callable discriminator is NOT executed
    call_count = 0

    def my_discriminator(v):
        nonlocal call_count
        call_count += 1
        return v.get("kind")

    callable_adapter = TypeAdapter(
        Annotated[
            Annotated[Cat, Tag("cat")] | Annotated[Dog, Tag("dog")],
            Discriminator(my_discriminator),
        ]
    )
    result = static_discriminate(callable_adapter, {"kind": "cat"})
    show(f"4. callable discriminator -> 'unknown_discriminator' (call_count={call_count})", result)

    # 5. Nested path discriminator
    from pydantic_core import core_schema

    choices = adapter.core_schema["choices"]
    nested_adapter = TypeAdapter(Annotated[Cat | Dog, Field(discriminator="kind")])
    nested_adapter.core_schema = core_schema.tagged_union_schema(
        choices=choices,
        discriminator=["outer", "inner"],
    )
    show("5. nested path ['outer','inner'] reads tag", static_discriminate(nested_adapter, {"outer": {"inner": "dog", "bark": "woof"}}))

    # 6. Alias path with explicit None is NOT missing
    class AliasCat(BaseModel):
        kind: Literal["cat"] = Field(alias="type")
        meow: str

    class AliasDog(BaseModel):
        kind: Literal["dog"] = Field(alias="type")
        bark: str

    alias_adapter = TypeAdapter(Annotated[AliasCat | AliasDog, Field(discriminator="kind")])
    show("6. alias path with explicit None -> 'unknown_tag' (not missing)", static_discriminate(alias_adapter, {"type": None, "meow": "purr"}))

    # 7. Validators are NOT called
    from pydantic import field_validator

    validator_calls = 0

    class ValidatedCat(BaseModel):
        kind: Literal["cat"]
        meow: str

        @field_validator("meow")
        @classmethod
        def validate_meow(cls, v):
            nonlocal validator_calls
            validator_calls += 1
            return v

    validated_adapter = TypeAdapter(Annotated[ValidatedCat | Dog, Field(discriminator="kind")])
    result = static_discriminate(validated_adapter, {"kind": "cat", "meow": "purr"})
    show(f"7. validator zero-call (validator_calls={validator_calls})", result)

    # 8. Unsupported path (scalar input)
    show("8. scalar input -> 'unsupported_path'", static_discriminate(adapter, "scalar"))

    print("=" * 60)
    print("  Support scope summary")
    print("=" * 60)
    print("""
  Supported:
    - str discriminator (field name)
    - list[str | int] discriminator (single path)
    - list[list[str | int]] discriminator (multiple alias paths,
      resolved in schema order)
    - Explicit None values in paths (not treated as missing)
    - Nested paths (e.g. ['outer', 'inner'])
    - Index paths on lists (e.g. [0])

  Not supported:
    - Callable discriminators (reported as 'unknown_discriminator',
      NOT executed)
    - Paths with non-string/non-int elements
    - Scalar inputs (cannot resolve key/index paths)

  Guarantees:
    - Does NOT run field validators
    - Does NOT read object property getters
    - Does NOT execute callable discriminators
    - Does NOT validate the input
""")


if __name__ == "__main__":
    main()
