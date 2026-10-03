"""Static report demo for the read-only serialization exposure auditor.

This script demonstrates the auditor's output on a realistic sample model.
It does *not* dump real instances -- the report is based entirely on the
model's fields and compiled core schema. No real sensitive values appear in
the output.

Run:
    python examples/serialization_audit_demo.py
"""

# ruff: noqa: T201, D100, D101, D102, D103

from __future__ import annotations

from pydantic_core import core_schema

from pydantic import BaseModel, Field, SecretStr, computed_field
from pydantic.experimental.serialization_audit import audit_model

# ---------------------------------------------------------------------------
# Sample models
# ---------------------------------------------------------------------------


class Address(BaseModel):
    street: str
    city: str
    zip_code: str = Field(serialization_alias='zip')


class User(BaseModel):
    """A user model with various serialization scenarios."""

    name: str = Field(serialization_alias='userName')
    email: str
    password: SecretStr
    ssn: SecretStr
    age: int = Field(exclude=True)
    internal_id: int = Field(repr=False)
    address: Address
    previous_addresses: list[Address] = []
    nickname: str | None = None

    @computed_field
    @property
    def display_name(self) -> str:
        return self.name

    @computed_field
    @property
    def initials(self) -> str:
        return self.name[0]


class SafeExpandAdapter:
    """A safe audit adapter that expands known serializers.

    This adapter explicitly opts in to expanding:
    - PlainSerializer fields that declare a return_schema
    - Computed fields that declare a return type

    It does *not* execute user code -- it only declares the output schema.
    """

    def __init__(self) -> None:
        self.serializer_calls = 0
        self.computed_calls = 0

    def expand_serializer(self, schema, *, field_name, path):
        self.serializer_calls += 1
        if isinstance(schema, dict) and 'return_schema' in schema:
            return schema['return_schema']
        return None

    def expand_computed_field(self, property_name, *, path):
        self.computed_calls += 1
        # Computed fields return a string in this model.
        return core_schema.str_schema()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    print('=' * 70)
    print('READ-ONLY SERIALIZATION EXPOSURE AUDITOR -- STATIC REPORT DEMO')
    print('=' * 70)
    print()
    print('Model: User')
    print('Note: No real instances are dumped. No sensitive values appear.')
    print()

    # --- Audit without adapter (all custom serializers/computed unknown) ---
    print('-' * 70)
    print('1. AUDIT WITHOUT ADAPTER (custom serializers & computed = unknown)')
    print('-' * 70)
    report = audit_model(User)
    print(report.to_tree())
    print()
    print(f'adapter_calls: {report.adapter_calls}')
    print(f'recursive_refs: {report.recursive_refs}')
    print()

    # --- Audit with adapter (expands known serializers) ---
    print('-' * 70)
    print('2. AUDIT WITH SAFE ADAPTER (expands known serializers)')
    print('-' * 70)
    adapter = SafeExpandAdapter()
    report = audit_model(User, adapter=adapter)
    print(report.to_tree())
    print()
    print(f'adapter_calls: {report.adapter_calls}')
    print(f'  serializer_calls: {adapter.serializer_calls}')
    print(f'  computed_calls: {adapter.computed_calls}')
    print()

    # --- JSON report ---
    print('-' * 70)
    print('3. JSON REPORT (machine-readable)')
    print('-' * 70)
    print(report.to_json(indent=2))
    print()

    # --- Verify no sensitive values ---
    print('-' * 70)
    print('4. SENSITIVE VALUE CHECK')
    print('-' * 70)
    tree = report.to_tree()
    json_str = report.to_json()
    has_secret_marker = 'default-protected' in tree or 'secret-str' in tree
    has_real_secret = '**********' in tree or '**********' in json_str
    print(f'  Has default-protected marker: {has_secret_marker}')
    print(f'  Contains real secret value:  {has_real_secret}')
    print()
    print('=' * 70)
    print('DEMO COMPLETE')
    print('=' * 70)


if __name__ == '__main__':
    main()
