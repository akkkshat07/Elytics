from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ColumnMetadata:
    name: str
    data_type: str
    is_nullable: bool
    character_maximum_length: int | None = None
    numeric_precision: int | None = None
    numeric_scale: int | None = None
    description: str | None = None


@dataclass
class TableMetadata:
    schema: str
    name: str
    columns: List[ColumnMetadata] = field(default_factory=list)
    sample_rows: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def full_name(self) -> str:
        return f"{self.schema}.{self.name}"

