"""Public type markers for the TileLang DSL v1 surface."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


@dataclass(frozen=True)
class ScalarType:
    name: str

    def __repr__(self) -> str:
        return self.name


class TensorView:
    """Bare TensorView annotation marker for TileLang DSL v1."""


class Tile:
    """Bare Tile annotation marker for TileLang DSL v1."""


@dataclass(frozen=True)
class WildcardType:
    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class TypeVariable:
    name: str

    def __repr__(self) -> str:
        return f"TypeVar({self.name!r})"


class MemorySpace(str, Enum):
    GM = "gm"
    UB = "ub"


class Pipe(str, Enum):
    MTE1 = "PIPE_MTE1"
    MTE2 = "PIPE_MTE2"
    V = "PIPE_V"
    MTE3 = "PIPE_MTE3"
    ALL = "PIPE_ALL"


class Event(str, Enum):
    ID0 = "EVENT_ID0"
    ID1 = "EVENT_ID1"
    ID2 = "EVENT_ID2"
    ID3 = "EVENT_ID3"
    ID4 = "EVENT_ID4"
    ID5 = "EVENT_ID5"
    ID6 = "EVENT_ID6"
    ID7 = "EVENT_ID7"


class MaskPattern(str, Enum):
    ALL = "PAT_ALL"
    ALLF = "PAT_ALLF"
    EVEN = "PAT_EVEN"
    ODD = "PAT_ODD"
    VL16 = "PAT_VL16"
    VL32 = "PAT_VL32"


@dataclass(frozen=True)
class TileConfig:
    fields: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "TileConfig":
        return cls(tuple(sorted(mapping.items())))


@dataclass(frozen=True)
class TileSpecialization:
    shape: tuple[int, ...]
    memory_space: MemorySpace
    config: TileConfig | None = None


i8 = ScalarType("i8")
i1 = ScalarType("i1")
i16 = ScalarType("i16")
i32 = ScalarType("i32")
i64 = ScalarType("i64")
f16 = ScalarType("f16")
bf16 = ScalarType("bf16")
f32 = ScalarType("f32")
PIPE = Pipe
EVENT = Event
PAT = MaskPattern
AnyFloat = WildcardType("AnyFloat")
AnyInt = WildcardType("AnyInt")
AnyType = WildcardType("AnyType")
AnyMask = WildcardType("AnyMask")


def TypeVar(name: str) -> TypeVariable:
    if not isinstance(name, str) or not name:
        raise TypeError("TypeVar name must be a non-empty string")
    return TypeVariable(name)


__all__ = [
    "ScalarType",
    "WildcardType",
    "TypeVariable",
    "TypeVar",
    "TensorView",
    "Tile",
    "MemorySpace",
    "Pipe",
    "Event",
    "PIPE",
    "EVENT",
    "MaskPattern",
    "PAT",
    "TileConfig",
    "TileSpecialization",
    "i1",
    "i8",
    "i16",
    "i32",
    "i64",
    "f16",
    "bf16",
    "f32",
    "AnyFloat",
    "AnyInt",
    "AnyType",
    "AnyMask",
]
