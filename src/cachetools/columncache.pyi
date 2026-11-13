from collections.abc import Callable, Iterable, Mapping
from typing import Any, Final, Generic, TypeVar

__all__: Final = (
    "CachedGroup",
    "CachePlan",
    "ColumnCache",
    "ColumnSource",
    "ConflictingColumnsError",
    "plan_column_read",
)

_V = TypeVar("_V")
_Cost = float | Callable[[tuple[Any, ...]], float]

class ConflictingColumnsError(ValueError):
    column: Any
    checksums: tuple[Any, ...]
    group_ids: tuple[Any, ...]
    def __init__(
        self,
        column: Any,
        checksums: Iterable[Any],
        group_ids: Iterable[Any],
    ) -> None: ...

class CachedGroup(Generic[_V]):
    group_id: Any
    revision: Any
    columns: tuple[Any, ...]
    checksums: dict[Any, Any]
    unpack_cost: float
    payload: Mapping[Any, _V] | None
    def __init__(
        self,
        group_id: Any,
        revision: Any,
        checksums: Mapping[Any, Any],
        unpack_cost: float = ...,
        payload: Mapping[Any, _V] | None = ...,
    ) -> None: ...
    def __eq__(self, other: object) -> bool: ...

class CachePlan:
    groups: tuple[CachedGroup[Any], ...]
    remote_columns: tuple[Any, ...]
    revision: Any
    unpack_cost: float
    remote_cost: float
    full_remote: bool
    def __init__(
        self,
        groups: Iterable[CachedGroup[Any]],
        remote_columns: Iterable[Any],
        revision: Any,
        unpack_cost: float,
        remote_cost: float,
        full_remote: bool,
    ) -> None: ...
    @classmethod
    def full_remote_read(
        cls, columns: Iterable[Any], revision: Any, remote_cost: float
    ) -> CachePlan: ...
    @property
    def total_cost(self) -> float: ...
    @property
    def hit(self) -> bool: ...
    def __eq__(self, other: object) -> bool: ...

def plan_column_read(
    groups: Iterable[CachedGroup[Any]],
    columns: Iterable[Any],
    revision: Any,
    *,
    remote_column_cost: _Cost = ...,
    full_remote_cost: _Cost | None = ...,
) -> CachePlan: ...

class ColumnSource(Generic[_V]):
    def read(
        self, partition: Any, columns: Iterable[Any], revision: Any
    ) -> Mapping[Any, _V]: ...
    def unpack(self, group: CachedGroup[_V]) -> Mapping[Any, _V]: ...
    def supplemental_cost(self, partition: Any, columns: Iterable[Any]) -> float: ...
    def full_remote_cost(
        self, partition: Any, columns: Iterable[Any], revision: Any
    ) -> float: ...
    def checksums(
        self,
        partition: Any,
        columns: Iterable[Any],
        revision: Any,
        values: Mapping[Any, _V],
    ) -> dict[Any, Any] | None: ...

class ColumnCache(Generic[_V]):
    source: ColumnSource[_V]
    default_unpack_cost: float
    def __init__(
        self, source: ColumnSource[_V], *, default_unpack_cost: float = ...
    ) -> None: ...
    def add(
        self,
        partition: Any,
        revision: Any,
        checksums: Mapping[Any, Any],
        unpack_cost: float | None = ...,
        payload: Mapping[Any, _V] | None = ...,
        group_id: Any = ...,
    ) -> CachedGroup[_V]: ...
    def groups(self, partition: Any) -> tuple[CachedGroup[_V], ...]: ...
    def plan(
        self, partition: Any, columns: Iterable[Any], revision: Any
    ) -> CachePlan: ...
    def read(
        self, partition: Any, columns: Iterable[Any], revision: Any
    ) -> list[_V]: ...
    def read_with_plan(
        self, partition: Any, columns: Iterable[Any], revision: Any
    ) -> tuple[list[_V], CachePlan]: ...
