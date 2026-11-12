from collections.abc import Iterable, Iterator, KeysView
from typing import Any, Generic, NamedTuple, TypeVar, overload

from . import Cache

__all__ = (
    "Generation",
    "Hit",
    "SourceCache",
    "SourceCacheRecord",
)

_ST = TypeVar("_ST")
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")
_T = TypeVar("_T")

class Generation(NamedTuple, Generic[_ST]):
    source: _ST
    revision: int

class SourceCacheRecord(NamedTuple, Generic[_VT]):
    value: _VT
    empty: bool
    generations: tuple[Generation[Any], ...]

class Hit(NamedTuple, Generic[_KT, _VT]):
    key: _KT
    value: _VT
    empty: bool
    generations: tuple[Generation[Any], ...]

class SourceCache(Generic[_KT, _VT]):
    def __init__(
        self,
        maxsize: float = ...,
        *,
        cache_factory: type[Cache[Any, Any]] = ...,
    ) -> None: ...
    def seal(self, source: _ST) -> Generation[_ST]: ...
    def withdraw(self, source: _ST) -> tuple[_KT, ...]: ...
    def store_value(
        self,
        key: _KT,
        value: _VT,
        generation: Generation[Any] | None = ...,
        *,
        generations: Iterable[Generation[Any]] = (),
    ) -> None: ...
    def store_empty(
        self,
        key: _KT,
        generation: Generation[Any] | None = ...,
        *,
        generations: Iterable[Generation[Any]] = (),
    ) -> None: ...
    def get_hit(self, key: _KT) -> Hit[_KT, _VT] | None: ...
    @overload
    def get_value(self, key: _KT) -> _VT | None: ...
    @overload
    def get_value(self, key: _KT, default: _VT | _T) -> _VT | _T: ...
    def is_empty(self, key: _KT) -> bool: ...
    def sealed(self, source: _ST) -> Generation[_ST] | None: ...
    def generations(self, key: _KT) -> tuple[Generation[Any], ...]: ...
    def clear(self) -> None: ...
    def keys(self) -> KeysView[_KT]: ...
    @property
    def maxsize(self) -> float: ...
    def __contains__(self, key: object) -> bool: ...
    def __iter__(self) -> Iterator[_KT]: ...
    def __len__(self) -> int: ...
