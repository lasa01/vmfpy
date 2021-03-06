import collections as _c
from typing import Any, Optional, Union, Dict, Mapping, List, Tuple, Iterator, Iterable, TypeVar, overload

T = TypeVar("T")
V = TypeVar("V")

class _kView(_c.KeysView[str]):
    def __iter__(self) -> Iterator[str]: ...

class _vView(_c.ValuesView[V]):
    def __iter__(self) -> Iterator[V]: ...

class _iView(_c.ItemsView[str, V]):
    def __iter__(self) -> Iterator[Tuple[str, V]]: ...

VDFKey = Union[str, Tuple[int, str]]
VDFData = Union[Dict[VDFKey, str], List[Tuple[VDFKey, str]]]
class VDFDict(Dict[VDFKey, V]):
    def _normalize_key(self, key: VDFKey) -> Tuple[int, str]: ...
    def __init__(self, data: Optional[VDFData] = ...) -> None: ...
    def __len__(self) -> int: ...
    def __setitem__(self, key: VDFKey, value: V) -> None: ...
    def __getitem__(self, key: VDFKey) -> V: ...
    def __delitem__(self, key: VDFKey) -> None: ...
    def __iter__(self) -> Iterator[str]: ...
    def __contains__(self, key: object) -> bool: ...
    def __eq__(self, other: Any) -> bool: ...
    def __ne__(self, other: Any) -> bool: ...
    def clear(self) -> None: ...
    def get(self, key: VDFKey, *args: Any) -> V: ...
    def setdefault(self, key: VDFKey, default: V = ...) -> V: ...
    @overload
    def pop(self, key: VDFKey) -> V: ...
    @overload
    def pop(self, key: VDFKey, default: Union[V, T] = ...) -> Union[V, T]: ...
    def popitem(self) -> Tuple[str, V]: ...
    @overload
    def update(self, __m: Mapping[VDFKey, V], **kwargs: V) -> None: ...
    @overload
    def update(self, __m: Iterable[Tuple[VDFKey, V]], **kwargs: V) -> None: ...
    @overload
    def update(self, **kwargs: V) -> None: ...
    def iterkeys(self) -> Iterator[str]: ...
    def keys(self) -> _kView: ...
    def itervalues(self) -> Iterator[Any]: ...
    def values(self) -> _vView[V]: ...
    def iteritems(self) -> Iterator[Tuple[str, V]]: ...
    def items(self) -> _iView[V]: ...
    def get_all_for(self, key: str) -> List[V]: ...
    def remove_all_for(self, key: str) -> None: ...
    def has_duplicates(self) -> bool: ...
