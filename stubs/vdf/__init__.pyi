from typing import Any, AnyStr, Callable, TypeVar, Type, Union, IO
from io import TextIOBase
from vdf.vdict import VDFDict as VDFDict

string_type = str
int_type = int
BOMS: str

# waiting for https://github.com/python/mypy/issues/731
T = TypeVar("T", bound=dict)
# workaround for https://github.com/python/typeshed/issues/1229
AnyTextIO = Union[TextIOBase, IO[str]]

def strip_bom(line: AnyStr) -> AnyStr: ...
def parse(fp: AnyTextIO, mapper: Type[T] = ..., merge_duplicate_keys: bool = ..., escaped: bool = ...) -> T: ...
def loads(s: str, mapper: Type[T] = ..., merge_duplicate_keys: bool = ..., escaped: bool = ...) -> T: ...
def load(fp: AnyTextIO, mapper: Type[T] = ..., merge_duplicate_keys: bool = ..., escaped: bool = ...) -> T: ...
def dumps(obj: dict, pretty: bool = ..., escaped: bool = ...) -> str: ...
def dump(obj: dict, fp: AnyTextIO, pretty: bool = ..., escaped: bool = ...) -> None: ...

class BASE_INT(int_type): ...
class UINT_64(BASE_INT): ...
class INT_64(BASE_INT): ...
class POINTER(BASE_INT): ...
class COLOR(BASE_INT): ...

BIN_NONE: bytes
BIN_STRING: bytes
BIN_INT32: bytes
BIN_FLOAT32: bytes
BIN_POINTER: bytes
BIN_WIDESTRING: bytes
BIN_COLOR: bytes
BIN_UINT64: bytes
BIN_END: bytes
BIN_INT64: bytes
BIN_END_ALT: bytes

def binary_loads(s: str, mapper: Type[T] = ..., merge_duplicate_keys: bool = ..., alt_format: bool = ...) -> T: ...
def binary_dumps(obj: dict, alt_format: bool = ...) -> bytes: ...
def vbkv_loads(s: str, mapper: Type[T] = ..., merge_duplicate_keys: bool = ...) -> T: ...
def vbkv_dumps(obj: Any) -> bytes: ...
