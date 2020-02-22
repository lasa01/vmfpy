from typing import Union, IO, Optional, List, NamedTuple, Mapping, Iterable, Iterator, Set, Dict, Callable, Tuple, cast
from pathlib import PurePosixPath
import os
from io import TextIOBase, BufferedIOBase, TextIOWrapper
import vpk

# workaround for https://github.com/python/typeshed/issues/1229
AnyTextIO = Union[TextIOBase, IO[str]]
AnyBinaryIO = Union[BufferedIOBase, IO[bytes]]


class VPKFileIOWrapper(BufferedIOBase):
    """An IO wrapper for a file inside a VPK archive."""
    def __init__(self, vpkf: vpk.VPKFile):
        self._vpkf = vpkf

    def save(self, path: str) -> None:
        self._vpkf.save(path)

    def verify(self) -> bool:
        return self._vpkf.verify()

    # BufferedIOBase implementation
    def close(self) -> None:
        super().close()
        self._vpkf.close()

    def read(self, size: Optional[int] = -1) -> bytes:
        if size is None:
            size = -1
        return self._vpkf.read(size)

    def read1(self, size: Optional[int] = -1) -> bytes:
        return self.read(size)

    def readable(self) -> bool:
        return True

    def readline(self, size: Optional[int] = -1) -> bytes:
        if size != -1:
            raise NotImplementedError()
        return self._vpkf.readline()

    def readlines(self, hint: Optional[int] = -1) -> List[bytes]:
        if hint != -1:
            raise NotImplementedError()
        return self._vpkf.readlines()

    def seek(self, offset: int, whence: int = 0) -> int:
        self._vpkf.seek(offset, whence)
        return self._vpkf.tell()

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._vpkf.tell()


def vmf_path(path: str) -> PurePosixPath:
    return PurePosixPath(path.replace("\\", "/").lower())


class DirContents(NamedTuple):
    dirs: Set[str]
    files: Set[str]


class VMFFileSystem(Mapping[PurePosixPath, AnyBinaryIO]):
    """File system for opening game files."""
    def __init__(self, dirs: Iterable[str] = None, paks: Iterable[str] = None, index_files: bool = False) -> None:
        self._dirs: Set[str] = set() if dirs is None else set(dirs)
        self._paks: Set[str] = set() if paks is None else set(paks)
        self._index: Dict[PurePosixPath, Callable[[], AnyBinaryIO]] = dict()
        self.tree: Dict[PurePosixPath, DirContents] = dict()
        if index_files:
            self.index_all()

    def add_dir(self, path: str) -> None:
        self._dirs.add(path)

    def remove_dir(self, path: str) -> None:
        self._dirs.remove(path)

    def add_pak(self, path: str) -> None:
        self._paks.add(path)

    def remove_pak(self, path: str) -> None:
        self._paks.remove(path)

    @staticmethod
    def _create_f_opener(p: str) -> Callable[[], AnyBinaryIO]:
        return lambda: open(p, 'rb')

    @staticmethod
    def _create_pf_opener(p: str, m: Tuple[bytes, int, int, int, int, int], v: vpk.VPK) -> Callable[[], AnyBinaryIO]:
        return lambda: VPKFileIOWrapper(v.get_vpkfile_instance(p, m))

    def iter_dir(self, directory: str) -> Iterator[Tuple[PurePosixPath, Callable[[], AnyBinaryIO]]]:
        root: str
        files: List[str]
        for root, _, files in os.walk(directory):
            root_path = vmf_path(root)
            for file_name in files:
                path = root_path.relative_to(vmf_path(directory)) / file_name.lower()
                yield (path, self._create_f_opener(os.path.join(root, file_name)))

    def iter_dirs(self) -> Iterator[Tuple[PurePosixPath, Callable[[], AnyBinaryIO]]]:
        for directory in self._dirs:
            yield from self.iter_dir(directory)

    def iter_pak(self, pak_file: str) -> Iterator[Tuple[PurePosixPath, Callable[[], AnyBinaryIO]]]:
        pak = vpk.open(pak_file)
        for pak_path, metadata in pak.read_index_iter():
            path = vmf_path(pak_path)
            yield (path, self._create_pf_opener(pak_path, metadata, pak))

    def iter_paks(self) -> Iterator[Tuple[PurePosixPath, Callable[[], AnyBinaryIO]]]:
        for pak_file in self._paks:
            yield from self.iter_pak(pak_file)

    def iter_all(self) -> Iterator[Tuple[PurePosixPath, Callable[[], AnyBinaryIO]]]:
        yield from self.iter_dirs()
        yield from self.iter_paks()

    def _do_index(self, index_iter: Iterator[Tuple[PurePosixPath, Callable[[], AnyBinaryIO]]]) -> None:
        for path, open_func in index_iter:
            self._index[path] = open_func
            directory = path.parent
            if directory not in self.tree:
                self.tree[directory] = DirContents(set(), set())
            self.tree[directory].files.add(path.name)
            last_parent = directory
            for parent in directory.parents:
                if parent not in self.tree:
                    self.tree[parent] = DirContents(set(), set())
                self.tree[parent].dirs.add(last_parent.name)
                last_parent = parent

    def index_dir(self, directory: str) -> None:
        self._do_index(self.iter_dir(directory))

    def index_dirs(self) -> None:
        self._do_index(self.iter_dirs())

    def index_pak(self, pak_file: str) -> None:
        self._do_index(self.iter_pak(pak_file))

    def index_paks(self) -> None:
        self._do_index(self.iter_paks())

    def index_all(self) -> None:
        self._do_index(self.iter_all())

    def clear_index(self) -> None:
        self._index.clear()
        self.tree.clear()

    def open_file(self, path: Union[str, PurePosixPath]) -> AnyBinaryIO:
        if isinstance(path, str):
            path = vmf_path(path)
        if path not in self._index:
            raise FileNotFoundError(path)
        return self._index[path]()

    def open_file_utf8(self, path: Union[str, PurePosixPath]) -> TextIOWrapper:
        return TextIOWrapper(cast(IO[bytes], self.open_file(path)), encoding='utf-8')

    def __getitem__(self, key: Union[str, PurePosixPath]) -> AnyBinaryIO:
        if isinstance(key, str):
            key = vmf_path(key)
        return self._index[key]()

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self) -> Iterator[PurePosixPath]:
        return iter(self._index)

    def __contains__(self, item: object) -> bool:
        return item in self._index
