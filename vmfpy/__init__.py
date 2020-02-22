import vdf
import vpk
import os
from pathlib import PurePosixPath
import re
from io import BufferedIOBase, TextIOBase, TextIOWrapper
from typing import Mapping
from typing import List, Set, Dict, Callable, Iterator, Iterable, Tuple, Optional, NamedTuple, IO, Union, TypeVar, Any
from typing import cast


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


class VMFParseException(Exception):
    def __init__(self, msg: str, context: str = None):
        super().__init__(f"VMF parsing failed: {msg}")
        self.msg = msg
        self._stack: List[str] = []
        if context is not None:
            self._stack.append(context)

    def __str__(self) -> str:
        return f"VMF parsing failed: {' '.join(reversed(self._stack))} {self.msg}"

    def pushstack(self, msg: str, context: str = None) -> None:
        self._stack.append(msg)
        if context is not None:
            self._stack.append(context)


_RT = TypeVar("_RT")


class _VMFParser():
    def __init__(self) -> None:
        self._context: Optional[str] = None

    def _check_str(self, name: str, value: Union[str, dict], full_name: str = None) -> str:
        if full_name is None:
            full_name = name
        if not isinstance(value, str):
            raise VMFParseException(f"{name} is not a str", self._context)
        return value

    def _parse_str(self, name: str, vdict: dict, full_name: str = None) -> str:
        if full_name is None:
            full_name = name
        if name not in vdict:
            raise VMFParseException(f"{full_name} doesn't exist", self._context)
        value = vdict[name]
        return self._check_str(name, value, full_name)

    def _parse_int(self, name: str, value: str) -> int:
        try:
            return int(value)
        except ValueError:
            raise VMFParseException(f"{name} is not a valid int", self._context)

    def _parse_int_str(self, name: str, vdict: dict) -> int:
        value = self._parse_str(name, vdict)
        return self._parse_int(name, value)

    def _parse_int_list(self, name: str, value: str) -> List[int]:
        try:
            return [int(s) for s in value.split(" ") if s != ""]
        except ValueError:
            raise VMFParseException(f"{name} contains an invalid int", self._context)

    def _parse_int_list_str(self, name: str, vdict: dict) -> List[int]:
        value = self._parse_str(name, vdict)
        return self._parse_int_list(name, value)

    def _parse_float(self, name: str, value: str) -> float:
        try:
            return float(value)
        except ValueError:
            raise VMFParseException(f"{name} is not a valid float", self._context)
        except OverflowError:
            raise VMFParseException(f"{name} is out of range", self._context)

    def _parse_float_str(self, name: str, vdict: dict) -> float:
        value = self._parse_str(name, vdict)
        return self._parse_float(name, value)

    def _parse_bool(self, name: str, vdict: dict) -> bool:
        intv = self._parse_int_str(name, vdict)
        if intv not in (0, 1):
            raise VMFParseException(f"{name} is not a valid bool", self._context)
        return bool(intv)

    def _check_dict(self, name: str, value: Union[str, dict]) -> vdf.VDFDict:
        if not isinstance(value, vdf.VDFDict):
            raise VMFParseException(f"{name} is not a dict", self._context)
        return value

    def _parse_dict(self, name: str, vdict: dict) -> vdf.VDFDict:
        if name not in vdict:
            raise VMFParseException(f"{name} doesn't exist", self._context)
        value = vdict[name]
        return self._check_dict(name, value)

    def _iter_parse_matrix(self, name: str, vdict: dict) -> Iterator[Tuple[int, str, str]]:
        value = self._parse_dict(name, vdict)
        for row_name in value:
            full_name = f"{name} {row_name}"
            if row_name[:3] != "row":
                raise VMFParseException(f"{name} contains invalid key", self._context)
            try:
                row_idx = int(row_name[3:])
            except ValueError:
                raise VMFParseException(f"{name} contains invalid row index", self._context)
            row_value: str = value[row_name]
            if not isinstance(row_value, str):
                raise VMFParseException(f"{name} contains a non-str value", self._context)
            yield (row_idx, row_value, full_name)

    def _parse_custom(self, parser: Callable[..., _RT], name: str, *args: Any) -> _RT:
        try:
            return parser(*args)
        except VMFParseException as e:
            e.pushstack(name, self._context)
            raise

    def _parse_custom_str(self, parser: Callable[[str], _RT], name: str, vdict: dict) -> _RT:
        value = self._parse_str(name, vdict)
        return self._parse_custom(parser, name, value)


_VECTOR_REGEX = re.compile(r"^\[(-?\d*\.?\d*e?-?\d*) (-?\d*\.?\d*e?-?\d*) (-?\d*\.?\d*e?-?\d*)]$")


class VMFVector(NamedTuple):
    """An XYZ location (or rotation) given by 3 float values."""
    x: float
    y: float
    z: float

    @staticmethod
    def parse_str(data: str) -> 'VMFVector':
        nums = data.split(" ")
        assert len(nums) == 3
        if len(nums) != 3:
            raise VMFParseException("vector doesn't contain 3 values")
        try:
            return VMFVector(*(float(s) for s in nums))
        except ValueError:
            raise VMFParseException("vector contains an invalid float")
        except OverflowError:
            raise VMFParseException("vector float out of range")

    @staticmethod
    def parse_sq_brackets(data: str) -> 'VMFVector':
        match = _VECTOR_REGEX.match(data)
        if match is None:
            raise VMFParseException("vector syntax is invalid (expected square-bracketed)")
        try:
            return VMFVector(*(float(s) for s in match.groups()))
        except ValueError:
            raise VMFParseException("vector contains an invalid float")
        except OverflowError:
            raise VMFParseException("vector float out of range")

    @staticmethod
    def parse_tuple(data: Tuple[str, str, str]) -> 'VMFVector':
        try:
            return VMFVector(*(float(s) for s in data))
        except ValueError:
            raise VMFParseException("vector contains an invalid float")
        except OverflowError:
            raise VMFParseException("vector float out of range")


class VMFColor(NamedTuple):
    """A color value using 3 integers between 0 and 255."""
    r: int
    g: int
    b: int


class VMFEntity(_VMFParser):
    """An entity."""
    def __init__(self, data: vdf.VDFDict, fs: VMFFileSystem):
        super().__init__()
        self.data = data
        self.fs = fs
        self.id = self._parse_int_str("id", data)
        """This is a unique number among other entity ids ."""
        self._context = f"(id {self.id})"
        self.classname: str = self._parse_str("classname", data)
        """This is the name of the entity class."""

        self.origin: Optional[VMFVector] = None
        """This is the point where the point entity exists."""
        if "origin" in data:
            self.origin = self._parse_custom_str(VMFVector.parse_str, "origin", data)
        self.spawnflags: Optional[int] = None
        """Indicates which flags are enabled on the entity."""
        if "spawnflags" in data:
            self.spawnflags = self._parse_int_str("spawnflags", data)


class VMFPointEntity(VMFEntity):
    """A point based entity."""
    def __init__(self, data: vdf.VDFDict, fs: VMFFileSystem):
        super().__init__(data, fs)
        if self.origin is None:
            raise VMFParseException("doesn't have an origin", self._context)
        self.origin: VMFVector


class VMFPropEntity(VMFPointEntity):
    """Adds a model to the world."""
    def __init__(self, data: vdf.VDFDict, fs: VMFFileSystem):
        super().__init__(data, fs)
        self.angles: VMFVector
        """This entity's orientation in the world."""
        if "angles" in data:
            self.angles = self._parse_custom_str(VMFVector.parse_str, "angles", data)
        else:
            self.angles = VMFVector(0, 0, 0)
        self.model = self._parse_str("model", data)
        """The model this entity should appear as."""
        self.skin: int
        """Some models have multiple skins. This value selects from the index, starting with 0."""
        if "skin" in data:
            self.skin = self._parse_int_str("skin", data)
        else:
            self.skin = 0

    def open_model(self) -> AnyBinaryIO:
        return self.fs.open_file(self.model)


class VMFOverlayEntity(VMFPointEntity):
    """More powerful version of a material projected onto existing surfaces."""
    def __init__(self, data: vdf.VDFDict, fs: VMFFileSystem):
        super().__init__(data, fs)

        self.material = self._parse_str("material", data)
        """The material to overlay."""
        self.materialpath = "materials/" + self.material + ".vmt"

        self.sides = self._parse_int_list_str("sides", data)
        """Faces on which the overlay will be applied."""

        self.renderorder: Optional[int] = None
        """Higher values render after lower values (on top). This value can be 0–3."""
        if "RenderOrder" in data:
            self.renderorder = self._parse_int_str("RenderOrder", data)

        self.startu = self._parse_float_str("StartU", data)
        """Texture coordinates for the image."""
        self.startv = self._parse_float_str("StartV", data)
        """Texture coordinates for the image."""
        self.endu = self._parse_float_str("EndU", data)
        """Texture coordinates for the image."""
        self.endv = self._parse_float_str("EndV", data)
        """Texture coordinates for the image."""

        self.basisorigin: VMFVector = self._parse_custom_str(VMFVector.parse_str, "BasisOrigin", data)
        """Offset of the surface from the position of the overlay entity."""
        self.basisu: VMFVector = self._parse_custom_str(VMFVector.parse_str, "BasisU", data)
        """Direction of the material's X-axis."""
        self.basisv: VMFVector = self._parse_custom_str(VMFVector.parse_str, "BasisV", data)
        """Direction of the material's Y-axis."""
        self.basisnormal: VMFVector = self._parse_custom_str(VMFVector.parse_str, "BasisNormal", data)
        """Direction out of the surface."""

        self.uv0: VMFVector = self._parse_custom_str(VMFVector.parse_str, "uv0", data)
        self.uv1: VMFVector = self._parse_custom_str(VMFVector.parse_str, "uv1", data)
        self.uv2: VMFVector = self._parse_custom_str(VMFVector.parse_str, "uv2", data)
        self.uv3: VMFVector = self._parse_custom_str(VMFVector.parse_str, "uv3", data)

    def open_material_file(self) -> TextIOWrapper:
        return TextIOWrapper(cast(IO[bytes],
                             self.fs.open_file(self.materialpath)),
                             encoding='utf-8')


class VMFLightEntity(VMFPointEntity):
    """Creates an invisible, static light source that shines in all directions."""
    def __init__(self, data: vdf.VDFDict, fs: VMFFileSystem):
        super().__init__(data, fs)
        light_list = self._parse_int_list_str("_light", data)
        if len(light_list) != 4:
            raise VMFParseException("_light doesn't have 4 values", self._context)
        self.color = VMFColor(*light_list[:3])
        """The RGB color of the light."""
        self.brightness = light_list[3]
        """The brightness of the light."""

        light_hdr_list = self._parse_int_list_str("_lightHDR", data)
        if len(light_list) != 4:
            raise VMFParseException("_lightHDR doesn't have 4 values", self._context)
        self.hdr_color = VMFColor(*light_hdr_list[:3])
        """Color override used in HDR mode. Default is -1 -1 -1 which means no change."""
        self.hdr_brightness = light_hdr_list[3]
        """Brightness override used in HDR mode. Default is 1 which means no change."""
        self.hdr_scale = self._parse_float_str("_lightscaleHDR", data)
        """A simple intensity multiplier used when compiling HDR lighting."""

        self.style: Optional[int] = None
        """Various Custom Appearance presets."""
        if "style" in data:
            self.style = self._parse_int_str("style", data)
        self.constant_attn: Optional[float] = None
        """Determines how the intensity of the emitted light falls off over distance."""
        if "_constant_attn" in data:
            self.constant_attn = self._parse_float_str("_constant_attn", data)
        self.linear_attn: Optional[float] = None
        """Determines how the intensity of the emitted light falls off over distance."""
        if "_linear_attn" in data:
            self.linear_attn = self._parse_float_str("_linear_attn", data)
        self.quadratic_attn: Optional[float] = None
        """Determines how the intensity of the emitted light falls off over distance."""
        if "_quadratic_attn" in data:
            self.quadratic_attn = self._parse_float_str("_quadratic_attn", data)
        self.fifty_percent_distance: Optional[float] = None
        """Distance at which brightness should have fallen to 50%. Overrides attn if non-zero."""
        if "_fifty_percent_distance" in data:
            self.fifty_percent_distance = self._parse_float_str("_fifty_percent_distance", data)
        self.zero_percent_distance: Optional[float] = None
        """Distance at which brightness should have fallen to (1/256)%. Overrides attn if non-zero."""
        if "_zero_percent_distance" in data:
            self.zero_percent_distance = self._parse_float_str("_zero_percent_distance", data)


class VMFSpotLightEntity(VMFLightEntity):
    """A cone-shaped, invisible light source."""
    def __init__(self, data: vdf.VDFDict, fs: VMFFileSystem):
        super().__init__(data, fs)
        self.angles: VMFVector
        """This entity's orientation in the world."""
        if "angles" in data:
            self.angles = self._parse_custom_str(VMFVector.parse_str, "angles", data)
        else:
            self.angles = VMFVector(0, 0, 0)
        self.pitch = self._parse_float_str("pitch", data)
        """Used instead of angles value for reasons unknown."""

        self.inner_cone = self._parse_int_str("_inner_cone", data)
        """The angle of the inner spotlight beam."""
        self.cone = self._parse_int_str("_cone", data)
        """The angle of the outer spotlight beam."""
        self.exponent = self._parse_int_str("_exponent", data)
        """Changes the distance between the umbra and penumbra cone."""


class VMFEnvLightEntity(VMFLightEntity):
    """Casts parallel directional lighting and diffuse skylight from the toolsskybox texture."""
    def __init__(self, data: vdf.VDFDict, fs: VMFFileSystem):
        super().__init__(data, fs)
        self.angles: VMFVector
        """This entity's orientation in the world."""
        if "angles" in data:
            self.angles = self._parse_custom_str(VMFVector.parse_str, "angles", data)
        else:
            self.angles = VMFVector(0, 0, 0)
        self.pitch = self._parse_float_str("pitch", data)
        """Used instead of angles value for reasons unknown."""

        amb_light_list = self._parse_int_list_str("_ambient", data)
        if len(amb_light_list) != 4:
            raise VMFParseException("_ambient doesn't have 4 values", self._context)
        self.amb_color = VMFColor(*amb_light_list[:3])
        """Color of the diffuse skylight."""
        self.amb_brightness = amb_light_list[3]
        """Brightness of the diffuse skylight."""

        amb_light_hdr_list = self._parse_int_list_str("_ambientHDR", data)
        if len(amb_light_hdr_list) != 4:
            raise VMFParseException("_ambientHDR doesn't have 4 values", self._context)
        self.amb_hdr_color = VMFColor(*amb_light_hdr_list[:3])
        """Override for ambient color when compiling HDR lighting."""
        self.amb_hdr_brightness = amb_light_hdr_list[3]
        """Override for ambient brightness when compiling HDR lighting."""
        self.amb_hdr_scale = self._parse_float_str("_AmbientScaleHDR", data)
        """Amount to scale the ambient light by when compiling for HDR."""

        self.sun_spread_angle = self._parse_float_str("SunSpreadAngle", data)
        """The angular extent of the sun for casting soft shadows."""


_PLANE_REGEX = re.compile(r"^\((-?\d*\.?\d*e?-?\d*) (-?\d*\.?\d*e?-?\d*) (-?\d*\.?\d*e?-?\d*)\) "
                          r"\((-?\d*\.?\d*e?-?\d*) (-?\d*\.?\d*e?-?\d*) (-?\d*\.?\d*e?-?\d*)\) "
                          r"\((-?\d*\.?\d*e?-?\d*) (-?\d*\.?\d*e?-?\d*) (-?\d*\.?\d*e?-?\d*)\)$")


class VMFPlane(NamedTuple):
    """"A fundamental two-dimensional object defined by three points."""
    btm_l: VMFVector
    top_l: VMFVector
    top_r: VMFVector

    @staticmethod
    def parse(data: str) -> 'VMFPlane':
        match = _PLANE_REGEX.match(data)
        if match is None:
            raise VMFParseException("plane syntax is invalid")
        try:
            floats = [float(s) for s in match.groups()]
        except ValueError:
            raise VMFParseException("plane contains an invalid float")
        except OverflowError:
            raise VMFParseException("plane float out of range")
        return VMFPlane(VMFVector(*floats[:3]),
                        VMFVector(*floats[3:6]),
                        VMFVector(*floats[6:9]))


_AXIS_REGEX = re.compile(r"^\[(-?\d*\.?\d*e?-?\d*) (-?\d*\.?\d*e?-?\d*) (-?\d*\.?\d*e?-?\d*) (-?\d*\.?\d*e?-?\d*)] "
                         r"(-?\d*\.?\d*e?-?\d*)$")


class VMFAxis(NamedTuple):
    """Texture specific axis."""
    x: float
    y: float
    z: float
    trans: float
    scale: float

    @staticmethod
    def parse(data: str) -> 'VMFAxis':
        match = _AXIS_REGEX.match(data)
        if match is None:
            raise VMFParseException("axis syntax is invalid")
        try:
            floats = [float(s) for s in match.groups()]
        except ValueError:
            raise VMFParseException("axis contains an invalid float")
        except OverflowError:
            raise VMFParseException("axis float out of range")
        return VMFAxis(*floats)


class VMFDispInfo(_VMFParser):
    """Deals with all the information for a displacement map."""
    def __init__(self, data: vdf.VDFDict):
        super().__init__()
        self.power = self._parse_int_str("power", data)
        """Used to calculate the number of rows and columns."""
        self.triangle_dimension = 2 ** self.power
        """The number of rows and columns in triangles."""
        self.dimension = self.triangle_dimension + 1
        """The number of rows and columns in vertexes."""
        self.startposition: VMFVector = self._parse_custom_str(VMFVector.parse_sq_brackets, "startposition", data)
        """The position of the bottom left corner in an actual x y z position."""
        self.elevation = self._parse_float_str("elevation", data)
        """A universal displacement in the direction of the vertex's normal added to all of the points."""
        self.subdiv = self._parse_bool("subdiv", data)
        """Marks whether or not the displacement is being subdivided."""

        self.normals: List[List[VMFVector]] = [[VMFVector(0, 0, 0) for _ in range(self.dimension)]
                                               for _ in range(self.dimension)]
        """Defines the normal line for each vertex."""
        for row_idx, row_value, row_name in self._iter_parse_matrix("normals", data):
            row_nums_it = iter(row_value.split(" "))
            vec_tuple: Tuple[str, str, str]
            for idx, vec_tuple in enumerate(zip(row_nums_it, row_nums_it, row_nums_it)):
                self.normals[row_idx][idx] = self._parse_custom(VMFVector.parse_tuple, row_name, vec_tuple)

        self.distances: List[List[float]] = [[0 for _ in range(self.dimension)] for _ in range(self.dimension)]
        """The distance values represent how much the vertex is moved along the normal line."""
        for row_idx, row_value, row_name in self._iter_parse_matrix("distances", data):
            for idx, num_str in enumerate(row_value.split(" ")):
                self.distances[row_idx][idx] = self._parse_float(row_name, num_str)

        self.offsets: List[List[VMFVector]] = [[VMFVector(0, 0, 0) for _ in range(self.dimension)]
                                               for _ in range(self.dimension)]
        """Lists all the default positions for each vertex in a displacement map."""
        for row_idx, row_value, row_name in self._iter_parse_matrix("offsets", data):
            row_nums_it = iter(row_value.split(" "))
            for idx, vec_tuple in enumerate(zip(row_nums_it, row_nums_it, row_nums_it)):
                self.offsets[row_idx][idx] = self._parse_custom(VMFVector.parse_tuple, row_name, vec_tuple)

        self.offset_normals: List[List[VMFVector]] = [[VMFVector(0, 0, 0) for _ in range(self.dimension)]
                                                      for _ in range(self.dimension)]
        """Defines the default normal lines that the normals are based from."""
        for row_idx, row_value, row_name in self._iter_parse_matrix("offset_normals", data):
            row_nums_it = iter(row_value.split(" "))
            for idx, vec_tuple in enumerate(zip(row_nums_it, row_nums_it, row_nums_it)):
                self.offset_normals[row_idx][idx] = self._parse_custom(VMFVector.parse_tuple, row_name, vec_tuple)

        self.alphas: List[List[float]] = [[0 for _ in range(self.dimension)] for _ in range(self.dimension)]
        """Contains a value for each vertex that represents how much of which texture to shown in blended materials."""
        for row_idx, row_value, row_name in self._iter_parse_matrix("alphas", data):
            for idx, num_str in enumerate(row_value.split(" ")):
                self.alphas[row_idx][idx] = self._parse_float(row_name, num_str)

        self.triangle_tags: List[List[Tuple[int, int]]] = [[(0, 0) for _ in range(self.triangle_dimension)]
                                                           for _ in range(self.triangle_dimension)]
        """Contains information specific to each triangle in the displacement."""
        for row_idx, row_value, row_name in self._iter_parse_matrix("triangle_tags", data):
            row_nums_it = iter(row_value.split(" "))
            for idx, tag_tuple in enumerate(zip(row_nums_it, row_nums_it)):
                self.triangle_tags[row_idx][idx] = (self._parse_int(row_name, tag_tuple[0]),
                                                    self._parse_int(row_name, tag_tuple[1]))

        allowed_verts_dict = self._parse_dict("allowed_verts", data)
        allowed_verts_value = self._parse_str("10", allowed_verts_dict, "allowed_verts 10")
        self.allowed_verts = tuple(self._parse_int_list("allowed_verts 10", allowed_verts_value))
        """This states which vertices share an edge with another displacement map, but not a vertex."""


class VMFSide(_VMFParser):
    """Defines all the data relevant to one side and just to that side."""
    def __init__(self, data: vdf.VDFDict, fs: VMFFileSystem):
        super().__init__()
        self.fs = fs
        """File system for opening game files."""
        self.id = self._parse_int_str("id", data)
        """A unique value among other sides ids."""
        self._context = f"(id {self.id})"
        self.plane: VMFPlane = self._parse_custom_str(VMFPlane.parse, "plane", data)
        """Defines the orientation of the face."""
        self.material = self._parse_str("material", data)
        """The directory and name of the texture the side has applied to it."""
        self.materialpath = "materials/" + self.material + ".vmt"
        self.uaxis: VMFAxis = self._parse_custom_str(VMFAxis.parse, "uaxis", data)
        """The u-axis and v-axis are the texture specific axes."""
        self.vaxis: VMFAxis = self._parse_custom_str(VMFAxis.parse, "vaxis", data)
        """The u-axis and v-axis are the texture specific axes."""
        self.rotation = self._parse_float_str("rotation", data)
        """The rotation of the given texture on the side."""
        self.lightmapscale = self._parse_int_str("lightmapscale", data)
        """The light map resolution on the face."""
        self.smoothing_groups = self._parse_int_str("smoothing_groups", data).to_bytes(4, 'little')
        """"Select a smoothing group to use for lighting on the face."""

        self.dispinfo: Optional[VMFDispInfo] = None
        """Deals with all the information for a displacement map."""
        if "dispinfo" in data:
            dispinfo_dict = self._parse_dict("dispinfo", data)
            self.dispinfo = self._parse_custom(VMFDispInfo, "dispinfo", dispinfo_dict)

    def open_material_file(self) -> TextIOWrapper:
        return TextIOWrapper(cast(IO[bytes],
                             self.fs.open_file(self.materialpath)),
                             encoding='utf-8')


class VMFSolid(_VMFParser):
    """Represents 1 single brush in Hammer."""
    def __init__(self, data: vdf.VDFDict, fs: VMFFileSystem):
        self.fs = fs
        """File system for opening game files."""
        self.id = self._parse_int_str("id", data)
        """A unique value among other solids' IDs."""
        self._context = f"(id {self.id})"
        dict_sides = data.get_all_for("side")
        self.sides: List[VMFSide] = list()
        for side in dict_sides:
            side = self._check_dict("side", side)
            self.sides.append(self._parse_custom(VMFSide, "side", side, self.fs))


class VMFBrushEntity(VMFEntity):
    """A brush based entity."""
    def __init__(self, data: vdf.VDFDict, fs: VMFFileSystem):
        super().__init__(data, fs)
        dict_solids = data.get_all_for("solid")
        self.solids: List[VMFSolid] = list()
        for solid in dict_solids:
            solid = self._check_dict("solid", solid)
            self.solids.append(self._parse_custom(VMFSolid, "solid", solid, self.fs))


class VMFWorldEntity(VMFBrushEntity):
    """Contains all the world brush information for Hammer."""
    def __init__(self, data: vdf.VDFDict, fs: VMFFileSystem):
        super().__init__(data, fs)
        if self.classname != "worldspawn":
            raise VMFParseException("classname is not worldspawn")
        self.skyname = self._parse_str("skyname", data)
        """The name of the skybox to be used."""
        self.skypath = "materials/skybox/" + self.skyname + ".vmt"

    def open_sky(self) -> TextIOWrapper:
        return TextIOWrapper(cast(IO[bytes],
                             self.fs.open_file(self.skypath)),
                             encoding='utf-8')


class VMF(_VMFParser):
    def __init__(self, file: AnyTextIO, fs: VMFFileSystem = VMFFileSystem()):
        super().__init__()
        self.fs = fs
        """File system for opening game files."""
        vdf_dict = vdf.load(file, mapper=vdf.VDFDict, merge_duplicate_keys=False, escaped=False)

        versioninfo = self._parse_dict("versioninfo", vdf_dict)
        self.editorversion = self._parse_int_str("editorversion", versioninfo)
        """The version of Hammer used to create the file, version 4.00 is "400"."""
        self.editorbuild = self._parse_int_str("editorbuild", versioninfo)
        """The patch number of Hammer the file was generated with."""
        self.mapversion = self._parse_int_str("mapversion", versioninfo)
        """This represents how many times you've saved the file, useful for comparing old or new versions."""
        self.prefab = self._parse_bool("prefab", versioninfo)
        """Whether this is a full map or simply a collection of prefabricated objects."""

        world_dict = self._parse_dict("world", vdf_dict)
        self.world: VMFWorldEntity = self._parse_custom(VMFWorldEntity, "world", world_dict, self.fs)
        """"Contains all the world brush information for Hammer."""

        self.entities: List[VMFEntity] = list()
        """List of all entities in the map."""
        self.overlay_entities: List[VMFOverlayEntity] = list()
        """List of info_overlays in the map."""
        self.env_light_entity: Optional[VMFEnvLightEntity] = None
        """List of light_environments in the map."""
        self.spot_light_entities: List[VMFSpotLightEntity] = list()
        """List of light_spots in the map."""
        self.light_entities: List[VMFLightEntity] = list()
        """List of other lights in the map."""
        self.func_entities: List[VMFBrushEntity] = list()
        """List of func (brush) entities in the map."""
        self.prop_entities: List[VMFPropEntity] = list()
        """List of prop entities in the map."""

        dict_entities = vdf_dict.get_all_for("entity")
        for entity in dict_entities:
            entity = self._check_dict("entity", entity)
            classname = self._parse_str("classname", entity, "entity classname")
            entity_inst: VMFEntity
            if classname == "info_overlay":
                entity_inst = self._parse_custom(VMFOverlayEntity, "entity (info_overlay)", entity, self.fs)
                self.overlay_entities.append(entity_inst)
            elif classname == "light_environment":
                entity_inst = self._parse_custom(VMFEnvLightEntity, "entity (light_environment)", entity, self.fs)
                self.env_light_entity = entity_inst
            elif classname == "light_spot":
                entity_inst = self._parse_custom(VMFSpotLightEntity, "entity (light_spot)", entity, self.fs)
                self.spot_light_entities.append(entity_inst)
            elif classname.startswith("light"):
                entity_inst = self._parse_custom(VMFLightEntity, "entity (light)", entity, self.fs)
                self.light_entities.append(entity_inst)
            elif classname.startswith("func"):
                entity_inst = self._parse_custom(VMFBrushEntity, "entity (func)", entity, self.fs)
                self.func_entities.append(entity_inst)
            elif classname.startswith("prop"):
                entity_inst = self._parse_custom(VMFPropEntity, "entity (prop)", entity, self.fs)
                self.prop_entities.append(entity_inst)
            elif "origin" in entity:
                entity_inst = self._parse_custom(VMFPointEntity, "entity", entity, self.fs)
            else:
                entity_inst = self._parse_custom(VMFEntity, "entity", entity, self.fs)
            self.entities.append(entity_inst)
