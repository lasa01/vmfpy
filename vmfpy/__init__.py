import vdf
import vpk
import os
import re
from io import BufferedIOBase, TextIOBase, TextIOWrapper
from typing import List, Dict, Callable, Iterator, Iterable, Tuple, Optional, NamedTuple, IO, Union, cast


# workaround for https://github.com/python/typeshed/issues/1229
AnyTextIO = Union[TextIOBase, IO[str]]
AnyBinaryIO = Union[BufferedIOBase, IO[bytes]]


class VPKFileIOWrapper(BufferedIOBase):
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


class VMFFileSystem():
    def __init__(self) -> None:
        self._dirs: List[str] = list()
        self._paks: List[str] = list()
        self._tree: Dict[str, Callable[[], AnyBinaryIO]] = dict()

    def add_dir(self, path: str) -> None:
        self._dirs.append(path)

    def remove_dir(self, path: str) -> None:
        self._dirs.remove(path)

    def add_pak(self, path: str) -> None:
        self._paks.append(path)

    def remove_pak(self, path: str) -> None:
        self._paks.remove(path)

    def index_files_iter(self) -> Iterator[Tuple[str, Callable[[], AnyBinaryIO]]]:
        for directory in self._dirs:
            root: str
            files: List[str]
            for root, _, files in os.walk(directory):
                for file_name in files:
                    path = os.path.join(root, file_name)
                    # workaround for https://github.com/python/typeshed/issues/1229
                    yield (path, lambda: cast(BufferedIOBase, open(path, 'rb')))
        for pak_file in self._paks:
            pak = vpk.open(pak_file)
            for path, metadata in pak.read_index_iter():
                yield (path, lambda: VPKFileIOWrapper(pak.get_vpkfile_instance(path, metadata)))

    def index_files(self) -> None:
        for path, open_func in self.index_files_iter():
            self._tree[path.lower()] = open_func

    def clear_index(self) -> None:
        self._tree.clear()

    def open_file(self, path: str) -> AnyBinaryIO:
        path = path.lower()
        if path not in self._tree:
            raise FileNotFoundError(path)
        return self._tree[path]()


_VECTOR_REGEX = re.compile(r"^\[(-?\d*\.?\d*) (-?\d*\.?\d*) (-?\d*\.?\d*)]$")


class VMFVector(NamedTuple):
    x: float
    y: float
    z: float


class VMFColor(NamedTuple):
    r: int
    g: int
    b: int


class VMFEntity():
    def __init__(self, data: vdf.VDFDict, fs: VMFFileSystem):
        self.data = data
        self.fs = fs
        self.id = int(data["id"])
        self.classname: str = data["classname"]
        if not isinstance(self.classname, str):
            raise ValueError("Invalid VMF file: entity classname is not a str")
        self.origin: Optional[VMFVector] = None
        if "origin" in data:
            origin_value = data["origin"]
            if not isinstance(origin_value, str):
                raise ValueError("Invalid VMF file: entity origin is not a str")
            self.origin = VMFVector(*(float(s) for s in origin_value.split(" ")))
        self.spawnflags: Optional[int] = None
        if "spawnflags" in data:
            self.spawnflags = int(data["spawnflags"])


class VMFPointEntity(VMFEntity):
    def __init__(self, data: vdf.VDFDict, fs: VMFFileSystem):
        super(data, fs)
        assert self.origin is not None
        self.origin: VMFVector


class VMFPropEntity(VMFPointEntity):
    def __init__(self, data: vdf.VDFDict, fs: VMFFileSystem):
        super(data, fs)
        self.angles: VMFVector
        if "angles" in data:
            self.angles = VMFVector(*(float(s) for s in data["angles"].split(" ")))
        else:
            self.angles = VMFVector(0, 0, 0)
        self.model = data["model"]
        if not isinstance(self.model, str):
            raise ValueError("Invalid VMF file: prop entity model is not a str")
        self.skin: int
        if "skin" in data:
            self.skin = int(data["skin"])
        else:
            self.skin = 0

    def open_model(self) -> AnyBinaryIO:
        return self.fs.open_file(self.model)


_PLANE_REGEX = re.compile(r"^\((-?\d*\.?\d*) (-?\d*\.?\d*) (-?\d*\.?\d*)\) "
                          r"\((-?\d*\.?\d*) (-?\d*\.?\d*) (-?\d*\.?\d*)\) "
                          r"\((-?\d*\.?\d*) (-?\d*\.?\d*) (-?\d*\.?\d*)\)$")


class VMFPlane(NamedTuple):
    btm_l: VMFVector
    top_l: VMFVector
    top_r: VMFVector


_AXIS_REGEX = re.compile(r"^\[(-?\d*\.?\d*) (-?\d*\.?\d*) (-?\d*\.?\d*) (-?\d*\.?\d*)] (-?\d*\.?\d*)$")


class VMFAxis(NamedTuple):
    x: float
    y: float
    z: float
    trans: float
    scale: float


class VMFDispInfo():
    def __init__(self, data: vdf.VDFDict):
        self.power = int(data["power"])
        self.triangle_dimension = 2 ** self.power
        self.dimension = self.triangle_dimension + 1

        startpos_match = _VECTOR_REGEX.match(data["startposition"])
        if startpos_match is None:
            raise ValueError("Invalid VMF file: dispinfo startposition is not valid")
        startpos_floats = [float(s) for s in startpos_match.groups()]
        self.startposition = VMFVector(*startpos_floats)

        self.elevation = float(data["elevation"])
        self.subdiv = bool(int(data["subdiv"]))

        normals_dict: vdf.VDFDict = data["normals"]
        if not isinstance(normals_dict, vdf.VDFDict):
            raise ValueError("Invalid VMF file: dispinfo normals is not a dict")
        self.normals: List[List[VMFVector]] = [[VMFVector(0, 0, 0)] * self.dimension] * self.dimension
        for row_name in normals_dict:
            if row_name[:4] != "row":
                raise ValueError("Invalid VMF file: invalid key in dispinfo normals")
            row_idx = int(row_name[4:])
            row_value: str = normals_dict[row_name]
            if not isinstance(row_value, str):
                raise ValueError("Invalid VMF file: a value in dispinfo normals is not a str")
            row_nums_it = iter(row_value.split(" "))
            for idx, vec_tuple in enumerate(zip(row_nums_it, row_nums_it, row_nums_it)):
                self.normals[row_idx][idx] = VMFVector(*(float(s) for s in vec_tuple))

        distances_dict: vdf.VDFDict = data["distances"]
        if not isinstance(distances_dict, vdf.VDFDict):
            raise ValueError("Invalid VMF file: dispinfo distances is not a dict")
        self.distances: List[List[float]] = [[0.] * self.dimension] * self.dimension
        for row_name in distances_dict:
            if row_name[:4] != "row":
                raise ValueError("Invalid VMF file: invalid key in dispinfo distances")
            row_idx = int(row_name[4:])
            row_value = distances_dict[row_name]
            if not isinstance(row_value, str):
                raise ValueError("Invalid VMF file: a value in dispinfo distances is not a str")
            for idx, num_str in enumerate(row_value.split(" ")):
                self.distances[row_idx][idx] = float(num_str)

        offsets_dict: vdf.VDFDict = data["offsets"]
        if not isinstance(offsets_dict, vdf.VDFDict):
            raise ValueError("Invalid VMF file: dispinfo offsets is not a dict")
        self.offsets: List[List[VMFVector]] = [[VMFVector(0, 0, 0)] * self.dimension] * self.dimension
        for row_name in offsets_dict:
            if row_name[:4] != "row":
                raise ValueError("Invalid VMF file: invalid key in dispinfo offsets")
            row_idx = int(row_name[4:])
            row_value = offsets_dict[row_name]
            if not isinstance(row_value, str):
                raise ValueError("Invalid VMF file: a value in dispinfo offsets is not a str")
            row_nums_it = iter(row_value.split(" "))
            for idx, vec_tuple in enumerate(zip(row_nums_it, row_nums_it, row_nums_it)):
                self.offsets[row_idx][idx] = VMFVector(*(float(s) for s in vec_tuple))

        offset_normals_dict: vdf.VDFDict = data["offset_normals"]
        if not isinstance(offset_normals_dict, vdf.VDFDict):
            raise ValueError("Invalid VMF file: dispinfo offset_normals is not a dict")
        self.offset_normals: List[List[VMFVector]] = [[VMFVector(0, 0, 0)] * self.dimension] * self.dimension
        for row_name in offset_normals_dict:
            if row_name[:4] != "row":
                raise ValueError("Invalid VMF file: invalid key in dispinfo offset_normals")
            row_idx = int(row_name[4:])
            row_value = offset_normals_dict[row_name]
            if not isinstance(row_value, str):
                raise ValueError("Invalid VMF file: a value in dispinfo offset_normals is not a str")
            row_nums_it = iter(row_value.split(" "))
            for idx, vec_tuple in enumerate(zip(row_nums_it, row_nums_it, row_nums_it)):
                self.offset_normals[row_idx][idx] = VMFVector(*(float(s) for s in vec_tuple))

        alphas_dict: vdf.VDFDict = data["alphas"]
        if not isinstance(alphas_dict, vdf.VDFDict):
            raise ValueError("Invalid VMF file: dispinfo alphas is not a dict")
        self.alphas: List[List[int]] = [[0] * self.dimension] * self.dimension
        for row_name in alphas_dict:
            if row_name[:4] != "row":
                raise ValueError("Invalid VMF file: invalid key in dispinfo alphas")
            row_idx = int(row_name[4:])
            row_value = alphas_dict[row_name]
            if not isinstance(row_value, str):
                raise ValueError("Invalid VMF file: a value in dispinfo alphas is not a str")
            for idx, num_str in enumerate(row_value.split(" ")):
                self.alphas[row_idx][idx] = int(num_str)

        triangle_tags_dict: vdf.VDFDict = data["triangle_tags"]
        if not isinstance(triangle_tags_dict, vdf.VDFDict):
            raise ValueError("Invalid VMF file: dispinfo triangle_tags is not a dict")
        self.triangle_tags: List[List[Tuple[int, int]]] = [[(0, 0)] * self.triangle_dimension] * self.triangle_dimension
        for row_name in triangle_tags_dict:
            if row_name[:4] != "row":
                raise ValueError("Invalid VMF file: invalid key in dispinfo triangle_tags")
            row_idx = int(row_name[4:])
            row_value = triangle_tags_dict[row_name]
            if not isinstance(row_value, str):
                raise ValueError("Invalid VMF file: a value in dispinfo triangle_tags is not a str")
            row_nums_it = iter(row_value.split(" "))
            for idx, tag_tuple in enumerate(zip(row_nums_it, row_nums_it)):
                self.triangle_tags[row_idx][idx] = (int(tag_tuple[0]), int(tag_tuple[1]))

        allowed_verts_dict: vdf.VDFDict = data["allowed_verts"]
        if not isinstance(allowed_verts_dict, vdf.VDFDict):
            raise ValueError("Invalid VMF file: dispinfo allowed_verts is not a dict")
        allowed_verts_value = allowed_verts_dict["10"]
        if not isinstance(allowed_verts_value, str):
            raise ValueError("Invalid VMF file: value of allowed_verts 10 is not a str")
        self.allowed_verts = tuple((int(s) for s in allowed_verts_value.split(" ")))


class VMFSide():
    def __init__(self, data: vdf.VDFDict, fs: VMFFileSystem):
        self.fs = fs

        self.id = int(data["id"])

        plane_match = _PLANE_REGEX.match(data["plane"])
        if plane_match is None:
            raise ValueError("Invalid VMF file: side plane is not valid")
        plane_floats = [float(s) for s in plane_match.groups()]
        self.plane = VMFPlane(VMFVector(*plane_floats[:3]),
                              VMFVector(*plane_floats[3:6]),
                              VMFVector(*plane_floats[6:9]))

        self.material = data["material"]
        if not isinstance(self.material, str):
            raise ValueError("Invalid VMF file: side material is not a str")
        self.materialpath = "materials/" + self.material + ".vmt"

        uaxis_match = _AXIS_REGEX.match(data["uaxis"])
        if uaxis_match is None:
            raise ValueError("Invalid VMF file: side uaxis is not valid")
        uaxis_floats = [float(s) for s in uaxis_match.groups()]
        self.uaxis = VMFAxis(*uaxis_floats)

        vaxis_match = _AXIS_REGEX.match(data["vaxis"])
        if vaxis_match is None:
            raise ValueError("Invalid VMF file: side vaxis is not valid")
        vaxis_floats = [float(s) for s in vaxis_match.groups()]
        self.vaxis = VMFAxis(*vaxis_floats)

        self.rotation = float(data["rotation"])
        self.lightmapscale = int(data["lightmapscale"])
        self.smoothing_groups = int(data["smoothing_groups"]).to_bytes(4, 'little')

        self.dispinfo: Optional[VMFDispInfo] = None
        if "dispinfo" in data:
            dispinfo_dict = data["dispinfo"]
            if not isinstance(dispinfo_dict, vdf.VDFDict):
                raise ValueError("Invalid VMF file: side dispinfo is not a dict")
            self.dispinfo = VMFDispInfo(dispinfo_dict)

    def open_material_file(self) -> TextIOWrapper:
        return TextIOWrapper(cast(IO[bytes],
                             self.fs.open_file(self.materialpath)),
                             encoding='utf-8')


class VMFSolid():
    def __init__(self, data: vdf.VDFDict, fs: VMFFileSystem):
        self.fs = fs
        self.id = int(data["id"])
        sides = data.get_all_for("side")
        for side in sides:
            if not isinstance(side, vdf.VDFDict):
                raise ValueError("Invalid VMF file: a solid side is not a dict")
        self._sides: List[vdf.VDFDict] = sides

    def iter_sides(self) -> Iterator:
        for side in self._sides:
            yield VMFSide(side, self.fs)


class VMFBrushEntity(VMFEntity):
    def __init__(self, data: vdf.VDFDict, fs: VMFFileSystem):
        super(data, fs)
        solids = data.get_all_for("solid")
        for solid in solids:
            if not isinstance(solid, vdf.VDFDict):
                raise ValueError("Invalid VMF file: a solid is not a dict")
        self._solids: List[vdf.VDFDict] = solids

    def iter_solids(self) -> Iterator[VMFSolid]:
        for solid in self._solids:
            yield VMFSolid(solid, self.fs)


class VMFWorldEntity(VMFBrushEntity):
    def __init__(self, data: vdf.VDFDict, fs: VMFFileSystem):
        super(data, fs)
        assert self.classname == "worldspawn"
        self.skyname = data["skyname"]
        if not isinstance(self.skyname, str):
            raise ValueError("Invalid VMF file: world skyname is not a str")
        self.skypath = "materials/skybox/" + self.skyname + ".vmt"

    def open_sky(self) -> TextIOWrapper:
        return TextIOWrapper(cast(IO[bytes],
                             self.fs.open_file(self.skypath)),
                             encoding='utf-8')


class VMF():
    def __init__(self, file: AnyTextIO, data_dirs: Iterable[str] = [], data_paks: Iterable[str] = []):
        self.fs = VMFFileSystem()
        for data_dir in data_dirs:
            self.fs.add_dir(data_dir)
        for data_pak in data_paks:
            self.fs.add_pak(data_pak)
        self.fs.index_files()
        vdf_dict = vdf.load(file, mapper=vdf.VDFDict, merge_duplicate_keys=False, escaped=False)

        versioninfo: vdf.VDFDict = vdf_dict["versioninfo"]
        if not isinstance(versioninfo, vdf.VDFDict):
            raise ValueError("Invalid VMF file: versioninfo is not a dict")
        self.editorversion = int(versioninfo["editorversion"])
        self.editorbuild = int(versioninfo["editorbuild"])
        self.mapversion = int(versioninfo["mapversion"])
        self.prefab = bool(int(versioninfo["prefab"]))

        world: vdf.VDFDict = vdf_dict["world"]
        if not isinstance(world, vdf.VDFDict):
            raise ValueError("Invalid VMF file: world is not a dict")
        self.world = VMFWorldEntity(world, self.fs)

        self.entities: List[VMFEntity] = list()
        self.func_entities: List[VMFBrushEntity] = list()
        self.prop_entities: List[VMFPropEntity] = list()

        entities = vdf_dict.get_all_for("entity")
        for entity in entities:
            if not isinstance(entity, vdf.VDFDict):
                raise ValueError("Invalid VMF file: entity is not a dict")
            classname: str = entity["classname"]
            if not isinstance(classname, str):
                raise ValueError("Invalid VMF file: entity classname is not a str")
            entity_inst: VMFEntity
            if classname.startswith("func"):
                entity_inst = VMFBrushEntity(entity, self.fs)
                self.func_entities.append(entity_inst)
            elif classname.startswith("prop"):
                entity_inst = VMFPropEntity(entity, self.fs)
                self.prop_entities.append(entity_inst)
            elif "origin" in entity:
                entity_inst = VMFPointEntity(entity, self.fs)
            else:
                entity_inst = VMFEntity(entity, self.fs)
            entities.append(entity_inst)
