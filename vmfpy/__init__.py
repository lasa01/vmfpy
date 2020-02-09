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


class VMFFileSystem():
    """File system for opening game files."""
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
        def _create_f_opener(p: str) -> Callable[[], AnyBinaryIO]:
            return lambda: open(p, 'rb')
        for directory in self._dirs:
            root: str
            files: List[str]
            for root, _, files in os.walk(directory):
                root = os.path.relpath(root, directory).replace("\\", "/")
                for file_name in files:
                    path = os.path.join(root, file_name)
                    yield (path, _create_f_opener(path))
        for pak_file in self._paks:
            pak = vpk.open(pak_file)

            def _create_pf_opener(p: str, m: Tuple[bytes, int, int, int, int, int]) -> Callable[[], AnyBinaryIO]:
                return lambda: VPKFileIOWrapper(pak.get_vpkfile_instance(p, m))
            for path, metadata in pak.read_index_iter():
                yield (path, _create_pf_opener(path, metadata))

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
        return VMFVector(*(float(s) for s in nums))

    @staticmethod
    def parse_sq_brackets(data: str) -> 'VMFVector':
        match = _VECTOR_REGEX.match(data)
        assert match is not None
        return VMFVector(*(float(s) for s in match.groups()))

    @staticmethod
    def parse_tuple(data: Tuple[str, str, str]) -> 'VMFVector':
        return VMFVector(*(float(s) for s in data))


class VMFColor(NamedTuple):
    """A color value using 3 integers between 0 and 255."""
    r: int
    g: int
    b: int


class VMFEntity():
    """An entity."""
    def __init__(self, data: vdf.VDFDict, fs: VMFFileSystem):
        self.data = data
        self.fs = fs
        self.id = int(data["id"])
        """This is a unique number among other entity ids."""
        self.classname: str = data["classname"]
        """This is the name of the entity class."""
        if not isinstance(self.classname, str):
            raise ValueError("Invalid VMF file: entity classname is not a str")
        self.origin: Optional[VMFVector] = None
        """This is the point where the point entity exists."""
        if "origin" in data:
            origin_value = data["origin"]
            if not isinstance(origin_value, str):
                raise ValueError("Invalid VMF file: entity origin is not a str")
            self.origin = VMFVector.parse_str(origin_value)
        self.spawnflags: Optional[int] = None
        """Indicates which flags are enabled on the entity."""
        if "spawnflags" in data:
            self.spawnflags = int(data["spawnflags"])


class VMFPointEntity(VMFEntity):
    """A point based entity."""
    def __init__(self, data: vdf.VDFDict, fs: VMFFileSystem):
        super().__init__(data, fs)
        assert self.origin is not None
        self.origin: VMFVector


class VMFPropEntity(VMFPointEntity):
    """Adds a model to the world."""
    def __init__(self, data: vdf.VDFDict, fs: VMFFileSystem):
        super().__init__(data, fs)
        self.angles: VMFVector
        """This entity's orientation in the world."""
        if "angles" in data:
            angles_value = data["angles"]
            if not isinstance(angles_value, str):
                raise ValueError("Invalid VMF file: prop entity angles is not a str")
            self.angles = VMFVector.parse_str(angles_value)
        else:
            self.angles = VMFVector(0, 0, 0)
        self.model = data["model"]
        """The model this entity should appear as."""
        if not isinstance(self.model, str):
            raise ValueError("Invalid VMF file: prop entity model is not a str")
        self.skin: int
        """Some models have multiple skins. This value selects from the index, starting with 0."""
        if "skin" in data:
            self.skin = int(data["skin"])
        else:
            self.skin = 0

    def open_model(self) -> AnyBinaryIO:
        return self.fs.open_file(self.model)


class VMFOverlayEntity(VMFPointEntity):
    """More powerful version of a material projected onto existing surfaces."""
    def __init__(self, data: vdf.VDFDict, fs: VMFFileSystem):
        super().__init__(data, fs)

        self.material = data["material"]
        """The material to overlay."""
        if not isinstance(self.material, str):
            raise ValueError("Invalid VMF file: overlay material is not a str")
        self.materialpath = "materials/" + self.material + ".vmt"

        sides_value: str = data["sides"]
        if not isinstance(sides_value, str):
            raise ValueError("Invalid VMF file: overlay sides is not a str")
        self.sides: List[int] = [int(s) for s in sides_value.split(" ")]
        """Faces on which the overlay will be applied."""

        self.renderorder: Optional[int] = None
        """Higher values render after lower values (on top). This value can be 0–3."""
        if "RenderOrder" in data:
            self.renderorder = int(data["RenderOrder"])

        self.startu = float(data["StartU"])
        """Texture coordinates for the image."""
        self.startv = float(data["StartV"])
        """Texture coordinates for the image."""
        self.endu = float(data["EndU"])
        """Texture coordinates for the image."""
        self.endv = float(data["EndV"])
        """Texture coordinates for the image."""
        self.basisorigin = VMFVector.parse_str(data["BasisOrigin"])
        """Offset of the surface from the position of the overlay entity."""
        self.basisu = VMFVector.parse_str(data["BasisU"])
        """Direction of the material's X-axis."""
        self.basisv = VMFVector.parse_str(data["BasisV"])
        """Direction of the material's Y-axis."""
        self.basisnormal = VMFVector.parse_str(data["BasisNormal"])
        """Direction out of the surface."""
        self.uv0 = VMFVector.parse_str(data["uv0"])
        self.uv1 = VMFVector.parse_str(data["uv1"])
        self.uv2 = VMFVector.parse_str(data["uv2"])
        self.uv3 = VMFVector.parse_str(data["uv3"])

    def open_material_file(self) -> TextIOWrapper:
        return TextIOWrapper(cast(IO[bytes],
                             self.fs.open_file(self.materialpath)),
                             encoding='utf-8')


class VMFLightEntity(VMFPointEntity):
    """Creates an invisible, static light source that shines in all directions."""
    def __init__(self, data: vdf.VDFDict, fs: VMFFileSystem):
        super().__init__(data, fs)
        light_value: str = data["_light"]
        if not isinstance(light_value, str):
            raise ValueError("Invalid VMF file: light _light is not a str")
        light_list = [int(s) for s in light_value.split(" ")]
        self.color = VMFColor(*light_list[:3])
        """The RGB color of the light."""
        self.brightness = light_list[3]
        """The brightness of the light."""

        light_hdr_value: str = data["_lightHDR"]
        if not isinstance(light_hdr_value, str):
            raise ValueError("Invalid VMF file: light _lightHDR is not a str")
        light_hdr_list = [int(s) for s in light_value.split(" ")]
        self.hdr_color = VMFColor(*light_hdr_list[:3])
        """Color override used in HDR mode. Default is -1 -1 -1 which means no change."""
        self.hdr_brightness = light_hdr_list[3]
        """Brightness override used in HDR mode. Default is 1 which means no change."""
        self.hdr_scale = float(data["_lightscaleHDR"])
        """A simple intensity multiplier used when compiling HDR lighting."""

        self.style: Optional[int] = None
        """Various Custom Appearance presets."""
        if "style" in data:
            self.style = int(data["style"])
        self.constant_attn: Optional[float] = None
        """Determines how the intensity of the emitted light falls off over distance."""
        if "_constant_attn" in data:
            self.constant_attn = float(data["_constant_attn"])
        self.linear_attn: Optional[float] = None
        """Determines how the intensity of the emitted light falls off over distance."""
        if "_linear_attn" in data:
            self.linear_attn = float(data["_linear_attn"])
        self.quadratic_attn: Optional[float] = None
        """Determines how the intensity of the emitted light falls off over distance."""
        if "_quadratic_attn" in data:
            self.quadratic_attn = float(data["_quadratic_attn"])
        self.fifty_percent_distance: Optional[float] = None
        """Distance at which brightness should have fallen to 50%. Overrides attn if non-zero."""
        if "_fifty_percent_distance" in data:
            self.fifty_percent_distance = float(data["_fifty_percent_distance"])
        self.zero_percent_distance: Optional[float] = None
        """Distance at which brightness should have fallen to (1/256)%. Overrides attn if non-zero."""
        if "_zero_percent_distance" in data:
            self.zero_percent_distance = float(data["_zero_percent_distance"])


class VMFSpotLightEntity(VMFLightEntity):
    """A cone-shaped, invisible light source."""
    def __init__(self, data: vdf.VDFDict, fs: VMFFileSystem):
        super().__init__(data, fs)
        self.angles: VMFVector
        """This entity's orientation in the world."""
        if "angles" in data:
            angles_value = data["angles"]
            if not isinstance(angles_value, str):
                raise ValueError("Invalid VMF file: prop entity angles is not a str")
            self.angles = VMFVector.parse_str(angles_value)
        else:
            self.angles = VMFVector(0, 0, 0)
        self.pitch = float(data["pitch"])
        """Used instead of angles value for reasons unknown."""

        self.inner_cone = int(data["_inner_cone"])
        """The angle of the inner spotlight beam."""
        self.cone = int(data["_cone"])
        """The angle of the outer spotlight beam."""
        self.exponent = int(data["_exponent"])
        """Changes the distance between the umbra and penumbra cone."""


class VMFEnvLightEntity(VMFLightEntity):
    """Casts parallel directional lighting and diffuse skylight from the toolsskybox texture."""
    def __init__(self, data: vdf.VDFDict, fs: VMFFileSystem):
        super().__init__(data, fs)
        self.angles: VMFVector
        """This entity's orientation in the world."""
        if "angles" in data:
            angles_value = data["angles"]
            if not isinstance(angles_value, str):
                raise ValueError("Invalid VMF file: prop entity angles is not a str")
            self.angles = VMFVector.parse_str(angles_value)
        else:
            self.angles = VMFVector(0, 0, 0)
        self.pitch = float(data["pitch"])
        """Used instead of angles value for reasons unknown."""

        amb_light_value: str = data["_ambient"]
        if not isinstance(amb_light_value, str):
            raise ValueError("Invalid VMF file: light _ambient is not a str")
        amb_light_list = [int(s) for s in amb_light_value.split(" ")]
        self.amb_color = VMFColor(*amb_light_list[:3])
        """Color of the diffuse skylight."""
        self.amb_brightness = amb_light_list[3]
        """Brightness of the diffuse skylight."""

        amb_light_hdr_value: str = data["_ambientHDR"]
        if not isinstance(amb_light_hdr_value, str):
            raise ValueError("Invalid VMF file: light _ambientHDR is not a str")
        amb_light_hdr_list = [int(s) for s in amb_light_value.split(" ")]
        self.amb_hdr_color = VMFColor(*amb_light_hdr_list[:3])
        """Override for ambient color when compiling HDR lighting."""
        self.amb_hdr_brightness = amb_light_hdr_list[3]
        """Override for ambient brightness when compiling HDR lighting."""
        self.amb_hdr_scale = float(data["_AmbientScaleHDR"])
        """Amount to scale the ambient light by when compiling for HDR."""

        self.sun_spread_angle = float(data["SunSpreadAngle"])
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
        assert match is not None
        floats = [float(s) for s in match.groups()]
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
        assert match is not None
        floats = [float(s) for s in match.groups()]
        return VMFAxis(*floats)


class VMFDispInfo():
    """Deals with all the information for a displacement map."""
    def __init__(self, data: vdf.VDFDict):
        self.power = int(data["power"])
        """Used to calculate the number of rows and columns."""
        self.triangle_dimension = 2 ** self.power
        """The number of rows and columns in triangles."""
        self.dimension = self.triangle_dimension + 1
        """The number of rows and columns in vertexes."""
        self.startposition = VMFVector.parse_sq_brackets(data["startposition"])
        """The position of the bottom left corner in an actual x y z position."""
        self.elevation = float(data["elevation"])
        """A universal displacement in the direction of the vertex's normal added to all of the points."""
        self.subdiv = bool(int(data["subdiv"]))
        """Marks whether or not the displacement is being subdivided."""

        normals_dict: vdf.VDFDict = data["normals"]
        if not isinstance(normals_dict, vdf.VDFDict):
            raise ValueError("Invalid VMF file: dispinfo normals is not a dict")
        self.normals: List[List[VMFVector]] = [[VMFVector(0, 0, 0)] * self.dimension] * self.dimension
        """Defines the normal line for each vertex."""
        for row_name in normals_dict:
            if row_name[:3] != "row":
                raise ValueError("Invalid VMF file: invalid key in dispinfo normals")
            row_idx = int(row_name[3:])
            row_value: str = normals_dict[row_name]
            if not isinstance(row_value, str):
                raise ValueError("Invalid VMF file: a value in dispinfo normals is not a str")
            row_nums_it = iter(row_value.split(" "))
            vec_tuple: Tuple[str, str, str]
            for idx, vec_tuple in enumerate(zip(row_nums_it, row_nums_it, row_nums_it)):
                self.normals[row_idx][idx] = VMFVector.parse_tuple(vec_tuple)

        distances_dict: vdf.VDFDict = data["distances"]
        if not isinstance(distances_dict, vdf.VDFDict):
            raise ValueError("Invalid VMF file: dispinfo distances is not a dict")
        self.distances: List[List[float]] = [[0.] * self.dimension] * self.dimension
        """The distance values represent how much the vertex is moved along the normal line."""
        for row_name in distances_dict:
            if row_name[:3] != "row":
                raise ValueError("Invalid VMF file: invalid key in dispinfo distances")
            row_idx = int(row_name[3:])
            row_value = distances_dict[row_name]
            if not isinstance(row_value, str):
                raise ValueError("Invalid VMF file: a value in dispinfo distances is not a str")
            for idx, num_str in enumerate(row_value.split(" ")):
                self.distances[row_idx][idx] = float(num_str)

        offsets_dict: vdf.VDFDict = data["offsets"]
        if not isinstance(offsets_dict, vdf.VDFDict):
            raise ValueError("Invalid VMF file: dispinfo offsets is not a dict")
        self.offsets: List[List[VMFVector]] = [[VMFVector(0, 0, 0)] * self.dimension] * self.dimension
        """Lists all the default positions for each vertex in a displacement map."""
        for row_name in offsets_dict:
            if row_name[:3] != "row":
                raise ValueError("Invalid VMF file: invalid key in dispinfo offsets")
            row_idx = int(row_name[3:])
            row_value = offsets_dict[row_name]
            if not isinstance(row_value, str):
                raise ValueError("Invalid VMF file: a value in dispinfo offsets is not a str")
            row_nums_it = iter(row_value.split(" "))
            for idx, vec_tuple in enumerate(zip(row_nums_it, row_nums_it, row_nums_it)):
                self.offsets[row_idx][idx] = VMFVector.parse_tuple(vec_tuple)
        offset_normals_dict: vdf.VDFDict = data["offset_normals"]
        if not isinstance(offset_normals_dict, vdf.VDFDict):
            raise ValueError("Invalid VMF file: dispinfo offset_normals is not a dict")
        self.offset_normals: List[List[VMFVector]] = [[VMFVector(0, 0, 0)] * self.dimension] * self.dimension
        """Defines the default normal lines that the normals are based from."""
        for row_name in offset_normals_dict:
            if row_name[:3] != "row":
                raise ValueError("Invalid VMF file: invalid key in dispinfo offset_normals")
            row_idx = int(row_name[3:])
            row_value = offset_normals_dict[row_name]
            if not isinstance(row_value, str):
                raise ValueError("Invalid VMF file: a value in dispinfo offset_normals is not a str")
            row_nums_it = iter(row_value.split(" "))
            for idx, vec_tuple in enumerate(zip(row_nums_it, row_nums_it, row_nums_it)):
                self.offset_normals[row_idx][idx] = VMFVector.parse_tuple(vec_tuple)

        alphas_dict: vdf.VDFDict = data["alphas"]
        if not isinstance(alphas_dict, vdf.VDFDict):
            raise ValueError("Invalid VMF file: dispinfo alphas is not a dict")
        self.alphas: List[List[float]] = [[0] * self.dimension] * self.dimension
        """Contains a value for each vertex that represents how much of which texture to shown in blended materials."""
        for row_name in alphas_dict:
            if row_name[:3] != "row":
                raise ValueError("Invalid VMF file: invalid key in dispinfo alphas")
            row_idx = int(row_name[3:])
            row_value = alphas_dict[row_name]
            if not isinstance(row_value, str):
                raise ValueError("Invalid VMF file: a value in dispinfo alphas is not a str")
            for idx, num_str in enumerate(row_value.split(" ")):
                self.alphas[row_idx][idx] = float(num_str)

        triangle_tags_dict: vdf.VDFDict = data["triangle_tags"]
        if not isinstance(triangle_tags_dict, vdf.VDFDict):
            raise ValueError("Invalid VMF file: dispinfo triangle_tags is not a dict")
        self.triangle_tags: List[List[Tuple[int, int]]] = [[(0, 0)] * self.triangle_dimension] * self.triangle_dimension
        """Contains information specific to each triangle in the displacement."""
        for row_name in triangle_tags_dict:
            if row_name[:3] != "row":
                raise ValueError("Invalid VMF file: invalid key in dispinfo triangle_tags")
            row_idx = int(row_name[3:])
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
        """This states which vertices share an edge with another displacement map, but not a vertex."""


class VMFSide():
    """Defines all the data relevant to one side and just to that side."""
    def __init__(self, data: vdf.VDFDict, fs: VMFFileSystem):
        self.fs = fs
        """File system for opening game files."""
        self.id = int(data["id"])
        """A unique value among other sides ids."""
        self.plane = VMFPlane.parse(data["plane"])
        """Defines the orientation of the face."""
        self.material = data["material"]
        """The directory and name of the texture the side has applied to it."""
        if not isinstance(self.material, str):
            raise ValueError("Invalid VMF file: side material is not a str")
        self.materialpath = "materials/" + self.material + ".vmt"
        self.uaxis = VMFAxis.parse(data["uaxis"])
        """The u-axis and v-axis are the texture specific axes."""
        self.vaxis = VMFAxis.parse(data["vaxis"])
        """The u-axis and v-axis are the texture specific axes."""
        self.rotation = float(data["rotation"])
        """The rotation of the given texture on the side."""
        self.lightmapscale = int(data["lightmapscale"])
        """The light map resolution on the face."""
        self.smoothing_groups = int(data["smoothing_groups"]).to_bytes(4, 'little')
        """"Select a smoothing group to use for lighting on the face."""

        self.dispinfo: Optional[VMFDispInfo] = None
        """Deals with all the information for a displacement map."""
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
    """Represents 1 single brush in Hammer."""
    def __init__(self, data: vdf.VDFDict, fs: VMFFileSystem):
        self.fs = fs
        """File system for opening game files."""
        self.id = int(data["id"])
        """A unique value among other solids' IDs."""
        dict_sides = data.get_all_for("side")
        self.sides: List[VMFSide] = list()
        for side in dict_sides:
            if not isinstance(side, vdf.VDFDict):
                raise ValueError("Invalid VMF file: a solid side is not a dict")
            self.sides.append(VMFSide(side, self.fs))


class VMFBrushEntity(VMFEntity):
    """A brush based entity."""
    def __init__(self, data: vdf.VDFDict, fs: VMFFileSystem):
        super().__init__(data, fs)
        dict_solids = data.get_all_for("solid")
        self.solids: List[VMFSolid] = list()
        for solid in dict_solids:
            if not isinstance(solid, vdf.VDFDict):
                raise ValueError("Invalid VMF file: a solid is not a dict")
            self.solids.append(VMFSolid(solid, self.fs))


class VMFWorldEntity(VMFBrushEntity):
    """Contains all the world brush information for Hammer."""
    def __init__(self, data: vdf.VDFDict, fs: VMFFileSystem):
        super().__init__(data, fs)
        assert self.classname == "worldspawn"
        self.skyname = data["skyname"]
        """The name of the skybox to be used."""
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
        """File system for opening game files."""
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
        """The version of Hammer used to create the file, version 4.00 is "400"."""
        self.editorbuild = int(versioninfo["editorbuild"])
        """The patch number of Hammer the file was generated with."""
        self.mapversion = int(versioninfo["mapversion"])
        """This represents how many times you've saved the file, useful for comparing old or new versions."""
        self.prefab = bool(int(versioninfo["prefab"]))
        """Whether this is a full map or simply a collection of prefabricated objects."""

        world_dict: vdf.VDFDict = vdf_dict["world"]
        if not isinstance(world_dict, vdf.VDFDict):
            raise ValueError("Invalid VMF file: world is not a dict")
        self.world = VMFWorldEntity(world_dict, self.fs)
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
            if not isinstance(entity, vdf.VDFDict):
                raise ValueError("Invalid VMF file: entity is not a dict")
            classname: str = entity["classname"]
            if not isinstance(classname, str):
                raise ValueError("Invalid VMF file: entity classname is not a str")
            entity_inst: VMFEntity
            if classname == "info_overlay":
                entity_inst = VMFOverlayEntity(entity, self.fs)
                self.overlay_entities.append(entity_inst)
            elif classname == "light_environment":
                entity_inst = VMFEnvLightEntity(entity, self.fs)
                self.env_light_entity = entity_inst
            elif classname == "light_spot":
                entity_inst = VMFSpotLightEntity(entity, self.fs)
                self.spot_light_entities.append(entity_inst)
            elif classname.startswith("light"):
                entity_inst = VMFLightEntity(entity, self.fs)
                self.light_entities.append(entity_inst)
            elif classname.startswith("func"):
                entity_inst = VMFBrushEntity(entity, self.fs)
                self.func_entities.append(entity_inst)
            elif classname.startswith("prop"):
                entity_inst = VMFPropEntity(entity, self.fs)
                self.prop_entities.append(entity_inst)
            elif "origin" in entity:
                entity_inst = VMFPointEntity(entity, self.fs)
            else:
                entity_inst = VMFEntity(entity, self.fs)
            self.entities.append(entity_inst)
