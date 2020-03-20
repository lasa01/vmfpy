import vdf
from .fs import VMFFileSystem, vmf_path, AnyBinaryIO, AnyTextIO
from pathlib import PurePosixPath
from collections import defaultdict
from typing import DefaultDict, Dict, NamedTuple, Tuple
import re


class VMTParseException(Exception):
    pass


_VEC2_REGEX = re.compile(r"^\[(-?\d*\.?\d*e?-?\d*) (-?\d*\.?\d*e?-?\d*)]$")
_VEC3_REGEX = re.compile(r"^\[(-?\d*\.?\d*e?-?\d*) (-?\d*\.?\d*e?-?\d*) (-?\d*\.?\d*e?-?\d*)]$")
_VEC4_REGEX = re.compile(r"^\[(-?\d*\.?\d*e?-?\d*) (-?\d*\.?\d*e?-?\d*) (-?\d*\.?\d*e?-?\d*) (-?\d*\.?\d*e?-?\d*)]$")


def _parse_vec2(value: str) -> Tuple[float, float]:
    match = _VEC2_REGEX.match(value)
    if match is None:
        raise VMTParseException("vector2 syntax is invalid")
    try:
        return tuple((float(s) for s in match.groups()))  # type: ignore
    except ValueError:
        raise VMTParseException("vector2 contains an invalid float")
    except OverflowError:
        raise VMTParseException("vector2 float out of range")


def _parse_vec3(value: str) -> Tuple[float, float, float]:
    match = _VEC3_REGEX.match(value)
    if match is None:
        raise VMTParseException("vector3 syntax is invalid")
    try:
        return tuple((float(s) for s in match.groups()))  # type: ignore
    except ValueError:
        raise VMTParseException("vector3 contains an invalid float")
    except OverflowError:
        raise VMTParseException("vector3 float out of range")


def _parse_vec4(value: str) -> Tuple[float, float, float, float]:
    match = _VEC4_REGEX.match(value)
    if match is None:
        raise VMTParseException("vector4 syntax is invalid")
    try:
        return tuple((float(s) for s in match.groups()))  # type: ignore
    except ValueError:
        raise VMTParseException("vector4 contains an invalid float")
    except OverflowError:
        raise VMTParseException("vector4 float out of range")


_I_COLOR_REGEX = re.compile(r"^{(\d+) (\d+) (\d+)}$")


class VMTColor(NamedTuple):
    r: float
    g: float
    b: float

    @staticmethod
    def _parse(value: str) -> 'VMTColor':
        match = _I_COLOR_REGEX.match(value)
        if match is not None:
            try:
                colors = [int(s) / 255 for s in match.groups()]
            except ValueError:
                raise VMTParseException("color contains an invalid int")
            return VMTColor(*colors)
        match = _VEC3_REGEX.match(value)
        if match is None:
            raise VMTParseException("color syntax is invalid")
        try:
            colors = [float(s) for s in match.groups()]
        except ValueError:
            raise VMTParseException("color contains an invalid float")
        except OverflowError:
            raise VMTParseException("color float out of range")
        return VMTColor(*colors)


class VMT():
    def __init__(self, file: AnyTextIO, fs: VMFFileSystem = VMFFileSystem()) -> None:
        self.fs = fs
        vdf_dict: dict = vdf.load(file, escaped=False)
        if len(vdf_dict) != 1:
            raise VMTParseException("material does not contain exactly 1 member")
        shader_name: str = next(iter(vdf_dict))
        self.shader = shader_name.lower()
        shader_dict: dict = vdf_dict[shader_name]
        if not isinstance(shader_dict, dict):
            raise VMTParseException("shader is not a dict")
        self.flags: DefaultDict[str, bool] = defaultdict(lambda: False)
        self.parameters: Dict[str, str] = {}
        for key in shader_dict:
            key_l = key.lower()
            if key.startswith("$") or key.startswith("%"):
                value = shader_dict[key]
                if not isinstance(value, str):
                    raise VMTParseException(f"{key} is not a str")
                self.parameters[key_l] = value
            elif key_l == "proxies":
                self.proxies: dict = shader_dict[key]
                if not isinstance(self.proxies, dict):
                    raise VMTParseException("proxies is not a dict")

    def param_as_int(self, param: str) -> int:
        try:
            return int(self.parameters[param])
        except ValueError:
            raise VMTParseException(f"{param} is not a valid int")

    def param_as_float(self, param: str) -> float:
        try:
            return float(self.parameters[param])
        except ValueError:
            raise VMTParseException(f"{param} is not a valid float")
        except OverflowError:
            raise VMTParseException(f"{param} is out of range")

    def param_as_bool(self, param: str) -> bool:
        intv = self.param_as_int(param)
        if intv not in (0, 1):
            raise VMTParseException(f"{param} is not a valid bool")
        return bool(intv)

    def param_flag(self, param: str) -> bool:
        try:
            return self.param_as_bool(param)
        except KeyError:
            return False

    def param_as_texture(self, param: str) -> PurePosixPath:
        return "materials" / vmf_path(self.parameters[param]).with_suffix(".vtf")

    def param_open_texture(self, param: str) -> AnyBinaryIO:
        return self.fs.open_file(self.param_as_texture(param))

    def param_as_vec2(self, param: str) -> Tuple[float, float]:
        return _parse_vec2(self.parameters[param])

    def param_as_vec3(self, param: str) -> Tuple[float, float, float]:
        return _parse_vec3(self.parameters[param])

    def param_as_vec4(self, param: str) -> Tuple[float, float, float, float]:
        return _parse_vec4(self.parameters[param])

    def param_as_color(self, param: str) -> VMTColor:
        return VMTColor._parse(self.parameters[param])