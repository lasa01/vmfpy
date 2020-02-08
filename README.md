# vmfpy

A Valve Map Format (VMF) parser.

## Installation
```
pip install git+https://github.com/lasa01/vmfpy.git#egg=vmfpy
```

## Usage
```py
import vmfpy

CSGO_PAK = "C:/Program Files (x86)/Steam/steamapps/common/Counter-Strike Global Offensive/csgo/pak01_dir.vpk"

vmf = vmfpy.VMF(open("test.vmf", encoding="utf-8"), data_paks=(CSGO_PAK,))

print(vmf.world.skyname)
print(vmf.world.solids[0].sides[0].material)

with vmf.func_entities[0].solids[0].sides[0].open_material_file() as mtlf:
    with open("out.mtl", "w", encoding="utf-8") as outf:
        for line in mtlf:
            outf.write(line)

```