from enum import StrEnum

import h5py

class GlobalParameters(StrEnum):
    SystemName = "SystemName"
    SolverName = "SolverName"
    LastModified = "Last-Modified"

def print_h5(name, obj: h5py.Group):
    print(f"{name}:{obj}")
    for key in obj.attrs.keys():
        print(f"\t{key}: {obj.attrs[key]}")

def pprint_h5(obj: h5py.Group, depth: int = 0):
    prefix = depth * "\t"
    if depth == 0:
        print("-----Beginning pprint-----")
        print(prefix+f"{obj.name}:")
    else:
        print(prefix+f"{str(obj.name).split("/")[-1]}:")
    print(prefix+"\tAttributes:")
    for key, value in obj.attrs.items():
        print(prefix+f"\t\t{key}: {value}")
    for key, value in obj.items():
        if isinstance(value, h5py.Group):
            pprint_h5(value, depth=depth+1)
        else:
            print(prefix+f"\t{key}: {value}")
    if depth == 0:
        print("-----Ending pprint-----")