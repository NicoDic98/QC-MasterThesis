from os import listdir
from os.path import isfile, join
from pathlib import Path
from contextlib import redirect_stdout

import h5py

data_folder = "results/data/"
plots_folder = "results/plots/"
default_id = 42


def fill_defaults_in_dict(my_dict: dict, my_default_dict: dict):
    for key, value in my_default_dict.items():
        if key not in my_dict:
            my_dict[key] = value


def print_h5(name, obj: h5py.Group):
    print(f"{name}:{obj}")
    for key in obj.attrs.keys():
        print(f"\t{key}: {obj.attrs[key]}")


def pprint_h5(obj: h5py.Group, depth: int = 0, basename=""):
    prefix = depth * "\t"
    if depth == 0:
        print("-----Beginning pprint-----")
        print(prefix + f"{basename}{obj.name}:")
    else:
        print(prefix + f"{str(obj.name).split("/")[-1]}:")
    print(prefix + "\tAttributes:")
    for key, value in obj.attrs.items():
        print(prefix + f"\t\t{key}: {value}")
    for key, value in obj.items():
        if isinstance(obj.get(key, getlink=True), h5py.ExternalLink):
            print(prefix + f"\t{obj.get(key, getlink=True)} :")
        if isinstance(value, h5py.Group):
            pprint_h5(value, depth=depth + 1)
        else:
            print(prefix + f"\t{key}: {value}")
            print(prefix + "\t\tAttributes:")
            for attr_key, attr_value in value.attrs.items():
                print(prefix + f"\t\t\t{attr_key}: {attr_value}")
    if depth == 0:
        print("-----Ending pprint-----")


def pprint_all(summary=False):
    hdf5_files = [f for f in listdir(data_folder) if (isfile(join(data_folder, f)) and (Path(f).suffix == '.hdf5'))]
    if summary:
        print("-----Beginning summary-----")
    for file in hdf5_files:
        try:
            with h5py.File(join(data_folder, file), "r") as f:
                if summary:
                    print(f"{file}{f.name}:")
                    for name in f.keys():
                        print(f"\t{name}")
                else:
                    pprint_h5(f, basename=file)
        except OSError as e:
            if "Unable to synchronously" in str(e):
                print(f"Unable to open source file: {file}")
            else:
                raise e
    if summary:
        print("-----Ending summary-----")


if __name__ == "__main__":
    with open('log.txt', 'w') as f:
        with redirect_stdout(f):
            pprint_all()
            pprint_all(True)
