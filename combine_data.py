from os import listdir, stat
from os.path import isfile, join
from pathlib import Path

import h5py

from misc import data_folder


def combine_data():
    print("-----Combining data-----")
    hdf5_files = [f for f in listdir(data_folder) if (isfile(join(data_folder, f)) and (Path(f).suffix == '.hdf5'))]

    # cmd_squ = ['squeue', '--me', '--json']
    # add_proc = subprocess.run(cmd_squ, capture_output=True, check=True)
    # return_text = add_proc.stdout.decode('utf-8')
    # my_json = json.loads(return_text)
    # job_list = my_json["jobs"]

    combinable_files = []
    for hd5_file in hdf5_files:
        name_components = hd5_file.split("-")
        if len(name_components) < 4:
            print(f"Skipping combined file {hd5_file}")
        elif len(name_components) == 4:
            print(f"Found uncombined file {hd5_file}")
            combinable_files.append(hd5_file)
        else:
            print(f"Skipping {hd5_file}")

    for combinable_file in combinable_files:
        base_name = combinable_file.rsplit("-", 1)[0]
        source_h5_file = f"{data_folder}{combinable_file}"
        target_h5_file = f"{data_folder}{base_name}.hdf5"
        source_file_size_gb = stat(source_h5_file).st_size / (1024 ** 3)
        try:
            target_file_size_gb = stat(target_h5_file).st_size / (1024 ** 3)
        except FileNotFoundError:
            target_file_size_gb = 0
        print(
            f"Linking {source_h5_file} ({source_file_size_gb:.2f} GB) to {target_h5_file} ({target_file_size_gb:.2f} GB)")

        try:
            with h5py.File(source_h5_file, "r") as fs:

                source_groups = list(fs.keys())
                if len(source_groups) != 1:
                    print(f"Skipping {source_h5_file}, it has {len(source_groups)} groups.")
                    continue
                group_name = source_groups[0]
                full_group_name = fs[group_name].name
            with h5py.File(target_h5_file, "a") as ft:
                # get return None if object is not present
                if isinstance(ft.get(group_name, getlink=True), h5py.ExternalLink):
                    del ft[group_name]
                ft[group_name] = h5py.ExternalLink(source_h5_file, full_group_name)
            # fs.copy(fs, ft, name=source_groups[0], expand_refs=True, expand_soft=True, expand_external=True)
        except OSError as e:
            if "Unable to synchronously" in str(e):
                print(f"Unable to open source file: {source_h5_file}")
            else:
                raise e
    print("-----Finished combining data-----")


if __name__ == "__main__":
    combine_data()
