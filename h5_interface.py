import itertools
from enum import StrEnum
from typing import Any

import h5py
import numpy as np


class DatasetParameters(StrEnum):
    NDataDims = "NDataDims"


class H5Saver:
    def __init__(self, group: h5py.Group, parameters_dict_list: dict[str, list]):
        self.group = group
        self.parameters_dict_list = parameters_dict_list
        self.parameters_list_dict = [dict(zip(parameters_dict_list.keys(), paired_parameters))
                                     for paired_parameters in
                                     itertools.product(
                                         *[parameters_dict_list[key] for key in parameters_dict_list.keys()])
                                     ]

        self.non_singular_keys = []
        for key, value in self.parameters_dict_list.items():
            if len(value) == 1:
                self.group.attrs[key] = value[0]
            elif len(value) > 1:
                self.group[key] = value
                # local_group[key].make_scale(key)
                self.non_singular_keys.append(key)
            else:
                raise NotImplementedError
        self.non_singular_indices_list = list(
            itertools.product(*[range(len(parameters_dict_list[key])) for key in self.non_singular_keys]))
        self.parameter_dims = [len(parameters_dict_list[key]) for key in self.non_singular_keys]

    def create_dataset_with_dim_labels(self, dataset_name: str, shape: list[int], data_dim_names: list[str],
                                       dtype: Any = np.float64):
        if len(shape) != len(data_dim_names):
            raise ValueError("shape and data_dim_names must have same length")
        self.group.create_dataset(dataset_name, self.parameter_dims + shape,
                                  dtype=dtype)
        for i, key in enumerate(self.non_singular_keys):
            # local_group[EDParameters.EigenValues].dims[i].attach_scale(local_group[key])
            self.group[dataset_name].dims[i].label = key
        for i, key in enumerate(data_dim_names):
            self.group[dataset_name].dims[len(self.non_singular_keys) + i].label = key
        self.group[dataset_name].attrs[DatasetParameters.NDataDims] = len(shape)
