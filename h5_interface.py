import itertools
from enum import StrEnum
from typing import Any

import h5py
import numpy as np


class DatasetParameters(StrEnum):
    NDataDims = "NDataDims"


class H5Saver:
    def __init__(self, group: h5py.Group, parameters_dict_list: dict[str, list]):
        """

        :param group: h5py Group to save under
        :param parameters_dict_list: A dictionary mapping parameter names to lists of parameter values
        """
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
                                       dtype: Any = np.float64, maxshape: list[int] = None):
        """

        :param dataset_name: Name of the dataset to be created
        :param shape: Shape of the dataset for a fixed set of parameters
        :param data_dim_names: Names of the data dimensions
        :param dtype: Data type
        :param maxshape: Maximum data shape
        :return: None
        """
        if np.issubdtype(dtype, np.floating):
            dtype = np.float64
        elif np.issubdtype(dtype, np.integer):
            dtype = np.int64
        elif np.issubdtype(dtype, np.complexfloating):
            dtype = np.complex128
        elif np.issubdtype(dtype, np.bool):
            dtype = np.bool
        else:
            dtype = h5py.string_dtype()

        if maxshape is None:
            pass
        else:
            if len(maxshape) != len(data_dim_names):
                raise ValueError("maxshape and data_dim_names must have same length")
            maxshape = self.parameter_dims + maxshape

        if len(shape) != len(data_dim_names):
            raise ValueError("shape and data_dim_names must have same length")
        self.group.create_dataset(dataset_name, self.parameter_dims + shape,
                                  dtype=dtype, maxshape=maxshape)
        for i, key in enumerate(self.non_singular_keys):
            # local_group[EDParameters.EigenValues].dims[i].attach_scale(local_group[key])
            self.group[dataset_name].dims[i].label = key
        for i, key in enumerate(data_dim_names):
            self.group[dataset_name].dims[len(self.non_singular_keys) + i].label = key
        self.group[dataset_name].attrs[DatasetParameters.NDataDims] = len(shape)


class H5Loader:
    def __init__(self, group: h5py.Group, dataset_name: str):
        """

        :param group: h5py Group to load under
        :param dataset_name: Name of the dataset to be loaded
        """
        self.group = group
        self.dataset = group[dataset_name]

    def retrieve_dependency(self, parameters: dict[str, int], dependency_names: list[str]):
        """
        :param parameters: A dictionary mapping parameter names to indices in the corresponding list of parameter values
        :param dependency_names: List of dependency names, which should not be fixed to one value
        :return: Dataset values, Corresponding dependency values
        """
        selected_indices = []
        mapping = []
        dependencies = [np.array([])] * len(dependency_names)
        for dim in list(self.dataset.dims)[:-self.dataset.attrs[DatasetParameters.NDataDims]]:
            if dim.label in parameters.keys():
                selected_indices.append(parameters[dim.label])
            elif dim.label in dependency_names:
                for i, dependency_name in enumerate(dependency_names):
                    if dependency_name == dim.label:
                        # dependencies[i] = np.array(dim[dim.label])
                        dependencies[i] = np.array(self.group[dim.label])
                        mapping.append(i)
                        selected_indices.append(slice(len(dependencies[i])))
                        break
            else:
                raise ValueError(f"You needed to specify an index for dimension {dim.label}")

        for dependency_name, dependency in zip(dependency_names, dependencies):
            if len(dependency) == 0:
                raise ValueError(f"No dimension is labeled as {dependency_name} dimension")
        # This moves the axes in the order in which the dependencies were given
        values = np.moveaxis(np.array(self.dataset)[*selected_indices], list(range(len(mapping))), mapping)
        return values, dependencies
