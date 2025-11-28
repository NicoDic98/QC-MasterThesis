import itertools
from enum import StrEnum
from typing import Any

import h5py
import numpy as np

from labels import VQEParameters


def adapt_dtype_for_h5(value):
    if (np.issubdtype(type(value), np.floating) or np.issubdtype(type(value), np.integer)
            or np.issubdtype(type(value), np.complexfloating) or np.issubdtype(type(value), np.bool)
            or isinstance(value, (np.ndarray, list, tuple))):
        return value
    else:
        return str(value)


def save_dict_as_attribute(group: h5py.Group, my_dict: dict[str, Any], name: str):
    group.create_group(name)
    for key, value in my_dict.items():
        if isinstance(value, dict):
            save_dict_as_attribute(group[name], value, key)
        else:
            group[name].attrs[key] = adapt_dtype_for_h5(value)


def load_attribute_as_dict(group: h5py.Group, recursive: bool = True) -> dict[str, Any]:
    ret = {}
    for key, value in group.attrs.items():
        ret[key] = value
    if recursive:
        for key, value in group.items():
            if isinstance(value, h5py.Group):
                ret[key] = load_attribute_as_dict(value)
    return ret


# todo: method for: load all as dict, use this as a different print option to file, such that subgroups can be folded in


class DatasetParameters(StrEnum):
    NDataDims = "NDataDims"


class H5Saver:
    def __init__(self, group: h5py.Group, parameters_dict_list: dict[str, list], first_open=True):
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
                if first_open:
                    self.group.attrs[key] = value[0]
            elif len(value) > 1:
                if first_open:
                    self.group[key] = value
                # local_group[key].make_scale(key)
                self.non_singular_keys.append(key)
            else:
                raise NotImplementedError
        self.non_singular_indices_list = list(
            itertools.product(*[range(len(parameters_dict_list[key])) for key in self.non_singular_keys]))
        self.parameter_dims = [len(parameters_dict_list[key]) for key in self.non_singular_keys]

    def create_dataset_with_dim_labels(self, dataset_name: str, shape: list[int], data_dim_names: list[str],
                                       dtype: Any = np.float64, maxshape: list[int | None] = None, fillvalue=None):
        """

        :param dataset_name: Name of the dataset to be created
        :param shape: Shape of the dataset for a fixed set of parameters
        :param data_dim_names: Names of the data dimensions
        :param dtype: Data type
        :param maxshape: Maximum data shape
        :param fillvalue: Fill value
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
                                  dtype=dtype, maxshape=maxshape, fillvalue=fillvalue)
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

    def retrieve_dependency(self, parameters: dict[str, int], dependency_names: list[str],
                            final_iteration_value: bool = True):
        """
        :param parameters: A dictionary mapping parameter names to indices in the corresponding list of parameter values
        :param dependency_names: List of dependency names, which should not be fixed to one value
        :param final_iteration_value: If true, return only the value in the final iteration
        :return: Dataset values, Corresponding dependency values, Dependency dictionary {Name: Axis}
        """
        selected_indices = []
        mapping = []
        dependencies = [np.array([])] * len(dependency_names)
        if self.dataset.attrs[DatasetParameters.NDataDims]:
            dataset_dims = list(self.dataset.dims)[:-self.dataset.attrs[DatasetParameters.NDataDims]]
        else:
            dataset_dims = list(self.dataset.dims)
        for dim in dataset_dims:
            if dim.label in parameters.keys():
                selected_indices.append(parameters[dim.label])
            elif dim.label in dependency_names:
                for i, dependency_name in enumerate(dependency_names):
                    if dependency_name == dim.label:
                        # dependencies[i] = np.array(dim[dim.label])
                        dependencies[i] = np.array(self.group[dim.label])
                        mapping.append(i)
                        selected_indices.append(slice(None))
                        break
            else:
                raise ValueError(f"You needed to specify an index for dimension {dim.label}")

        for dependency_name, dependency in zip(dependency_names, dependencies):
            if len(dependency) == 0:
                raise ValueError(f"No dimension is labeled as {dependency_name} dimension")

        # If there is an IterationAxis, move it to the front of the NDataDims
        source = list(range(len(mapping)))
        has_iteration_axis = False

        # The following is necessary such that the source entry for the IterationAxis is correct
        # even when some parameters are fixed
        if self.dataset.attrs[DatasetParameters.NDataDims]:
            temp = list(self.dataset.dims)[-self.dataset.attrs[DatasetParameters.NDataDims]:]
            offset = len(list(self.dataset.dims)) - self.dataset.attrs[DatasetParameters.NDataDims] - len(parameters)
        else:
            temp = []
            offset = 0
        for i, dim in enumerate(temp):
            if dim.label == VQEParameters.IterationAxis:
                if has_iteration_axis:
                    raise ValueError(f"Iteration axis {dim.label} has already been defined")
                has_iteration_axis = True
                source.append(i + offset)
                mapping.append(len(mapping))

        # This moves the axes in the order in which the dependencies were given
        values = np.moveaxis(np.array(self.dataset)[*selected_indices], source, mapping)
        if final_iteration_value and has_iteration_axis:
            temp = [*values.shape]
            temp.pop(mapping[-1])
            final_shape = (*temp,)
            flat_shape = (-1, *values.shape[-self.dataset.attrs[DatasetParameters.NDataDims]:])  # is non 0
            values = values.reshape(flat_shape)

            n_iter_h5_loader = H5Loader(self.group, VQEParameters.NIterations)
            n_iterations, _, _ = n_iter_h5_loader.retrieve_dependency(parameters, dependency_names)

            if list(values.shape)[0] == 1:
                values = values.reshape(flat_shape[1:])
                values = values[n_iterations.flat]
            else:
                values = values[range(len(values)), n_iterations.flat]

            values = values.reshape(final_shape)

        temp = dependency_names.copy()
        for dim in list(self.dataset.dims)[len(dataset_dims): len(self.dataset.dims)]:
            if dim.label == VQEParameters.IterationAxis:
                if not final_iteration_value:
                    temp.insert(mapping[-1], dim.label)
            else:
                temp.append(dim.label)

        dim_labels = {}
        for i, dep_name in enumerate(temp):
            dim_labels[dep_name] = i

        return values, dependencies, dim_labels
