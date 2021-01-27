"""
Utilities
"""

from typing import Any, Dict, Union

import cf_xarray  # noqa: F401 pylint: disable=W0611
from xarray import DataArray, Dataset


def assign_coordinates(ds: Dataset) -> Dataset:
    """
    Assign coordinates to all variables in a Dataset.
    Coordinates have `standard_name`: {'longitude', 'latitude', 'depth', 'time'}

    Parameters
    ----------
    ds: Dataset
        Dataset to wich assign coordinates

    Returns
    -------
    Dataset
    """

    # Make coordinates xarray coordinates
    coords = [
        ds.cf.standard_names.get(coord, [])
        for coord in {"longitude", "latitude", "depth", "time"}
    ]
    ds = ds.set_coords(sum(coords, []))
    coords = sum(ds.cf.coordinates.values(), [])

    for var in set(ds.variables) - set(coords):
        dims = set(ds[var].dims)

        # Assign coordinates attribute
        ds[var].encoding.pop("coordinates", None)
        ds[var].attrs.pop("coordinates", None)
        ds[var].attrs.pop("cell_measures", None)
        coordinates = []
        for coord_name in coords:
            if set(ds[coord_name].dims) <= dims:
                coordinates += [coord_name]
        if coordinates:
            ds[var].attrs["coordinates"] = " ".join(sorted(coordinates))

    return ds


def assign_cell_measures(ds: Dataset) -> Dataset:
    """
    Assign cell measures to all variables in a Dataset.
    Cell measures have `standard_name` starting with `cell_`

    Parameters
    ----------
    ds: Dataset
        Dataset to wich assign cell measures

    Returns
    -------
    Dataset
    """

    measures = {
        name: std.split("cell_", 1)[-1]
        for std, names in ds.cf.standard_names.items()
        if std.startswith("cell_")
        for name in names
    }
    ds = ds.set_coords(list(measures))

    for var, da in ds.data_vars.items():
        dims = set(da.dims)

        # Assign measures
        ds[var].attrs.pop("cell_measures", None)
        cell_measures: Dict[str, str] = {}
        for name, std in measures.items():
            if set(ds[name].dims) <= dims:
                if std not in cell_measures or set(ds[cell_measures[std]].dims) <= set(
                    ds[name].dims
                ):
                    # Assign measures with more dimensions
                    cell_measures[std] = name
        if cell_measures:
            list_str = [": ".join([key, value]) for key, value in cell_measures.items()]
            ds[var].attrs["cell_measures"] = " ".join(sorted(list_str))

    return ds


def assign_cell_measures_and_coordinates(ds: Dataset) -> Dataset:
    """
    Assign coordinates and cell measures to all variables in a Dataset.

    Parameters
    ----------
    ds: Dataset
        Dataset to which assing cell measures and coordinates

    Returns
    -------
    Dataset

    See Also
    --------
    assign_coordinates
    assign_cell_measures
    """

    ds = ds.reset_coords()
    ds = assign_cell_measures(assign_coordinates(ds))
    return ds.cf[ds.variables]


def update_attrs(
    obj: Union[Dataset, DataArray], attrs: Dict[str, Any], override: bool = False
) -> Union[Dataset, DataArray]:
    """
    Update attributes of an object

    Parameters
    ----------
    obj: Dataset, DataArray
        Object to which add attributes
    attrs: dict
        Dictionary mapping variable names to attributes
    override: bool
        Override existing attributes

    Returns
    -------
    Dataset, DataArray
    """

    if isinstance(obj, DataArray) and obj.name in attrs:
        obj.attrs = (
            {**obj.attrs, **attrs[obj.name]}
            if override
            else {**attrs[obj.name], **obj.attrs}
        )
        return obj

    for dim in set(obj.dims).intersection(set(attrs)):
        obj[dim] = obj[dim]
    for var in set(obj.variables).intersection(set(attrs)):
        obj[var].attrs = (
            {**obj[var].attrs, **attrs[var]}
            if override
            else {**attrs[var], **obj[var].attrs}
        )

    return obj
