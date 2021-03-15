"""
Utilities
"""

from typing import Dict, List, Optional

import cf_xarray  # noqa: F401 pylint: disable=W0611
from cf_xarray.utils import parse_cell_methods_attr
from xarray import Dataset


def assign_coordinates_and_measures(
    ds: Dataset, arbitrary_measures: Optional[Dict[str, List[str]]] = None
) -> Dataset:
    """
    Assign coordinates to variables in a dataset

    Parameters
    ----------
    ds: Dataset
        Dataset with variables and CF coordinates to assign.

    arbitrary_measures: dict, optional
        Dictionary mapping arbitrary measures to variables

    Returns
    -------
    Dataset
    """

    # Sort by number of dimensions so we only retain coords/measures
    # with greatest number of matching dimensions
    coords = {}
    for key, value in ds.cf.coordinates.items():
        sort = [len(ds[var].sizes) for var in value]
        value = [coord for _, coord in sorted(zip(sort, value))]
        coords[key] = value[::-1]

    arbitrary_measures = arbitrary_measures if arbitrary_measures else {}
    measures = {**ds.cf.cell_measures, **arbitrary_measures}
    for key, value in dict(measures).items():
        sort = [len(ds[var].sizes) for var in value]
        value = [measure for _, measure in sorted(zip(sort, value))]
        measures[key] = value[::-1]

    for var in ds.variables:

        # Reset coordinates ans measures
        ds[var].attrs.pop("coordinates", None)
        ds[var].encoding.pop("coordinates", None)
        ds[var].attrs.pop("cell_measures", None)

        # Skip coordinates
        if var in sum(coords.values(), []) or var in ds.dims:
            continue

        # Assign coordinates
        for value in coords.values():
            for coord in value:
                if set(ds[coord].dims) <= set(ds[var].dims):
                    ds[var].attrs["coordinates"] = " ".join(
                        sorted(ds[var].attrs.get("coordinates", "").split() + [coord])
                    )
                    break

        # Skip measures
        if var in sum(measures.values(), []):
            continue

        # Assign measures
        for key, value in measures.items():
            for measure in value:
                if set(ds[measure].dims) <= set(ds[var].dims):
                    parsed = parse_cell_methods_attr(
                        ds[var].attrs.get("cell_measures", "")
                    )
                    parsed[key] = measure
                    ds[var].attrs["cell_measures"] = " ".join(
                        [f"{key}: {parsed[key]}" for key in sorted(parsed)]
                    )
                    break

    return ds
