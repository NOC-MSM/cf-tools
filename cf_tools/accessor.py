"""
Accessor for all models
"""
from typing import Union

import cf_xarray  # noqa: F401 pylint: disable=W0611
import xarray as xr
from xarray import DataArray, Dataset


@xr.register_dataset_accessor("cf_tools")
class Accessor:
    """
    Parent class for all models
    """

    def __init__(self, xarray_obj: Union[Dataset, DataArray]):
        """
        Initialize class

        Parameters
        ----------
        xarray_obj: Dataset, DataArray
            xarray object to access
        """

        self._obj = xarray_obj

    def _get_or_compute(self, varname):

        return hasattr(self, varname)
