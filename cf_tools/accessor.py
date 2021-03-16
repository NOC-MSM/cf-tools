"""
Accessor for all models
"""
from typing import Any, Dict, Optional, Union

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
        self._options: Optional[Dict[str, Any]] = None
        self._old_options: Optional[Dict[str, Any]] = None

    def __enter__(self):
        self._old_options = self.options

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._options = self._old_options

    def set_options(self, **kwargs: Any):
        """
        Set options

        Parameters
        ----------
        **kwargs: Any
            Options to set
        """

        self._options = {**self._options, **kwargs} if self._options else kwargs

    def unset_options(self, *args: str):
        """
        Options to unset

        Parameters
        ----------
        *args: str
            Options to unset
        """

        if self._options:
            for arg in args:
                self._options.pop(arg, None)

    @property
    def options(self) -> Dict[str, Any]:
        """
        Options currently set

        Returns
        -------
        dict:
            Options currently set
        """

        return self._options if self._options else dict()
