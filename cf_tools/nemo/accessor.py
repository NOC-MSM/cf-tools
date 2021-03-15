"""
NEMO-specific compute functions
"""

import cf_xarray  # noqa: F401 pylint: disable=W0611
import xarray as xr
from xarray import DataArray

from ..accessor import Accessor


@xr.register_dataset_accessor("nemo_tools")
class NemoAccessor(Accessor):
    """
    Child class for NEMO
    """

    @property
    def sea_floor_depth_below_geoid(self) -> DataArray:
        """
        Compute sea_floor_depth_below_geoid if missing

        Returns
        -------
        DataArray
        """

        if "sea_floor_depth_below_geoid" in self._obj.cf:
            return self._obj["sea_floor_depth_below_geoid"]

        # Variables
        thickness = self._obj.cf["e3t_0"]
        level = self._obj.cf["model_level_number_at_sea_floor"]

        # Fortran to python level
        pylevel = xr.where(level > 0, level - 1, 0)

        # Compute
        cumsum = thickness.cf.cumsum("Z")
        bathy = cumsum.cf.isel(Z=pylevel.compute()).where(level > 0)

        # Attributes
        bathy.attrs["standard_name"] = "sea_floor_depth_below_geoid"
        bathy.attrs["long_name"] = "Bathymetry"
        bathy.attrs["units"] = "m"
        bathy.attrs["history"] = "Computed offline"

        return bathy
