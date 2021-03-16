"""
NEMO-specific compute functions
"""

import cf_xarray  # noqa: F401 pylint: disable=W0611
import xarray as xr
from xarray import DataArray, Dataset
from xgcm import Grid

from ..accessor import Accessor


@xr.register_dataset_accessor("nemo_tools")
class NemoAccessor(Accessor):
    """
    Child class for NEMO
    """

    @property
    def sea_floor_depth_below_geoid(self) -> DataArray:
        """
        Retrun sea_floor_depth_below_geoid computing it if missing

        Returns
        -------
        DataArray
        """

        if "sea_floor_depth_below_geoid" in self._obj.cf:
            return self._obj.cf["sea_floor_depth_below_geoid"]

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

    @property
    def vertical(self) -> Dataset:
        """
        Return vertical coordinates computing them if missing

        Returns
        -------
        Dataset
            Dataset with vertical coordinates
        """

        obj = self._obj
        attrs = dict(
            standard_name="depth",
            long_name="Depth",
            units="m",
            positive="down",
            history="Computed offline",
        )
        grid = self.grid(error=False)

        for prefix in {"gdept", "gdepw"}:
            for suffix in {"_0", "_1d"}:
                # Pick depth and thickness
                depth_name = prefix + suffix
                thick_name = f"e3{'t' if prefix.endswith('w') else 'w'}{suffix}"
                # Compute
                if depth_name not in obj.cf.coordinates.get("vertical", []):
                    da = self._obj.cf[thick_name]
                    # First of t is half of w thickness
                    if prefix.endswith("t"):
                        da = da.where(da.cf["Z"] != da.cf["Z"].cf.isel(Z=0), da / 2)
                    da = grid.cumsum(da, "Z", boundary="fill", fill_value=0)
                    da.attrs = attrs
                    obj = obj.assign_coords({depth_name: da})

        return obj.cf[["vertical"]]

    def grid(self, error: bool = True) -> Grid:
        """
        Create xgcm Grid object

        Parameters
        ----------
        error: bool
            Raise error if options must be set

        Returns
        -------
        Grid:
            xgcm Grid object

        Raises
        ------
        ValueError
            If periodic is not set or can't be inferred
        """

        # Periodic
        periodic = self.options.get("periodic")
        if periodic is None and "jperio" in self._obj.variables:
            jperio = self._obj["jperio"]
            if jperio == 0:
                periodic = False
            elif jperio == 1:
                periodic = ["X"]
            elif jperio == 2:
                periodic = ["Y"]
            elif jperio == 7:
                periodic = ["X", "Y"]
        if periodic is None and error:
            raise ValueError(
                "`periodic` can't be inferred "
                "and must be set using `.nemo_tools.set_options`. "
                "See: https://xgcm.readthedocs.io/en/latest/api.html#grid"
            )

        # Metrics
        names_dict = dict(x_spacing="X", y_spacing="Y", thickness="Z")
        metrics = {}
        for cf_name, xgcm_name in names_dict.items():
            metric = self._obj.cf.cell_measures.get(cf_name)
            if metric:
                metrics[xgcm_name] = metric

        # Kwargs
        kwargs = dict()
        if periodic is not None:
            kwargs["periodic"] = periodic
        kwargs["metrics"] = metrics

        return Grid(self._obj, **kwargs)
