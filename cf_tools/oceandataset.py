"""
OceanDataset
"""
import warnings
from typing import Any, Dict, Union

import cf_xarray  # noqa: F401 pylint: disable=W0611
import xarray as xr
from xarray import DataArray, Dataset
from xgcm import Grid
from xgcm.comodo import get_axis_positions_and_coords

from .utils import assign_cell_measures_and_coordinates

_VALID_OPTIONS = dict(
    periodic="See xgcm.Grid",
)


class OceanDataset:
    """
    Class to work with general circulation model output
    """

    def __init__(self, dataset: Dataset, add_missing: bool = False):
        """
        Class to work with general circulation model output

        Parameters
        ----------
        dataset: Dataset
            Dataset with all variables
        add_missing: bool
            Whether to add missing coordinates and cell_measures
        """

        self._dataset = dataset
        self._options: Dict[str, Any] = {}
        self._initialize_options()

        if add_missing:
            self.missing_areas(add=True)
            self.missing_depths_at_rest(add=True)

    def _initialize_options(self):

        pass

    @property
    def dataset(self) -> Dataset:
        """
        CF-Compliant Dataset

        Returns
        -------
        Dataset
        """

        return assign_cell_measures_and_coordinates(self._dataset)

    @property
    def domain(self) -> Dataset:
        """
        CF-Compliant Dataset with timeless variables

        Returns
        -------
        Dataset
        """

        ds = self.dataset
        domain = ds.cf[[var for var in ds.data_vars if "T" not in ds.cf[var].cf.axes]]

        return assign_cell_measures_and_coordinates(domain)

    @property
    def options(self) -> Dict[str, Any]:
        """
        Dictionary mapping set options to their values

        Returns
        -------
        dict
        """

        return self._options

    @property
    def valid_options(self) -> Dict[str, str]:
        """
        Dictionary mapping valid options to their descriptions

        Returns
        -------
        dict
        """

        return _VALID_OPTIONS

    def set_options(self, **kwargs: Any):
        """
        Set options

        Parameters
        ----------
        kwargs: Any
            Options to set

        Raises
        ------
        ValueError
            If the option is not valid
        """

        # Error
        diff_options = set(kwargs) - set(self.valid_options)
        if diff_options:
            raise ValueError(
                "\n".join(
                    [
                        f"{diff_options} are not valid options.",
                        f"Valid options are {self.valid_options}.",
                    ]
                )
            )

        # Warning
        for key, value in kwargs.items():
            if key not in self.options or value != self.options[key]:
                warnings.warn("Setting option {}".format({key: value}))

        self._options = {**self._options, **kwargs}

    def _get_options(self, *args: str) -> Dict[str, Any]:

        options = {
            option: self.options[option] for option in args if option in self.options
        }

        missing = set(args) - set(options)
        if missing:
            raise ValueError(
                f"Use self.set_options() to set the following options: {missing}."
            )

        return options

    @property
    def grid(self) -> Grid:
        """
        xgcm Grid object

        Returns
        -------
        Grid
        """

        ds = self.dataset
        kwargs = self._get_options("periodic")
        return Grid(ds, **kwargs)

    def _seprate_arakawa(self, ds: Dataset) -> Dict[str, Dataset]:

        datasets = {}
        for grid, arakawa_axes in self._arakawa_mapper.items():
            all_axes = {key: set([value]) for key, value in arakawa_axes.items()}
            hor_axes = dict(all_axes)
            hor_axes.pop("Z")
            arakawa_vars = []
            for var in ds.variables:
                var_axes = {
                    key: set(value) for key, value in ds.cf[var].cf.axes.items()
                }
                var_axes.pop("T", None)
                if not var_axes:
                    # Time only
                    arakawa_vars += [var]
                elif all_axes == var_axes:
                    # Matching X, Y, and Z
                    arakawa_vars += [var]
                elif "w" not in grid and hor_axes == var_axes:
                    # 2D, matching X and Y
                    arakawa_vars += [var]
            datasets[grid] = ds[arakawa_vars]

        return {
            grid: assign_cell_measures_and_coordinates(ds)
            for grid, ds in datasets.items()
        }

    @property
    def arakawa_datasets(self) -> Dict[str, Dataset]:
        """
        Dictionary mapping arakawa grids to datasets with all variables
        on that grid.

        Returns
        -------
        dict
        """

        return self._seprate_arakawa(self.dataset)

    @property
    def arakawa_domains(self) -> Dict[str, Dataset]:
        """
        Dictionary mapping arakawa grids to datasets with all domain variables
        on that grid.

        Returns
        -------
        dict
        """

        return self._seprate_arakawa(self.domain)

    @property
    def _arakawa_mapper(self) -> Dict[str, Dict[str, str]]:

        ds = self.dataset

        x_pos = get_axis_positions_and_coords(ds, "X")
        assert len(x_pos) == 2
        y_pos = get_axis_positions_and_coords(ds, "Y")
        assert len(y_pos) == 2
        z_pos = get_axis_positions_and_coords(ds, "Z")
        assert len(z_pos) == 2

        mapper: Dict[str, Dict[str, str]] = {
            grid: {} for grid in {"t", "u", "v", "f", "w", "uw", "vw", "fw"}
        }
        for grid in mapper.keys():
            # X
            if not any(hgrid in grid for hgrid in {"u", "f"}):
                mapper[grid]["X"] = x_pos["center"]
            else:
                mapper[grid]["X"] = x_pos[list(set(x_pos) - {"center"})[0]]

            # Y
            if not any(hgrid in grid for hgrid in {"v", "f"}):
                mapper[grid]["Y"] = y_pos["center"]
            else:
                mapper[grid]["Y"] = y_pos[list(set(y_pos) - {"center"})[0]]

            # Z
            if not grid.endswith("w"):
                mapper[grid]["Z"] = z_pos["center"]
            else:
                mapper[grid]["Z"] = z_pos[list(set(z_pos) - {"center"})[0]]

        return mapper

    def add(self, obj: Union[Dataset, DataArray], **kwargs: Any):
        """
        Add variables to the dataset.

        Parameters
        ----------
        obj: Dataset, DataArray
            Object to add
        **kwargs: Any
            Additional arguments to pass into `xarray.merge`
        """

        ds = xr.merge([self.dataset.reset_coords(), obj.reset_coords()], **kwargs)
        self._dataset = ds

    def missing_areas(self, prefix: str = "area", add: bool = False) -> Dataset:
        """
        Compute missing areas.

        Parameters
        ----------
        prefix: str
            Prefix for the DataArrays
        add: bool
            Whether to add missing variables to the dataset

        Returns
        -------
        Dataset
        """

        areas = []
        for pos, ds in self.arakawa_datasets.items():
            if pos.endswith("w") or "area" in ds.cf.cell_measures:
                continue
            area = ds.cf["xdist"] * ds.cf["ydist"]
            area.attrs["standard_name"] = "cell_area"
            area.attrs["units"] = "m2"
            area.attrs["history"] = "Computed offline"
            areas += [area.rename(prefix + pos)]
        areas = xr.merge(areas)

        return self._maybe_add_and_return(areas, add)

    def missing_depths_at_rest(
        self, prefix: str = "depth0", add: bool = False
    ) -> Dataset:
        """
        Compute missing depths at rest.

        Parameters
        ----------
        prefix: str
            Prefix for the DataArrays
        add: bool
            Whether to add missing variables to the dataset

        Returns
        -------
        Dataset
        """

        map_out_in = dict(t="w", u="uw", v="vw", w="t", uw="u", vw="v", f="fw", fw="f")
        arakawa_domains = self.arakawa_domains
        grid = self.grid

        depths = []
        for pos_out, ds_out in arakawa_domains.items():
            if "depth" not in ds_out.cf.standard_names:
                pos_in = map_out_in[pos_out]

                # If missing, use t or w and interpolate
                if "cell_thickness" not in arakawa_domains[pos_in].cf.standard_names:
                    pos_in = "w" if "w" in pos_in else "t"
                    thickness = arakawa_domains[pos_in].cf["cell_thickness"]
                    thickness = grid.interp(
                        thickness,
                        [
                            axis
                            for axis in {"X", "Y"}
                            if thickness.cf.axes[axis] != ds_out.cf.axes[axis]
                        ],
                        boundary="extend",
                    )
                else:
                    thickness = arakawa_domains[pos_in].cf["cell_thickness"]

                # Surface at t is half of w
                if "w" not in pos_out:
                    thickness = thickness.where(thickness.cf["Z"] > 0, 0.5 * thickness)

                # Set up attributes
                depth = grid.cumsum(thickness, "Z", boundary="fill", fill_value=0)
                depth.attrs["standard_name"] = "depth"
                depth.attrs["units"] = "m"
                depth.attrs["history"] = "Computed offline"
                depths += [depth.rename(prefix + pos_out)]
        depths = xr.merge(depths)

        return self._maybe_add_and_return(depths, add)

    def _maybe_add_and_return(
        self, obj: Union[Dataset, DataArray], add: bool
    ) -> Union[Dataset, DataArray]:

        if add:
            self.add(obj)

        return assign_cell_measures_and_coordinates(obj)
