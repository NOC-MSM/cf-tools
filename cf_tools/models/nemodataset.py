"""
NemoDataset:
Representation of NEMO model output.
Built on top of OceanDataset.
"""
# pylint: disable=C0116

import warnings
from typing import Any, Dict, Optional, Set, Union

import cf_xarray  # noqa: F401 pylint: disable=W0611
import xarray as xr
from xarray import DataArray, Dataset

from ..oceandataset import OceanDataset
from ..utils import assign_cell_measures_and_coordinates, update_attrs

# Dimensions to rename
_GRID_DIMS: Dict[str, Dict[str, Set[str]]] = dict(
    t=dict(x_c={"x"}, y_c={"y"}, z_c={"z", "nav_lev", "deptht"}),
    u=dict(x_r={"x"}, y_c={"y"}, z_c={"z", "nav_lev", "depthu"}),
    v=dict(x_c={"x"}, y_r={"y"}, z_c={"z", "nav_lev", "depthv"}),
    f=dict(x_r={"x"}, y_r={"y"}, z_c={"z", "nav_lev", "depthf"}),
    w=dict(x_c={"x"}, y_c={"y"}, z_l={"z", "nav_lev", "depthw"}),
    uw=dict(x_r={"x"}, y_c={"y"}, z_l={"z", "nav_lev", "depthuw"}),
    vw=dict(x_c={"x"}, y_r={"y"}, z_l={"z", "nav_lev", "depthvw"}),
    fw=dict(x_r={"x"}, y_r={"y"}, z_l={"z", "nav_lev", "depthfw"}),
)

# Variables to place on the appropriave arakawa grid
_DOMAIN_VARS: Dict[str, Set[str]] = dict(
    # t is default
    u={"glamu", "gphiu", "gdepu_0", "e1u", "e2u", "e3u_0", "umask", "umaskutil"},
    v={"glamv", "gphiv", "gdepv_0", "e1v", "e2v", "e3v_0", "vmask", "vmaskutil"},
    f={
        "glamf",
        "gphif",
        "gdepf_0",
        "e1f",
        "e2f",
        "e3f_0",
        "ff_f",
        "fmask",
        "fmaskutil",
    },
    w={"gdepw_1d", "gdepw_0", "e3w_1d", "e3w_0"},
    uw={"gdepuw_0", "e3uw_0"},
    vw={"gdepvw_0", "e3vw_0"},
    fw=set(),
)

# Missing CF-attributes
_ATTRS: Dict[str, Dict[str, Any]] = dict(
    # Axes
    x_c=dict(axis="X"),
    x_r=dict(axis="X", c_grid_axis_shift=0.5),
    y_c=dict(axis="Y"),
    y_r=dict(axis="Y", c_grid_axis_shift=0.5),
    z_c=dict(axis="Z"),
    z_l=dict(axis="Z", c_grid_axis_shift=-0.5),
    # Coordinates
    t=dict(standard_name="time"),
    time_counter=dict(standard_name="time"),
    glamt=dict(standard_name="longitude", units="degree_east"),
    glamu=dict(standard_name="longitude", units="degree_east"),
    glamv=dict(standard_name="longitude", units="degree_east"),
    glamf=dict(standard_name="longitude", units="degree_east"),
    gphit=dict(standard_name="latitude", units="degree_north"),
    gphiu=dict(standard_name="latitude", units="degree_north"),
    gphiv=dict(standard_name="latitude", units="degree_north"),
    gphif=dict(standard_name="latitude", units="degree_north"),
    gdept_0=dict(standard_name="depth", units="m"),
    gdepu_0=dict(standard_name="depth", units="m"),
    gdepv_0=dict(standard_name="depth", units="m"),
    gdepf_0=dict(standard_name="depth", units="m"),
    gdepw_0=dict(standard_name="depth", units="m"),
    gdepuw_0=dict(standard_name="depth", units="m"),
    gdepvw_0=dict(standard_name="depth", units="m"),
    # Cell measures
    e1t=dict(standard_name="cell_xdist", units="m"),
    e1u=dict(standard_name="cell_xdist", units="m"),
    e1v=dict(standard_name="cell_xdist", units="m"),
    e1f=dict(standard_name="cell_xdist", units="m"),
    e2t=dict(standard_name="cell_ydist", units="m"),
    e2u=dict(standard_name="cell_ydist", units="m"),
    e2v=dict(standard_name="cell_ydist", units="m"),
    e2f=dict(standard_name="cell_ydist", units="m"),
    e3t_0=dict(standard_name="cell_thickness", units="m"),
    e3u_0=dict(standard_name="cell_thickness", units="m"),
    e3v_0=dict(standard_name="cell_thickness", units="m"),
    e3f_0=dict(standard_name="cell_thickness", units="m"),
    e3w_0=dict(standard_name="cell_thickness", units="m"),
    e3uw_0=dict(standard_name="cell_thickness", units="m"),
    e3vw_0=dict(standard_name="cell_thickness", units="m"),
    areat=dict(standard_name="cell_area", units="m2"),
    areau=dict(standard_name="cell_area", units="m2"),
    areav=dict(standard_name="cell_area", units="m2"),
    areaf=dict(standard_name="cell_area", units="m2"),
    # Variables
    ff_t=dict(standard_name="coriolis_parameter", units="s-1"),
    ff_f=dict(standard_name="coriolis_parameter", units="s-1"),
    bottom_level=dict(standard_name="model_level_number_at_sea_floor", units="1"),
    top_level=dict(standard_name="model_level_number_at_sea_surface", units="1"),
    bathy_metry=dict(standard_name="sea_floor_depth", units="m"),
    mbathy=dict(standard_name="sea_floor_depth", units="m"),
    tmask=dict(standard_name="sea_binary_mask", units="1"),
    umask=dict(standard_name="sea_binary_mask", units="1"),
    vmask=dict(standard_name="sea_binary_mask", units="1"),
    fmask=dict(standard_name="sea_binary_mask", units="1"),
    tmaskutil=dict(standard_name="sea_binary_mask", units="1"),
    umaskutil=dict(standard_name="sea_binary_mask", units="1"),
    vmasktuil=dict(standard_name="sea_binary_mask", units="1"),
    fmaskutil=dict(standard_name="sea_binary_mask", units="1"),
)


class NemoDataset(OceanDataset):
    """
    Class to work with NEMO model output
    """

    def __init__(
        self,
        domain_or_dataset: Dataset,
        output: Optional[Dict[str, Dataset]] = None,
        add_missing: bool = False,
    ):
        """
        Class to work with NEMO model output

        Parameters
        ----------
        domain_or_dataset: Dataset
            Either NEMO domain_cfg/mesh_mask dataset, or a CF-compliant NEMO dataset
        output: dict
            Dictionary mapping grids to NEMO output datasets
            Supported grids: {t, u, v, f, w, uw, vw, fw}
        add_missing: bool
            Whether to add missing coordinates and cell_measures
        """

        if not output:
            self._dataset = domain_or_dataset
        else:
            self._dataset = None
            self._domain = domain_or_dataset
            self._output = output

        super().__init__(self.dataset, add_missing=add_missing)

    def _initialize_options(self):

        self._options = {}

        # Periodic
        jperio = self.dataset.get("jperio")
        if jperio is not None:
            jperio = int(jperio.values)
            if jperio == 0:
                periodic = False
            elif jperio == 1:
                periodic = ["X"]
            elif jperio == 2:
                periodic = ["Y"]
            elif jperio == 7:
                periodic = ["X", "Y"]
            else:
                periodic = None
            if periodic is not None:
                self.set_options(periodic=periodic)

    @property
    def dataset(self) -> Dataset:
        """
        CF-compliant dataset

        Returns
        -------
        Dataset
        """

        if not self._dataset:

            self._manipulate_domain()
            self._manipulate_output()
            self._merge_domain_and_output()
            self._replace_time_counter()

        return self._dataset

    def _manipulate_domain(self):

        domain = self._domain
        # Update attributes
        domain = update_attrs(domain, _ATTRS)
        # Squeeze and drop time
        domain = domain.squeeze(domain.cf.axes.get("T", []), drop=True)
        domain = domain.cf.drop(domain.cf.standard_names.get("time", []))
        # Place domain on Arakawa grids
        for var in domain.variables:
            da = domain[var]
            pos = "t"  # default
            for key, value in _DOMAIN_VARS.items():
                pos = key if var in value else pos
            domain[var] = _place_on_grid(da, pos)
        # Add axes and attributes
        domain = domain.assign_coords(
            {
                dim: update_attrs(domain[dim], _ATTRS)
                for dim in domain.dims
                if dim in _ATTRS
            }
        )

        self._domain = domain.reset_coords()

    def _manipulate_output(self):

        output = self._output

        # Loop over grids
        for pos, ds in output.items():
            # Rename area
            if "area" in ds.variables:
                if "w" not in pos:
                    rename = dict(area="area" + pos)
                    warnings.warn(f"Renaming {rename}".replace(":", " to"))
                    ds = ds.rename(rename)
                else:
                    warnings.warn(f"Dropping [area] on grid [{pos}]")
                    ds = ds.drop_vars("area")
            # Drop space coordinates and variables in domain
            ds = assign_cell_measures_and_coordinates(update_attrs(ds, _ATTRS))
            vars2drop = list(set(self._domain.variables).intersection(ds.variables))
            vars2drop += list(
                ds.cf[
                    [coord for coord in ds.cf.coordinates if coord != "time"]
                ].variables
            )
            ds = ds.drop_vars(list(set(vars2drop)))
            # Place on grid
            ds = _place_on_grid(ds, pos)
            output[pos] = ds.reset_coords()

        self._output = output

    def _merge_domain_and_output(self):

        ds = xr.merge([self._domain] + list(self._output.values()))
        self._dataset = assign_cell_measures_and_coordinates(ds)
        del self._domain, self._output

    def _replace_time_counter(self):

        ds = self._dataset

        # Replace time counter with the idendical time_centered or instant
        if "time_counter" in ds.dims and "time_counter" in ds.cf.axes.get("T", []):
            counter = ds["time_counter"]
            for time in set(ds.cf.coordinates["time"]) - {"time_counter"}:
                if counter.equals(ds[time]):
                    ds = ds.swap_dims({"time_counter": time})
                    bounds = counter.attrs.get("bounds")
                    ds = ds.drop_vars(
                        ["time_counter", bounds] if bounds else "time_counter"
                    )

        # Assign axis and shift
        time_dim = [time for time in ds.cf.axes.get("T", []) if time in ds.dims]
        if len(time_dim) == 1:
            time_dim = time_dim[0]
            ds[time_dim].attrs["axis"] = "T"
            ds[time_dim].attrs.pop("c_grid_axis_shift", None)
            for time in set(ds.cf.axes["T"]) - {time_dim}:
                bounds = ds[time].attrs.get("bounds")
                if bounds:
                    ds[bounds] = (
                        ds[bounds].swap_dims({time_dim: time}).reset_coords(drop=True)
                    )
                else:
                    ds[time] = (
                        ds[time].swap_dims({time_dim: time}).reset_coords(drop=True)
                    )
                ds[time].attrs["axis"] = "T"
                if time_dim.endswith("_instant") and time.endswith("_centered"):
                    ds[time].attrs["c_grid_axis_shift"] = -0.5
                elif time_dim.endswith("_centered") and time.endswith("_instant"):
                    ds[time].attrs["c_grid_axis_shift"] = 0.5

        self._dataset = assign_cell_measures_and_coordinates(ds)


def _place_on_grid(
    obj: Union[Dataset, DataArray], grid: str
) -> Union[Dataset, DataArray]:

    grid = grid.lower()
    assert (
        grid in _GRID_DIMS
    ), f"Grid [{grid}] not supported. Supported grid: {set(_GRID_DIMS)}"

    rename_dict = {
        value: key
        for key, values in _GRID_DIMS[grid].items()
        for value in values
        if value in obj.dims
    }

    # Dataset
    if isinstance(obj, Dataset):
        return obj.rename_dims(rename_dict)

    # DataArray
    var = obj.name
    obj = Dataset({var: obj})
    obj = obj.rename_dims(rename_dict)
    return obj[var]
