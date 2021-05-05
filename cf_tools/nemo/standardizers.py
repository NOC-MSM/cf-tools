"""
Make nemo files CF and COMODO compliant.
"""

import warnings
from datetime import timedelta
from typing import Dict, List, Optional

import cf_xarray  # noqa: F401 pylint: disable=W0611
import numpy as np
import xarray as xr
from xarray import Dataset
from xgcm.autogenerate import generate_grid_ds

from ..utils import assign_coordinates_and_measures
from .accessor import NemoAccessor  # noqa: F401 pylint: disable=W0611


def standardize_domain(ds: Dataset, add_missing_coords: bool = False) -> Dataset:
    """
    Make NEMO mesh_mask or domain_cfg CF and COMODO compliant.

    Parameters
    ----------
    ds: Dataset
        xarray representation of a mesh_mask or domain_cfg
    add_missing_coords: bool
        Add missing coordinates, such as vertical

    Returns
    -------
    Dataset
    """

    # Drop nav_*
    ds = ds.drop_vars([var for var in ds.variables if var.startswith("nav_")])
    if "nav_lev" in ds.dims:
        ds = ds.rename_dims(nav_lev="z")

    # Guess
    ds = ds.assign_coords({dim: ds[dim] for dim in ds.dims})
    ds = ds.set_coords(ds.variables)
    ds = ds.cf.guess_coord_axis()

    # Squeeze
    ds = ds.cf.squeeze([key for key in ["time", "T"] if key in ds.cf], drop=True)

    # z is axes, others are vertical coordinates
    for var in set(ds.cf.axes.get("Z", [])):
        if not var.startswith("z"):
            ds[var].attrs.setdefault("standard_name", "depth")
            ds[var].attrs.pop("axis")

    # COMODO attributes
    if all(len(value) == 1 for value in ds.cf.axes.values()):
        # Generate missing dimensions
        ds = generate_grid_ds(
            ds,
            axes_dims_dict={key: value[0] for key, value in ds.cf.axes.items()},
            position={
                key: ("center", "left" if key == "Z" else "right") for key in ds.cf.axes
            },
        )
        # Rename dimensions
        ds = _place_on_grid(ds)

    # Add missing attributes
    ds = _add_missing_attrs(ds)

    # Add missing coordinates
    if add_missing_coords and "vertical" not in ds.cf:
        ds = ds.merge(ds.nemo_tools.vertical)

    # Assign coordinates and cell_measures
    ds = assign_coordinates_and_measures(
        ds, arbitrary_measures=_find_arbitrary_measures(ds)
    )

    return ds


def standardize_output(
    ds: Dataset,
    domain: Dataset,
    hgrid: Optional[str] = None,
    add_missing_coords: bool = False,
) -> Dataset:
    """
    Make NEMO output CF and COMODO compliant.

    Parameters
    ----------
    ds: Dataset
        xarray representation of NEMO output on a single grid

    domain: Dataset
        xarray representation of a mesh_mask or domain_cfg

    hgrid: str, optional
        Horizontal grid. Options: {"T", "U", "V", "F"}

    add_missing_coords: bool
        Add missing coordinates, such as vertical

    Returns
    -------
    Dataset

    Raises
    ------
    ValueError
        If output dataset contains multiple depth dimensions,
        or hgrid=None and the horizontal grid cannot be inferred (e.g., 2D output).
    """

    if not any(
        dim.endswith(suffix) for dim in ds.dims for suffix in {"_left", "_right"}
    ):
        rename_dict: Dict[str, str] = {}
        hgrid = hgrid.upper() if hgrid else ""

        # Drop nav_*
        ds = ds.drop_vars(
            [
                var
                for var in ds.variables
                for prefix in ["bounds_nav_", "nav_"]
                if var.startswith(prefix)
            ]
        )

        # Find z dim
        zdims = [dim for dim in ds.dims if dim.startswith("depth")]

        # Checks
        error = None
        if not zdims and not hgrid:
            error = "The horizontal grid cannot be inferred. Please provide `hgrid`."
        elif len(zdims) > 1:
            error = "The dataset contains multiple depth dimensions."
        if error:
            raise ValueError(error)

        # Vertical
        if zdims:
            zdim = zdims[0]
            ds = ds.drop_vars([zdim, zdim + "_bounds"], errors="ignore")
            rename_dict[zdim] = "z_left" if zdim[-1] == "w" else "z"
            if not hgrid:
                hgrid = zdim[-1].upper() if zdim[-1] != "w" else "T"

        # Horizontal
        hgrids = ["T", "U", "V", "F"]
        if hgrid not in hgrids:
            raise ValueError(
                f"{hgrid} is not a valid horizontal grid. Available options: {hgrids}"
            )
        rename_dict["x"] = "x_right" if hgrid in ["U", "F"] else "x"
        rename_dict["y"] = "y_right" if hgrid in ["V", "F"] else "y"

        # Rename area
        if "area" in ds.variables:
            ds = ds.rename(area="area" + hgrid.lower())
            for var, variable in ds.variables.items():
                if "cell_measures" in variable.attrs:
                    variable.attrs["cell_measures"] = variable.attrs[
                        "cell_measures"
                    ].replace("area: area", "area: area" + hgrid.lower())

        # Rename
        ds = ds.rename(rename_dict)

        # Get rid of time_counter
        ds = _swap_time(ds)

    # Merge with domain
    domain = standardize_domain(domain, add_missing_coords=add_missing_coords)
    ds = xr.merge([domain, ds], compat="override")

    # Assign coordinates and cell_measures
    ds = assign_coordinates_and_measures(
        ds, arbitrary_measures=_find_arbitrary_measures(ds)
    )

    return ds


def _add_missing_attrs(ds: Dataset) -> Dataset:
    # pylint: disable=R0912, R0915

    coord_suffixes = {"", "_right", "_left", "_outer", "_inner", "_center"}

    namflags = {
        "jperio": "Domain boundary condition",
        "jpiglo": "Size of the domain for the i direction",
        "jpjglo": "Size of the domain for the j direction",
        "jpkglo": "Size of the domain for the k direction",
        "ln_isfcav": "Ice shelf cavities flag",
        "ln_sco": "s-coordinate flag",
        "ln_zco": "z-coordinate with full step bathymetry flag",
        "ln_zps": "z-coordinate with partial step bathymetry flag",
        "ORCA": "Name of the configuration (1: ORCA)",
        "ORCA_index": "Nominal resolution",
    }

    unrecognized = set()
    for var, variable in ds.variables.items():

        # NEMO flags
        if var in namflags:
            variable.attrs.setdefault("long_name", namflags[var])

        # Coordinates
        elif var in ["x" + suffix for suffix in coord_suffixes]:
            variable.attrs.setdefault("long_name", "x-dimension of the grid")

        elif var in ["y" + suffix for suffix in coord_suffixes]:
            variable.attrs.setdefault("long_name", "y-dimension of the grid")

        elif var in ["z" + suffix for suffix in coord_suffixes]:
            variable.attrs.setdefault("long_name", "z-dimension of the grid")

        # Variables
        elif var.startswith("e1"):
            variable.attrs.setdefault("long_name", "Grid spacing in the x direction")
            variable.attrs.setdefault("units", "m")

        elif var.startswith("e2"):
            variable.attrs.setdefault("long_name", "Grid spacing in the y direction")
            variable.attrs.setdefault("units", "m")

        elif var.startswith("e3"):
            variable.attrs.setdefault("standard_name", "cell_thickness")
            variable.attrs.setdefault("long_name", "Cell thickness")
            variable.attrs.setdefault("units", "m")

        elif var.startswith("ff"):
            variable.attrs.setdefault("standard_name", "coriolis_parameter")
            variable.attrs.setdefault("long_name", "Coriolis parameter")
            variable.attrs.setdefault("units", "s-1")

        elif var.startswith("gdep"):
            variable.attrs.setdefault("standard_name", "depth")
            variable.attrs.setdefault("long_name", "Depth")
            variable.attrs.setdefault("units", "m")
            variable.attrs.setdefault("positive", "down")

        elif var.startswith("glam"):
            variable.attrs.setdefault("standard_name", "longitude")
            variable.attrs.setdefault("long_name", "Longitude")
            variable.attrs.setdefault("units", "degree_east")

        elif var.startswith("gphi"):
            variable.attrs.setdefault("standard_name", "latitude")
            variable.attrs.setdefault("long_name", "Latitude")
            variable.attrs.setdefault("units", "degree_north")

        elif var.endswith("mask"):
            variable.attrs.setdefault("standard_name", "sea_binary_mask")
            variable.attrs.setdefault("long_name", "Sea mask (1 = sea, 0 = land)")
            variable.attrs.setdefault("units", "1")

        elif var.endswith("maskutil"):
            variable.attrs.setdefault("standard_name", "sea_binary_mask")
            variable.attrs.setdefault(
                "long_name", "Sea surface mask (1 = sea, 0 = land)"
            )
            variable.attrs.setdefault("units", "1")

        elif var in ["bathy_metry", "bathy_meter", "hw"]:
            variable.attrs.setdefault("standard_name", "sea_floor_depth_below_geoid")
            variable.attrs.setdefault("long_name", "Bathymetry")
            variable.attrs.setdefault("units", "m")

        elif var in ["mbathy", "bottom_level", "mhw"]:
            variable.attrs.setdefault(
                "standard_name", "model_level_number_at_sea_floor"
            )
            variable.attrs.setdefault("long_name", "Last wet point")
            variable.attrs.setdefault("units", "1")

        elif var in ["misf", "top_level"]:
            variable.attrs.setdefault(
                "long_name", "First wet point expressed in model levels"
            )
            variable.attrs.setdefault("units", "1")

        elif var == "isfdraft":
            variable.attrs.setdefault(
                "long_name", "Ice shelf draft expressed in model levels"
            )
            variable.attrs.setdefault("units", "m")

        elif var == "stiffness":
            variable.attrs.setdefault(
                "long_name", "Haney (1991) hydrostatic condition ratio"
            )
            variable.attrs.setdefault("units", "1")

        else:
            unrecognized.add(var)

    if unrecognized:
        warnings.warn(
            f"Pre-defined attributes NOT available for:\n{sorted(unrecognized)}",
            UserWarning,
        )

    # Remove duplicates
    for std_name in ["sea_floor_depth_below_geoid", "model_level_number_at_sea_floor"]:
        var_names = sorted(ds.cf.standard_names.get(std_name, []))
        if len(var_names) > 1:
            for var in var_names[1:]:
                ds[var].attrs.pop("standard_name")
            warnings.warn(
                f"There are multiple variables with standard_name {std_name!r}:"
                f" {var_names!r}. The standard_name attribute has therefore been"
                f" removed from {var_names[1:]}",
                UserWarning,
            )

    return ds


def _place_on_grid(ds: Dataset) -> Dataset:

    roots = {"e1", "e2", "e3", "gdep", "glam", "gphi", "ff", "mask", "maskutil"}
    hgrids = {"f", "t", "u", "v", ""}
    vgrids = {"w", ""}
    grids = {hgrid + vgrid for hgrid in hgrids for vgrid in vgrids}
    suffixes = {"_0", "_1d"}
    separators = {"", "_"}
    tvars = [
        "isfdraft",
        "mbathy",
        "misf",
        "bathy_metry",
        "bathy_meter",
        "top_level",
        "bottom_level",
        "mhw",
        "hw",
        "stiffness",
    ]

    unrecognized = set()
    for var, variable in ds.variables.items():
        if var in tvars:
            rename_dict = dict(X="x", Y="y")
        elif any(
            var == root + separator + hgrid
            for root in roots
            for separator in separators
            for hgrid in hgrids
        ):
            # Example: glam{t,u,v,f}
            rename_dict = dict(
                X="x_right" if var[-1] in ["u", "f"] else "x",
                Y="y_right" if var[-1] in ["v", "f"] else "y",
            )
        elif any(var == hgrid + root for hgrid in hgrids for root in roots):
            # Example: {t,u,v,f}mask
            rename_dict = dict(
                X="x_right" if var[0] in ["u", "f"] else "x",
                Y="y_right" if var[0] in ["v", "f"] else "y",
            )
        elif any(
            var == root + grid + suffix
            for root in roots
            for grid in grids
            for suffix in suffixes
        ):
            # Example: e3{t,u,v,w,uw,vw}_0
            for suffix in suffixes:
                if var.endswith(suffix):
                    var_nosuffix = var.replace(suffix, "")
            rename_dict = dict(
                X="x_right"
                if any(
                    var_nosuffix.endswith(hgrid + vgrid)
                    for hgrid in {"u", "f"}
                    for vgrid in vgrids
                )
                else "x",
                Y="y_right"
                if any(
                    var_nosuffix.endswith(hgrid + vgrid)
                    for hgrid in {"v", "f"}
                    for vgrid in vgrids
                )
                else "y",
                Z="z_left" if var_nosuffix.endswith("w") else "z",
            )
        else:
            # Skip coordinates and dimensionless variables
            if variable.dims and var not in sum(ds.cf.axes.values(), []):
                unrecognized.add(var)
            continue

        if var.endswith("_1d"):
            for axis in {"X", "Y"}:
                rename_dict.pop(axis)
        ds = _rename_var_axes(ds, var, rename_dict)

    if unrecognized:
        warnings.warn(
            f"Pre-defined grid NOT available for:\n{sorted(unrecognized)}",
            UserWarning,
        )

    return ds


def _rename_var_axes(ds: Dataset, var: str, rename_dict: Dict[str, str]) -> Dataset:

    da = ds.cf[var]
    da = da.cf.rename(**rename_dict)
    ds[var] = da.assign_coords({coord: ds[coord] for coord in da.coords})

    return ds


def _find_arbitrary_measures(ds: Dataset) -> Dict[str, List[str]]:

    arbitrary_measures = dict(
        x_spacing=[var for var in ds.variables if var.startswith("e1")],
        y_spacing=[var for var in ds.variables if var.startswith("e2")],
        thickness=ds.cf.standard_names["cell_thickness"],
    )

    return arbitrary_measures


def _swap_time(ds: Dataset) -> Dataset:

    # Upgrade time coordinates to dimensions
    dscounter = ds.cf[["time_counter"]]
    time2swap = None
    for time in ds.cf.axes.get("T", []):
        # Time coordinates are axes as well
        ds[time].attrs.setdefault("axis", "T")
        # Time identical to counter will be swapped later
        if ds[time].equals(ds["time_counter"]):
            time2swap = time if time != "time_counter" else time2swap
            continue
        # Upgrade
        dstime = ds.cf[[time]]
        dstime = dstime.swap_dims(time_counter=time)
        dstime = dstime.drop_vars(dscounter.variables).reset_coords()
        # Find shift
        diff = (
            dstime[time].values.tolist()[0]
            - dscounter["time_counter"].values.tolist()[0]
        )
        diff = diff.total_seconds() if isinstance(diff, timedelta) else diff
        sign = np.sign(diff)
        if sign:
            dstime[time].attrs.setdefault("c_grid_axis_shift", 0.5 * sign)
        # Substitute
        ds = xr.merge([dstime, ds], compat="override")

    # Swap
    if time2swap:
        ds = ds.swap_dims(time_counter=time2swap)
        ds = ds.drop_vars(dscounter.variables)

    return ds
