"""
Tests specific for NEMO
"""
# pylint: disable=C0116

import os

import pytest
import xarray as xr
from numpy.testing import assert_equal
from xarray.testing import assert_allclose, assert_identical

import cf_tools.nemo  # noqa: F401 pylint: disable=W0611
from cf_tools.nemo import standardize_domain, standardize_output

from .datasets import orca2_ice_pisces

std_domain = standardize_domain(orca2_ice_pisces["domain_cfg"])
std_mesh = standardize_domain(orca2_ice_pisces["mesh_mask"])
std_ds = xr.merge(
    [
        standardize_output(
            orca2_ice_pisces[key], std_mesh, hgrid="T" if key == "icemod" else None
        )
        for key in {"grid_T", "grid_U", "grid_V", "grid_W", "icemod"}
    ]
)


@pytest.mark.parametrize("ds", [std_domain, std_mesh])
def test_domain_axes(ds):

    actual = set(ds.cf.axes["X"])
    expected = {"x", "x_right"}
    assert actual == expected

    actual = set(ds.cf.axes["Y"])
    expected = {"y", "y_right"}
    assert actual == expected

    actual = set(ds.cf.axes["Z"])
    expected = {"z", "z_left"}
    assert actual == expected

    assert "T" not in ds.cf.axes


@pytest.mark.parametrize("ds", [std_domain, std_mesh])
def test_domain_coordinates(ds):

    actual = set(ds.cf.coordinates["longitude"])
    expected = {"glamt", "glamu", "glamv", "glamf"}
    assert actual == expected

    actual = set(ds.cf.coordinates["latitude"])
    expected = {"gphit", "gphiu", "gphiv", "gphif"}
    assert actual == expected

    actual = set(ds.cf.coordinates.get("vertical", []))
    expected = (
        {"gdept_0", "gdept_1d", "gdepw_0", "gdepw_1d"}
        if os.path.basename(ds.encoding["source"]) == "mesh_mask.nc"
        else set()
    )
    assert actual == expected

    assert "time" not in ds.cf.coordinates


@pytest.mark.parametrize("ds", [std_domain, std_mesh])
def test_domain_cell_measures(ds):

    actual = set(ds.cf.cell_measures["x_spacing"])
    expected = (
        {"e1t", "e1u", "e1v", "e1f"}
        if os.path.basename(ds.encoding["source"]) == "mesh_mask.nc"
        else {"e1t", "e1f"}
    )
    assert actual == expected

    actual = set(ds.cf.cell_measures["y_spacing"])
    expected = (
        {"e2t", "e2u", "e2v", "e2f"}
        if os.path.basename(ds.encoding["source"]) == "mesh_mask.nc"
        else {"e2t", "e2f"}
    )
    assert actual == expected

    actual = set(ds.cf.cell_measures.get("thickness", []))
    expected = (
        {"e3t_0", "e3u_0", "e3v_0", "e3f_0"}
        if os.path.basename(ds.encoding["source"]) == "mesh_mask.nc"
        else set()
    )
    assert actual == expected


@pytest.mark.parametrize("ds", [std_domain, std_mesh])
def test_domain_standard_names(ds):

    actual = set(ds.cf.standard_names["cell_thickness"])
    expected = {
        "e3t_0",
        "e3u_0",
        "e3v_0",
        "e3f_0",
        "e3w_0",
        "e3uw_0",
        "e3vw_0",
        "e3t_1d",
        "e3w_1d",
    }
    assert actual == expected

    actual = set(ds.cf.standard_names["coriolis_parameter"])
    expected = {"ff_t", "ff_f"}
    assert actual == expected

    actual = set(ds.cf.standard_names["model_level_number_at_sea_floor"])
    expected = {
        "mbathy"
        if os.path.basename(ds.encoding["source"]) == "mesh_mask.nc"
        else "bottom_level"
    }
    assert actual == expected

    actual = set(ds.cf.standard_names.get("sea_binary_mask", []))
    expected = (
        {"tmask", "umask", "vmask", "fmask", "tmaskutil", "umaskutil", "vmaskutil"}
        if os.path.basename(ds.encoding["source"]) == "mesh_mask.nc"
        else set()
    )
    assert actual == expected


@pytest.mark.parametrize("ds", [std_domain, std_mesh])
def test_recreate_domain(ds):

    # Recreate identical objects
    ds1 = standardize_domain(ds)
    assert_identical(ds, ds1)


@pytest.mark.parametrize(
    "ds,dims,hgrid",
    [
        (orca2_ice_pisces["grid_T"], {"x", "y", "z", "time_centered"}, None),
        (orca2_ice_pisces["grid_U"], {"x_right", "y", "z", "time_centered"}, None),
        (orca2_ice_pisces["grid_V"], {"x", "y_right", "z", "time_centered"}, None),
        (orca2_ice_pisces["grid_W"], {"x", "y", "z_left", "time_centered"}, None),
        (orca2_ice_pisces["icemod"], {"x", "y", "ncatice", "time_centered"}, "T"),
    ],
)
@pytest.mark.parametrize("domain", [std_mesh, std_domain])
def test_output(ds, domain, dims, hgrid):

    # Test error
    if hgrid:
        with pytest.raises(ValueError):
            output = standardize_output(ds, domain, hgrid=None)

    # Standardize output
    output = standardize_output(ds, domain, hgrid=hgrid)

    # Dropped variables
    missing = set(ds.variables) - set(output.variables)
    missing_lon = {"nav_lon", "nav_lat"}
    missing_lat = {"time_counter", "time_counter_bounds"}
    assert missing_lon <= missing
    assert missing_lat <= missing
    assert all(var.startswith("dep") for var in missing - missing_lon - missing_lat)

    # Dimensions
    for var in set(output.variables) & set(ds.variables):
        if not var.startswith("time"):
            assert set(output[var].dims) <= dims
        else:
            assert set(output[var].dims) <= {var.replace("_bounds", ""), "axis_nbounds"}


def test_all_output():

    domain = std_mesh
    ds = std_ds

    # Axes
    expected = {**domain.cf.axes, **{"T": ["time_instant", "time_centered"]}}
    actual = ds.cf.axes
    assert {key: set(value) for key, value in expected.items()} == {
        key: set(value) for key, value in actual.items()
    }

    # Coordinates
    expected = {**domain.cf.coordinates, **{"time": ["time_instant", "time_centered"]}}
    actual = ds.cf.coordinates
    assert {key: set(value) for key, value in expected.items()} == {
        key: set(value) for key, value in actual.items()
    }

    # Cell measures
    expected = {
        **domain.cf.cell_measures,
        **{
            "thickness": {
                "e3t",
                "e3u",
                "e3v",
                "e3w",
                "e3t_0",
                "e3u_0",
                "e3v_0",
                "e3f_0",
            }
        },
    }
    actual = ds.cf.cell_measures
    assert {key: set(value) for key, value in expected.items()} == {
        key: set(value) for key, value in actual.items()
    }

    # Cell thickness
    expected = domain.cf.standard_names["cell_thickness"] + ["e3t", "e3u", "e3v", "e3w"]
    actual = ds.cf.standard_names["cell_thickness"]
    assert set(expected) == set(actual)


def test_sea_floor_depth_below_geoid():

    ds = std_mesh
    bathy = ds.nemo_tools.sea_floor_depth_below_geoid
    depth = ds["gdept_0"]

    actual = depth.where(depth > bathy)
    actual = actual.where(actual.isel(z=-1).notnull(), 0).argmin("z").values
    expected = ds["mbathy"].values
    assert_equal(expected, actual)


def test_vertical():

    expected = std_mesh.cf[["vertical"]]
    actual = std_domain.nemo_tools.vertical
    for var in actual.cf.coordinates["vertical"]:
        actual[var].attrs.pop("history")
    assert_identical(expected, actual)


def test_transect():

    ds = std_ds

    lons = [ds["glamt"].min().values, ds["glamt"].max().values]
    lats = [ds["gphit"].min().values, ds["gphit"].max().values]
    transect = ds.nemo_tools.extract_transect_along_f(lons, lats)

    # Dimensions
    expected = {
        "location",
        "axis_nbounds",
        "z",
        "z_left",
        "time_centered",
        "time_instant",
        "ncatice",
    }
    actual = set(transect.dims)
    assert expected == actual

    # Attributes
    for var in set(ds.variables) - {"x", "x_right", "y", "y_right"}:
        expected = ds[var].attrs
        actual = transect[var].attrs
        assert expected == actual

    # Coordinates
    for coord in ["glam", "gphi"]:
        assert all(
            transect[coord + "u"].notnull() + transect[coord + "v"].notnull() == 1
        )
        vel_coord = transect[coord + "u"].where(
            transect[coord + "u"].notnull(), transect[coord + "v"]
        )
        for grid in ["t", "f"]:
            assert_allclose(vel_coord, transect[coord + grid], 1.0e-3)
