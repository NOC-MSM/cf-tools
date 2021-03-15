"""
Tests specific for NEMO
"""
# pylint: disable=C0116

import os

import pytest
import xarray as xr
from xarray.testing import assert_identical

from cf_tools.nemo import standardize_domain, standardize_output

from .datasets import orca2_ice_pisces


@pytest.mark.parametrize(
    "obj", [orca2_ice_pisces["mesh_mask"], orca2_ice_pisces["domain_cfg"]]
)
def test_domain_axes(obj):

    ds = standardize_domain(obj)

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


@pytest.mark.parametrize(
    "obj", [orca2_ice_pisces["mesh_mask"], orca2_ice_pisces["domain_cfg"]]
)
def test_domain_coordinates(obj):

    ds = standardize_domain(obj)

    actual = set(ds.cf.coordinates["longitude"])
    expected = {"glamt", "glamu", "glamv", "glamf"}
    assert actual == expected

    actual = set(ds.cf.coordinates["latitude"])
    expected = {"gphit", "gphiu", "gphiv", "gphif"}
    assert actual == expected

    actual = set(ds.cf.coordinates.get("vertical", []))
    expected = (
        {"gdept_0", "gdept_1d", "gdepw_0", "gdepw_1d"}
        if os.path.basename(obj.encoding["source"]) == "mesh_mask.nc"
        else set()
    )
    assert actual == expected

    assert "time" not in ds.cf.coordinates


@pytest.mark.parametrize(
    "obj", [orca2_ice_pisces["mesh_mask"], orca2_ice_pisces["domain_cfg"]]
)
def test_domain_cell_measures(obj):

    ds = standardize_domain(obj)

    actual = set(ds.cf.cell_measures["x_spacing"])
    expected = (
        {"e1t", "e1u", "e1v", "e1f"}
        if os.path.basename(obj.encoding["source"]) == "mesh_mask.nc"
        else {"e1t", "e1f"}
    )
    assert actual == expected

    actual = set(ds.cf.cell_measures["y_spacing"])
    expected = (
        {"e2t", "e2u", "e2v", "e2f"}
        if os.path.basename(obj.encoding["source"]) == "mesh_mask.nc"
        else {"e2t", "e2f"}
    )
    assert actual == expected

    actual = set(ds.cf.cell_measures.get("thickness", []))
    expected = (
        {"e3t_0", "e3u_0", "e3v_0", "e3f_0"}
        if os.path.basename(obj.encoding["source"]) == "mesh_mask.nc"
        else set()
    )
    assert actual == expected


@pytest.mark.parametrize(
    "obj", [orca2_ice_pisces["mesh_mask"], orca2_ice_pisces["domain_cfg"]]
)
def test_domain_standard_names(obj):

    ds = standardize_domain(obj)

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
        if os.path.basename(obj.encoding["source"]) == "mesh_mask.nc"
        else "bottom_level"
    }
    assert actual == expected

    actual = set(ds.cf.standard_names.get("sea_binary_mask", []))
    expected = (
        {"tmask", "umask", "vmask", "fmask", "tmaskutil", "umaskutil", "vmaskutil"}
        if os.path.basename(obj.encoding["source"]) == "mesh_mask.nc"
        else set()
    )
    assert actual == expected


@pytest.mark.parametrize(
    "obj", [orca2_ice_pisces["mesh_mask"], orca2_ice_pisces["domain_cfg"]]
)
def test_recreate_domain(obj):

    # Recreate identical objects
    ds = standardize_domain(obj)
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
@pytest.mark.parametrize(
    "domain", [orca2_ice_pisces["mesh_mask"], orca2_ice_pisces["domain_cfg"]]
)
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


@pytest.mark.parametrize(
    "domain", [orca2_ice_pisces["mesh_mask"], orca2_ice_pisces["domain_cfg"]]
)
def test_merge_output(domain):

    datasets = [
        standardize_output(
            orca2_ice_pisces[key], domain, hgrid="T" if key == "icemod" else None
        )
        for key in {"grid_T", "grid_U", "grid_V", "grid_W", "icemod"}
    ]
    xr.merge(datasets)
