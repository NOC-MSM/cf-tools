"""
Test NemoDataset
"""
# pylint: disable=C0116
import pytest
from xarray.testing import assert_allclose, assert_equal, assert_identical

from cf_tools import NemoDataset, OceanDataset

from .datasets import amm12_cf_output, amm12_domain, amm12_mesh_mask, amm12_output

amm12 = NemoDataset(amm12_domain, amm12_output)
amm12_mesh_cf = NemoDataset(amm12_mesh_mask, amm12_cf_output)
amm12_mesh = NemoDataset(amm12_mesh_mask, amm12_output)
amm12_missing = NemoDataset(amm12_domain, amm12_output, add_missing=True)


@pytest.mark.parametrize(
    "domain,output", [(amm12_domain, amm12_output), (amm12_mesh_mask, amm12_cf_output)]
)
def test_variables(domain, output):

    ds = NemoDataset(domain, output).dataset

    grids = {"t", "u", "v", "w"}

    expected = list(domain.variables)
    expected += sum([list(output[grid].variables) for grid in grids], [])
    expected += ["x_c", "y_c", "x_r", "y_r", "z_c", "z_l"]
    expected = set(expected)
    expected -= {"x", "y", "z"}
    expected -= {"depth" + grid for grid in grids}
    expected -= {f"depth{grid}_bounds" for grid in grids}
    expected -= {"t", "time_counter", "time_counter_bounds"}
    if "area" in expected:
        expected -= {"area"}
        expected += {"areat", "areau", "areav"}
    actual = set(ds.variables)
    assert actual == expected


def test_rebuild():

    expected = amm12.dataset

    # Rebuild from dataset
    actual = NemoDataset(expected).dataset
    assert_identical(expected, actual)

    # Rebuild from domain and output
    actual = NemoDataset(amm12.domain, amm12.arakawa_datasets).dataset
    assert_identical(expected, actual)

    # Rebuild using OceanDataset
    actual = OceanDataset(amm12.dataset).dataset
    assert_identical(expected, actual)


@pytest.mark.parametrize(
    "ds,mesh_cf", [(amm12.dataset, False), (amm12_mesh_cf.dataset, True)]
)
def test_cf_attributes(ds, mesh_cf):

    hgrids = {"t", "u", "v", "f"}

    # Axes
    assert set(ds.cf.axes) == {"X", "Y", "Z", "T"}
    assert set(ds.cf.axes["X"]) == {"x_c", "x_r"}
    assert set(ds.cf.axes["Y"]) == {"y_c", "y_r"}
    assert set(ds.cf.axes["Z"]) == {"z_c", "z_l"}
    assert set(ds.cf.axes["T"]) == {"time_instant", "time_centered"}

    # Coordinates
    expected_coords = ["longitude", "latitude", "time"]
    expected_coords += ["vertical"] if mesh_cf else []
    assert set(ds.cf.coordinates) == set(expected_coords)
    assert set(ds.cf.coordinates["longitude"]) == {"glam" + grid for grid in hgrids}
    assert set(ds.cf.coordinates["latitude"]) == {"gphi" + grid for grid in hgrids}
    assert set(ds.cf.coordinates["time"]) == {"time_instant", "time_centered"}
    if mesh_cf:
        assert set(ds.cf.coordinates["vertical"]) == {"gdept_0", "gdepw_0"}

    # Cell measures
    expected_meas = ["xdist", "ydist", "thickness"]
    expected_meas += ["area"] if mesh_cf else []
    assert set(ds.cf.cell_measures) == set(expected_meas)
    assert set(ds.cf.cell_measures["xdist"]) == {"e1" + grid for grid in hgrids}
    assert set(ds.cf.cell_measures["ydist"]) == {"e2" + grid for grid in hgrids}

    expected_thick = ["e3" + grid for grid in {"t", "u", "v", "w"}]
    if mesh_cf:
        # Thicknesses added to masks
        expected_thick += [f"e3{grid}_0" for grid in hgrids]
    assert set(ds.cf.cell_measures["thickness"]) == set(expected_thick)
    std_thick = [f"e3{grid}" for grid in {"t", "u", "v", "w"}] + [
        f"e3{grid}_0" for grid in {"t", "u", "v", "f", "w", "uw", "vw"}
    ]
    assert set(ds.cf.standard_names["cell_thickness"]) == set(std_thick)

    if mesh_cf:
        assert set(ds.cf.cell_measures["area"]) == {"areat", "areau", "areav"}


def test_areas():

    for grid in {"t", "u", "v"}:
        assert_allclose(
            amm12_missing.dataset["area" + grid], amm12_mesh_cf.dataset["area" + grid]
        )


def test_depths():

    for name_mesh, name_out in zip(["gdept_0", "gdepw_0"], ["depth0t", "depth0w"]):
        assert_equal(
            amm12_mesh_cf.dataset[name_mesh].reset_coords(drop=True),
            amm12_missing.dataset[name_out].reset_coords(drop=True),
        )


def test_options():

    assert amm12.options["periodic"] is False
