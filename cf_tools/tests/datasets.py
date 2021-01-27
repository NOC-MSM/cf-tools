"""
Open datasets for testing
"""
import os

import xarray as xr

from .conftest import data_path

amm12_path = os.path.join(data_path, "AMM12")
amm12_domain = xr.open_dataset(os.path.join(amm12_path, "domain.nc"))
amm12_mesh_mask = xr.open_dataset(os.path.join(amm12_path, "mesh_mask.nc"))
amm12_output = {
    grid: xr.open_dataset(os.path.join(amm12_path, f"{grid}.nc"))
    for grid in {"t", "u", "v", "w"}
}
amm12_cf_output = {
    grid: xr.open_dataset(os.path.join(amm12_path, f"{grid}_cfmeta.nc"))
    for grid in {"t", "u", "v", "w"}
}
