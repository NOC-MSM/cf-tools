"""
Dataset used for testing
"""

import os

import xarray as xr

from .conftest import data_path

orca2_ice_pisces_path = os.path.join(data_path, "ORCA2_ICE_PISCES")
orca2_ice_pisces = dict(
    domain_cfg=xr.open_dataset(
        os.path.join(orca2_ice_pisces_path, "ORCA_R2_zps_domcfg.nc"), chunks=-1
    ),
    mesh_mask=xr.open_dataset(
        os.path.join(orca2_ice_pisces_path, "mesh_mask.nc"), chunks=-1
    ),
    grid_T=xr.open_mfdataset(os.path.join(orca2_ice_pisces_path, "*grid_T*.nc")),
    grid_U=xr.open_mfdataset(os.path.join(orca2_ice_pisces_path, "*grid_U*.nc")),
    grid_V=xr.open_mfdataset(os.path.join(orca2_ice_pisces_path, "*grid_V*.nc")),
    grid_W=xr.open_mfdataset(os.path.join(orca2_ice_pisces_path, "*grid_W*.nc")),
    icemod=xr.open_mfdataset(os.path.join(orca2_ice_pisces_path, "*icemod*.nc")),
)
orca2_ice_pisces = {
    key: value.isel(x=slice(100, 120), y=slice(100, 110))
    for key, value in orca2_ice_pisces.items()
}
