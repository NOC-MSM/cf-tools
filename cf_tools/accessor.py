"""
Accessor for all models
"""
from typing import Any, Dict, List, Optional, Union

import cf_xarray  # noqa: F401 pylint: disable=W0611
import numpy as np
import xarray as xr
from xarray import DataArray, Dataset
from xesmf import Regridder
from xgcm import Grid


@xr.register_dataset_accessor("cf_tools")
class Accessor:
    """
    Parent class for all models
    """

    def __init__(self, xarray_obj: Union[Dataset, DataArray]):
        """
        Initialize class

        Parameters
        ----------
        xarray_obj: Dataset, DataArray
            xarray object to access
        """

        self._obj = xarray_obj
        self._options: Optional[Dict[str, Any]] = None
        self._old_options: Optional[Dict[str, Any]] = None

    def __enter__(self):
        self._old_options = self.options

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._options = self._old_options

    def set_options(self, **kwargs: Any):
        """
        Set options

        Parameters
        ----------
        **kwargs: Any
            Options to set
        """

        self._options = {**self._options, **kwargs} if self._options else kwargs

    def unset_options(self, *args: str):
        """
        Options to unset

        Parameters
        ----------
        *args: str
            Options to unset
        """

        if self._options:
            for arg in args:
                self._options.pop(arg, None)

    @property
    def options(self) -> Dict[str, Any]:
        """
        Options currently set

        Returns
        -------
        dict:
            Options currently set
        """

        return self._options if self._options else {}

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
            If periodic is not set
        """

        # Periodic
        periodic = self.options.get("periodic")
        if periodic is None and error:
            raise ValueError(
                "`periodic` must be set using `.cf_tools.set_options`. "
                "See: https://xgcm.readthedocs.io/en/latest/api.html#grid"
            )

        # Kwargs
        kwargs = dict()
        if periodic is not None:
            kwargs["periodic"] = periodic

        return Grid(self._obj, **kwargs)

    @property
    def _separated_arakawa_grids(self) -> Dict[str, Dataset]:

        axes = self.grid(error=False).axes
        obj = self._obj

        obj = obj.drop_vars(
            [var for var in obj.data_vars if not {"X", "Y"} <= set(obj[var].cf.axes)]
        )

        assert {"X", "Y"} <= set(axes), "Object does not have X and Y axes"
        x_coords = axes["X"].coords
        y_coords = axes["Y"].coords
        assert (
            len(x_coords) == len(y_coords) == 2
        ), "Only two X and Y dimensions are allowed"

        dims_to_drop = dict(
            T=[dim for shift, dim in x_coords.items() if shift != "center"]
            + [dim for shift, dim in y_coords.items() if shift != "center"],
            U=[dim for shift, dim in x_coords.items() if shift == "center"]
            + [dim for shift, dim in y_coords.items() if shift != "center"],
            V=[dim for shift, dim in x_coords.items() if shift != "center"]
            + [dim for shift, dim in y_coords.items() if shift == "center"],
            F=[dim for shift, dim in x_coords.items() if shift == "center"]
            + [dim for shift, dim in y_coords.items() if shift == "center"],
        )

        return {key: obj.drop_dims(value) for key, value in dims_to_drop.items()}

    def extract_transect_along_f(self, lons: List[float], lats: List[float]) -> Dataset:
        """
        Return a transect defined by U and V grid points.
        T and F fields are interpolated, whereas U and V are not.

        Parameters
        ----------
        lons: list
            Longitudes defining the transect
        lats: list
            Latitudes defining the transect

        Returns
        -------
        Dataset
            Transect along F grid
        """

        # Extract transect defined by F points.
        arakawas = self._separated_arakawa_grids
        ds = _extract_transect(arakawas["F"], lons, lats, no_boundaries=True)

        # F: average adjacent points.
        coords = {
            coord: da.rolling(station=2).mean().isel(station=slice(1, None))
            for coord, da in ds.coords.items()
            if "station" in da.dims
        }
        ds = ds.rolling(station=2).mean().isel(station=slice(1, None))
        for coord, da in coords.items():
            ds[coord] = da
        ds["station"] = ds["station"]
        arakawas["F"] = ds

        # U, V: no interpolation.
        for grid in ["U", "V"]:
            ds = arakawas[grid]
            mask = xr.where(
                np.logical_and(
                    *(arakawas["F"].cf[axis].isin(ds.cf[axis]) for axis in ("X", "Y"))
                ),
                1,
                0,
            )
            arakawas[grid] = ds.cf.sel(
                {
                    axis: arakawas["F"].cf[axis].where(mask, drop=True)
                    for axis in ("X", "Y")
                }
            )

        # T: linear interpolation.
        ds = arakawas["T"]
        ds = ds.cf.sel(
            {
                axis: xr.concat(
                    [
                        arakawas["F"]
                        .cf[axis]
                        .where(
                            arakawas["F"].cf[axis].isin(ds.cf[axis]),
                            arakawas["F"].cf[axis] + shift,
                        )
                        for shift in (-0.5, 0.5)
                    ],
                    "tmp_dim",
                )
                for axis in ("X", "Y")
            }
        )
        coords = {
            coord: da.mean("tmp_dim", keep_attrs=True)
            for coord, da in ds.coords.items()
            if "tmp_dim" in da.dims
        }
        ds = ds.mean("tmp_dim", keep_attrs=True)
        arakawas["T"] = ds.assign_coords(coords)

        # Merge
        arakawas_merged = xr.merge(arakawas.values())
        missing_vars = set(self._obj.variables) - set(arakawas_merged.variables)
        arakawas_merged = arakawas_merged.cf.drop_vars(["X", "Y"])

        return xr.merge([arakawas_merged, self._obj[list(missing_vars)]])

    def _volume_flux_along_axis(self, axis: str):

        options = ("x", "y")
        assert axis in options, f"Axis `{axis}` not available. Options: {options}"

        if f"ocean_volume_{axis}_transport" in self._obj.cf:
            return self._obj.cf[f"ocean_volume_{axis}_transport"]

        # Compute transport
        velocity = self._obj.cf[f"sea_water_{axis}_velocity"]
        cell_area = (
            velocity.cf[f"{'x' if axis=='y' else 'y'}_spacing"]
            * velocity.cf["thickness"]
        )
        transport = velocity * cell_area

        # Attributes
        transport.attrs["standard_name"] = f"ocean_volume_{axis}_transport"
        transport.attrs["long_name"] = f"Volume flux along {axis}-axis"
        transport.attrs["units"] = "m3 s-1"
        transport.attrs["history"] = "Computed offline"
        for attr in {"coordinates", "cell_measures"} & set(velocity.attrs):
            transport.attrs[attr] = velocity.attrs.get(attr)

        return transport

    @property
    def ocean_volume_x_transport(self) -> DataArray:
        """
        Return ocean_volume_x_transport computing it if missing

        Returns
        -------
        DataArray
        """

        return self._volume_flux_along_axis("x")

    @property
    def ocean_volume_y_transport(self):
        """
        Return ocean_volume_y_transport computing it if missing

        Returns
        -------
        DataArray
        """

        return self._volume_flux_along_axis("y")

    def ocean_volume_transport_across_line(
        self,
        flip_x: Optional[bool] = None,
        flip_y: Optional[bool] = None,
        mask: Optional[DataArray] = None,
    ):
        """
        Return ocean_volume_transport_across_line computing it if missing

        Parameters
        ----------
        flip_x: bool, optional
            Whether to flip x-velocity
        flip_y: bool, optional
            Whether to flip y-velocity
        mask: DataArray, optional
            Mask to apply to the velocities

        Returns
        -------
        DataArray
        """

        x_transport = self.ocean_volume_x_transport
        y_transport = self.ocean_volume_y_transport
        if flip_x:
            x_transport *= -1
        if flip_y:
            y_transport *= -1
        transport = x_transport.fillna(y_transport)
        if mask is not None:
            transport = transport.where(mask, 0)
        transport = transport.cf.sum(
            ["station"] + ["Z"] if "Z" in transport.cf.axes else []
        )

        # Attributes
        transport.attrs["standard_name"] = "ocean_volume_transport_across_line"
        transport.attrs["long_name"] = "Volume flux across line"
        transport.attrs["units"] = "m3 s-1"
        transport.attrs["history"] = "Computed offline"

        return transport


def _extract_transect(
    ds: Dataset,
    lons: List[float],
    lats: List[float],
    no_boundaries: Optional[bool] = None,
) -> Dataset:

    # Find indexes defining the transect (exclude first/last row/column)
    ds_in = ds.cf[["longitude", "latitude"]]
    if no_boundaries:
        ds_in = ds_in.cf.isel(X=slice(1, -1), Y=slice(1, -1))
    ds_out = Dataset(dict(lon=lons, lat=lats))
    regridder = Regridder(ds_in, ds_out, "nearest_s2d", locstream_out=True)
    iinds, jinds = xr.broadcast(
        *(
            DataArray(range(ds_in.cf.sizes[axis]), dims=ds_in.cf[axis].dims)
            for axis in ("X", "Y")
        )
    )
    iinds = regridder(iinds).values.astype(int)
    jinds = regridder(jinds).values.astype(int)

    # Add points halving steps until -1 <= step <= 1
    insert_inds = None
    while insert_inds is None or len(insert_inds):
        idiff, jdiff = (np.diff(inds) for inds in (iinds, jinds))
        mask = np.logical_or(np.abs(idiff) > 1, np.abs(jdiff) > 1)
        insert_inds = np.where(mask)[0]
        iinds, jinds = (
            np.insert(inds, insert_inds + 1, inds[insert_inds] + diff[insert_inds] // 2)
            for inds, diff in zip([iinds, jinds], [idiff, jdiff])
        )

    # Remove diagonal jumps
    idiff, jdiff = (np.diff(inds) for inds in (iinds, jinds))
    mask = np.logical_and(idiff, jdiff)
    insert_inds = np.where(mask)[0]
    iinds = np.insert(iinds, insert_inds + 1, iinds[insert_inds + 1])
    jinds = np.insert(jinds, insert_inds + 1, jinds[insert_inds])
    if no_boundaries:
        iinds += 1
        jinds += 1

    # Sanity check
    idiff, jdiff = (np.diff(inds) for inds in (iinds, jinds))
    assert all(idiff * jdiff == 0) & all(
        np.abs(idiff + jdiff) == 1
    ), "Path is not continuous"

    return ds.cf.isel(
        X=DataArray(iinds, dims="station"), Y=DataArray(jinds, dims="station")
    )
