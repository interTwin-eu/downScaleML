import xarray as xr
import dask.array as da
import numpy as np
from datetime import date
from typing import Dict, Optional, Tuple

def match_to_mid_resolution(source_ds: xr.Dataset, target_ds: xr.Dataset, 
                          lat_name: str = 'lat', lon_name: str = 'lon',
                          num_mid_lats: Optional[int] = None, 
                          num_mid_lons: Optional[int] = None) -> xr.Dataset:
    """Interpolate datasets to common mid-resolution grid."""
    if num_mid_lats is None:
        num_mid_lats = len(target_ds[lat_name]) // 4
    if num_mid_lons is None:
        num_mid_lons = len(target_ds[lon_name]) // 4

    min_lat = max(source_ds[lat_name].min().item(), target_ds[lat_name].min().item())
    max_lat = min(source_ds[lat_name].max().item(), target_ds[lat_name].max().item())
    min_lon = max(source_ds[lon_name].min().item(), target_ds[lon_name].min().item())
    max_lon = min(source_ds[lon_name].max().item(), target_ds[lon_name].max().item())

    mid_lats = np.linspace(min_lat, max_lat, num_mid_lats)
    mid_lons = np.linspace(min_lon, max_lon, num_mid_lons)

    mid_coords = {
        lat_name: xr.DataArray(mid_lats, dims=lat_name),
        lon_name: xr.DataArray(mid_lons, dims=lon_name)
    }

    source_mid = source_ds.interp(mid_coords, method='linear')
    target_mid = target_ds.interp(mid_coords, method='linear')

    extra_dims = {dim: source_mid.coords[dim] for dim in source_mid.dims
                 if dim not in [lat_name, lon_name]}

    if extra_dims:
        target_expanded = target_mid.expand_dims(extra_dims).broadcast_like(source_mid)
    else:
        target_expanded = target_mid

    source_mid, target_expanded = xr.align(source_mid, target_expanded)
    target_expanded = target_expanded.chunk(source_mid.chunks)

    return xr.merge([source_mid, target_expanded]).astype('float32')

def encode_doys(ds: xr.Dataset, time_dim: str = 'time', 
               spatial_dims: Optional[Tuple[str, str]] = None,
               extra_dims: Optional[list] = None, 
               inplace: bool = False) -> xr.Dataset:
    """Encode day of year as cyclical features."""
    if not inplace:
        ds = ds.copy()
    
    def get_spatial_dims(ds):
        dims = set(ds.dims)
        y_dim = next((d for d in ['y', 'lat', 'latitude', 'lats'] if d in dims), None)
        x_dim = next((d for d in ['x', 'lon', 'longitude', 'long', 'lons'] if d in dims), None)
        if y_dim is None or x_dim is None:
            raise ValueError(f"Could not detect spatial dimensions. Available dimensions: {list(dims)}")
        return y_dim, x_dim
    
    y_dim, x_dim = spatial_dims if spatial_dims else get_spatial_dims(ds)
    target_dims = [time_dim] + (extra_dims if extra_dims else []) + [y_dim, x_dim]
    
    doys = ds[time_dim].values.astype('datetime64[D]')
    doys = da.asarray([date.timetuple(doy.astype(object)).tm_yday for doy in doys])
    
    sin_doy, cos_doy = (np.sin(2 * np.pi * doys / 365), 
                        np.cos(2 * np.pi * doys / 365))
    
    for dim in target_dims[1:]:
        repeats = len(ds[dim])
        sin_doy = da.repeat(sin_doy[..., None], repeats, axis=-1)
        cos_doy = da.repeat(cos_doy[..., None], repeats, axis=-1)
    
    sin_doy = sin_doy.reshape([len(ds[dim]) for dim in target_dims])
    cos_doy = cos_doy.reshape([len(ds[dim]) for dim in target_dims])
    
    ds['sin_doy'] = (target_dims, sin_doy)
    ds['cos_doy'] = (target_dims, cos_doy)
    
    for name in ['sin_doy', 'cos_doy']:
        ds[name].attrs.update({
            'long_name': f"{'Sine' if 'sin' in name else 'Cosine'} of day of year",
            'units': 'unitless',
            'description': f"Cyclical encoding of day of year"
        })
    
    return ds