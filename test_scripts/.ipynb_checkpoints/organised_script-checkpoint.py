# Import statements

import os
import xarray as xr
import dask.array as da
from lightgbm import LGBMRegressor
from dask.distributed import LocalCluster, Client, performance_report
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import psutil
from datetime import date


def match_to_mid_resolution(source_ds, target_ds, lat_name='lat', lon_name='lon', num_mid_lats=30, num_mid_lons=30):
    # Get coordinate bounds from union of source and target
    min_lat = max(source_ds[lat_name].min().item(), target_ds[lat_name].min().item())
    max_lat = min(source_ds[lat_name].max().item(), target_ds[lat_name].max().item())
    min_lon = max(source_ds[lon_name].min().item(), target_ds[lon_name].min().item())
    max_lon = min(source_ds[lon_name].max().item(), target_ds[lon_name].max().item())
    
    # Create mid-resolution grid
    mid_lats = np.linspace(min_lat, max_lat, num_mid_lats)
    mid_lons = np.linspace(min_lon, max_lon, num_mid_lons)
    
    # Interpolate both datasets to mid-resolution grid using bilinear interpolation
    mid_coords = {
        lat_name: xr.DataArray(mid_lats, dims=lat_name),
        lon_name: xr.DataArray(mid_lons, dims=lon_name)
    }
    
    source_mid = source_ds.interp(mid_coords, method='linear')
    
    return source_mid

def normalize_dataset(ds):
    # Create a copy to avoid modifying the original dataset
    ds_normalized = ds.copy()
    
    # Loop through all data variables
    for var in ds.data_vars:
        # Subtract the minimum (along all dimensions except the variable's own)
        min_val = ds[var].min(keep_attrs=True)
        ds_normalized[var] = ds[var] - min_val
        
        # Divide by the maximum (after min subtraction)
        max_val = ds_normalized[var].max(keep_attrs=True)
        ds_normalized[var] = ds_normalized[var] / max_val
        
        # Preserve attributes if they exist
        if 'attrs' in ds[var].attrs:
            ds_normalized[var].attrs.update(ds[var].attrs)
    
    return ds_normalized

def encode_cyclical_features(values, max_value):
    """Encode cyclical features using sine and cosine transformations."""
    sin = np.sin(2 * np.pi * values / max_value)
    cos = np.cos(2 * np.pi * values / max_value)
    return sin, cos

def repeat_along_axis(arr, repeats, axis):
    """Repeat array along specified axis."""
    return da.repeat(arr[None, ...], repeats, axis=axis)

def get_spatial_dims(ds):
    """
    Detect spatial dimension names in the dataset.
    Returns (y_dim, x_dim) tuple based on common naming conventions.
    """
    dims = set(ds.dims)
    
    y_candidates = ['y', 'lat', 'latitude', 'lats']
    x_candidates = ['x', 'lon', 'longitude', 'long', 'lons']
    
    y_dim = next((d for d in y_candidates if d in dims), None)
    x_dim = next((d for d in x_candidates if d in dims), None)
    
    if y_dim is None or x_dim is None:
        raise ValueError(
            f"Could not detect spatial dimensions. Available dimensions: {list(dims)}. "
            f"Tried y names: {y_candidates}, x names: {x_candidates}"
        )
    
    return y_dim, x_dim

def get_existing_chunks(ds, dims):
    """
    Get chunking pattern from existing variables in the dataset.
    Returns dict of {dim: chunksize} for the specified dimensions.
    """
    chunks = {}
    for var in ds.data_vars.values():
        if hasattr(var.data, 'chunks'):
            var_chunks = dict(zip(var.dims, var.data.chunks))
            for dim in dims:
                if dim in var_chunks and dim not in chunks:
                    # Take first chunk size found for each dimension
                    chunks[dim] = var_chunks[dim][0]
        if all(dim in chunks for dim in dims):
            break
    return chunks or None

def encode_doys(ds, dim_order=('time', None, None), inplace=False):
    """
    Encode day of year as cyclical features and add to dataset,
    preserving existing chunking structure.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Input dataset containing time dimension
    dim_order : tuple, optional
        Dimension order for output arrays as (time_dim, y_dim, x_dim).
        Use None for automatic detection. Default: ('time', None, None)
    inplace : bool, optional
        If True, modify the dataset in place (default: False)
    
    Returns:
    --------
    xarray.Dataset
        Dataset with sin_doy and cos_doy variables added
    """
    
    if not inplace:
        ds = ds.copy()
    
    # Determine dimension names
    time_dim = dim_order[0] if dim_order[0] is not None else 'time'
    y_dim, x_dim = get_spatial_dims(ds) if dim_order[1] is None else (dim_order[1], dim_order[2])
    dims = (time_dim, y_dim, x_dim)
    
    # Get existing chunking pattern
    chunks = get_existing_chunks(ds, dims)
    
    # Compute day of the year
    doys = ds[time_dim].values.astype('datetime64[D]')
    doys = da.asarray([date.timetuple(doy.astype(object)).tm_yday for doy in doys])
    
    # Encode cyclical features
    sin_doy, cos_doy = encode_cyclical_features(doys, 365)
    
    # Repeat along spatial dimensions
    target_shape = tuple(len(ds[dim]) for dim in dims)
    repeat = int(np.prod(target_shape[1:]))  # y * x
    sin_doy = repeat_along_axis(sin_doy, repeat, 0).reshape(target_shape)
    cos_doy = repeat_along_axis(cos_doy, repeat, 0).reshape(target_shape)
    
    # Apply existing chunking pattern
    if chunks:
        current_chunks = {dims.index(dim): chunks[dim] for dim in dims if dim in chunks}
        sin_doy = sin_doy.rechunk(current_chunks)
        cos_doy = cos_doy.rechunk(current_chunks)
    
    # Add to dataset with attributes
    ds['sin_doy'] = (dims, sin_doy)
    ds['cos_doy'] = (dims, cos_doy)
    
    for name in ['sin_doy', 'cos_doy']:
        ds[name].attrs.update({
            'long_name': f"{'Sine' if 'sin' in name else 'Cosine'} of day of year",
            'units': 'unitless',
            'description': f"Cyclical encoding of day of year using {'sine' if 'sin' in name else 'cosine'} transform"
        })
    
    return ds

def pixel_regression(X_pixel, y_pixel, test_data):
    """Perform LGBM regression for a single pixel"""
    # Convert to numpy arrays (this will trigger computation for this pixel)
    X = X_pixel
    y = y_pixel
    
    # Remove NaN values
    mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
    X_clean = X[mask]
    y_clean = y[mask]
    
    if len(y_clean) < 10:  # Minimum samples threshold
        return np.nan * np.zeros((test_data.shape[0],))
    
    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42
    )
    
    # LGBM model
    params = {
        'verbose': -1,
    }
    
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse')
    
    # 3. Predict for each ensemble member
    predictions = np.zeros((test_data.shape[0], test_data.shape[1]))  # [time, ensemble_members]
    
    for ens in range(test_data.shape[1]):  # Loop over ensemble members
        test_ens = test_data[:, ens, :]  # [time, features] for this ensemble
        mask_test = ~np.any(np.isnan(test_ens), axis=1)
        
        if np.sum(mask_test) > 0:
            predictions[mask_test, ens] = model.predict(test_ens[mask_test])
        else:
            predictions[:, ens] = np.nan  # If all NaN, fill with NaN

    return predictions



cluster = LocalCluster(
    n_workers=7,  # Fewer workers = fewer WebSocket connections
    threads_per_worker=2,
    worker_dashboard_address=False,
    diagnostics_port=None# Disable per-worker dashboards
)
client = Client(cluster)    

# Opening Key Datasets

from openeo.local import LocalConnection

# Initialize the local connection
local_conn = LocalConnection("./")

# Define the STAC collection URL
stac_item = "https://stac.intertwin.fedcloud.eu/collections/ERA5_T2M_SSRD_TP"

# Specify the spatial extent (bounding box)
spatial_extent = {
    "west": 11.0,
    "east": 12.0,
    "south": 46.0,
    "north": 47.0
}

# Specify the temporal extent
temporal_extent = ["2000-01-01", "2020-12-31"]

# Load the data cube with specified parameters
era5_single = local_conn.load_stac(
    url=stac_item,
    spatial_extent=spatial_extent,
    temporal_extent=temporal_extent,
    bands=["data"]
).execute()

# Convert to xarray Dataset and select the 't2m' variable
era5_single = era5_single.to_dataset(dim='bands')["t2m"].to_dataset()

stac_item = "https://stac.intertwin.fedcloud.eu/collections/ERA5_PRESSURE"

from openeo.local import LocalConnection
local_conn = LocalConnection("./")

era5_pressure = local_conn.load_stac(
    url=stac_item,
    spatial_extent=spatial_extent,
    temporal_extent=temporal_extent,
    bands=["data"]
).execute()
era5_pressure = era5_pressure.sel(lon=slice(11, 12), lat=slice(47, 46)).to_dataset(dim='bands')

ERA5 = xr.merge([era5_single, era5_pressure])

stac_item = "https://stac.intertwin.fedcloud.eu/collections/EMO1_TA24_PR_RG_PET_DAILY"

from openeo.local import LocalConnection
local_conn = LocalConnection("./")

emo1 = local_conn.load_stac(
    url=stac_item,
    bands=["data"],
    spatial_extent=spatial_extent,
    temporal_extent=temporal_extent,
).execute()
EMO1 = emo1.sel(lon=slice(11, 12), lat=slice(47, 46)).to_dataset(dim='bands')["ta24"].to_dataset()

stac_item = "https://stac.intertwin.fedcloud.eu/collections/EMO1_DEM"

from openeo.local import LocalConnection
local_conn = LocalConnection("./")

dem = local_conn.load_stac(
    url=stac_item,
    spatial_extent=spatial_extent,
    bands=["data"]
).execute()
dem = dem.sel(lon=slice(11, 12), lat=slice(47, 46)).to_dataset(dim='bands')["dem"].to_dataset()



single = xr.open_zarr("/mnt/CEPH_PROJECTS/InterTwin/Climate_Downscaling/EMO1_DOWNSCALING/data/SEAS5_AUGUST_2021_SINGLE.zarr/", chunks={}).sel(lon=slice(11, 12), lat=slice(47, 46))
pressure = xr.open_zarr("/mnt/CEPH_PROJECTS/InterTwin/Climate_Downscaling/EMO1_DOWNSCALING/data/SEAS5_AUGUST_2021_PRESSURE.zarr/", chunks={}).sel(lon=slice(11, 12), lat=slice(47, 46))
SEAS5 = xr.merge([single["t2m"], pressure])


SEAS5_mid = match_to_mid_resolution(SEAS5, dem).astype('float32')
ERA5_mid = match_to_mid_resolution(ERA5, dem).astype('float32')
EMO1_mid =  match_to_mid_resolution(EMO1, dem).astype('float32')


# Example usage:
encode_doys(ERA5_mid_normalized, inplace=True)  # Modifies dataset in place
encode_doys(SEAS5_mid_normalized, inplace=True)  # Modifies dataset in place

# Assuming your datasets are already loaded as xarray objects
# train_X: xarray Dataset with variables as features (lat, lon, time)
# train_y: xarray DataArray with target variable (lat, lon, time)
# test_X: xarray Dataset with ensemble dimension (lat, lon, time, ensemble_member)

# Align all datasets to ensure consistent coordinates
train_X, train_y = xr.align(ERA5_mid_normalized, EMO1_mid)
#test_X = SEAS5_mid_normalized.reindex_like(train_X, method='nearest')

# Stack spatial dimensions for easier processing
train_X_stacked = train_X.stack(pixel=('lat', 'lon'))
train_y_stacked = train_y.stack(pixel=('lat', 'lon'))
test_X_stacked = SEAS5_mid_normalized.stack(pixel=('lat', 'lon'))

X_dask = train_X_stacked.to_array().data  # Already a Dask array
y_dask = train_y_stacked.to_array().data  # Already a Dask array
test_dask = test_X_stacked.to_array().data  # Already a Dask array

# Reshape for pixel processing
X_reshaped = X_dask.transpose(2, 1, 0).rechunk(chunks=(100, 7670, 8))  # (pixel, time, variable)
y_reshaped = y_dask.transpose(2, 1, 0).squeeze().rechunk(chunks=(100, 7670))     # (pixel, time)
test_reshaped = test_dask.transpose(3, 1, 2, 0).rechunk(chunks=(100, 216, 51, 8))  # (pixel, time, ensemble, variable)

X_reshaped = X_reshaped.persist()
y_reshaped = y_reshaped.persist()


# Map the regression function over all pixels
results = da.map_blocks(
    lambda x, y, t: np.array([pixel_regression(x[i], y[i], t[i]) 
                              for i in range(x.shape[0])]),
    X_reshaped,
    y_reshaped,
    test_reshaped[:,:,:2,:],
    dtype=float
)

with performance_report(filename="dask-report.html"):
    # Your computation here
    predictions = results.compute()

print("Completed Running the Script!")
