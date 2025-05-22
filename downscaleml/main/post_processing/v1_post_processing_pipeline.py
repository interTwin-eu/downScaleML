import os
import psutil
import logging
from pathlib import Path

import xarray as xr
import numpy as np
import dask.array as da
from dask.diagnostics import ProgressBar
from dask.distributed import LocalCluster, Client
import pyet
from openeo.local import LocalConnection


def setup_logger():
    logger = logging.getLogger("downscale_logger")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def remove_sea_areas(main_ds, mask_ds):
    mask_aligned = mask_ds.reindex_like(main_ds, method='nearest')
    land_mask = mask_aligned == 1
    masked_ds = main_ds.copy()
    for var in masked_ds.data_vars:
        masked_ds[var] = masked_ds[var].where(land_mask)
    return masked_ds


def main():
    logger = setup_logger()

    # Setup Dask cluster
    total_cores = psutil.cpu_count(logical=True)
    n_workers = max(1, total_cores // 2)
    threads_per_worker = max(1, total_cores // n_workers)

    logger.info(f"Setting up Dask cluster with {n_workers} workers and {threads_per_worker} threads per worker")
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        worker_dashboard_address=False,
        diagnostics_port=None
    )
    client = Client(cluster)

    logger.info("Dask cluster initialized")

    # Load DEM and create land-sea mask
    stac_item = "https://stac.intertwin.fedcloud.eu/collections/EMO1_DEM"
    spatial_extent = {"west": 4, "east": 16, "south": 42, "north": 51}
    local_conn = LocalConnection("./")

    dem = local_conn.load_stac(
        url=stac_item,
        spatial_extent=spatial_extent,
        bands=["dem"]
    ).execute()
    dem = dem.to_dataset(dim='bands')["dem"].to_dataset()
    dem = dem.isel(time=0)
    dem = dem.drop_vars("time").rename({'lat': 'y', 'lon': 'x'})
    dem = dem.sortby("y")
    dem_cleaned = dem.where(dem >= 0)
    land_sea_mask = xr.where(dem_cleaned.notnull(), 1, 0)
    land_sea_mask = land_sea_mask.rename({"dem": "land_sea_mask"})
    land_sea_mask = land_sea_mask.land_sea_mask.compute()

    logger.info("Land-sea mask generated")

    # Define base paths
    base_paths = {
        "t2m": Path("/mnt/CEPH_PROJECTS/InterTwin/Climate_Downscaling/EMO1_DOWNSCALING/stage_1/v3/t2m_v3"),
        "tp": Path("/mnt/CEPH_PROJECTS/InterTwin/Climate_Downscaling/EMO1_DOWNSCALING/stage_1/v3/tp_v3_1/tp"),
        "ssrd": Path("/mnt/CEPH_PROJECTS/InterTwin/Climate_Downscaling/EMO1_DOWNSCALING/stage_1/v3/ssrd_v3_1/ssrd")
    }

    # Get available month_years
    month_years = sorted([
        f.name.split("_seas5_")[1].replace(".zarr", "")
        for f in base_paths["t2m"].glob("t2m_seas5_*.zarr")
    ])

    combined_datasets = {}

    for my in month_years:
        datasets = []
        for var, path in base_paths.items():
            zarr_path = path / f"{var}_seas5_{my}.zarr"
            if zarr_path.exists():
                ds = xr.open_zarr(zarr_path)
                datasets.append(ds)
            else:
                logger.warning(f"Missing {zarr_path}")
        if datasets:
            combined = xr.merge(datasets)
            combined_datasets[my] = combined

    logger.info("All input datasets loaded and combined")

    # Process datasets
    for my, ds in combined_datasets.items():
        logger.info(f"Processing {my}...")

        t2m_c = ds['t2m'] - 273.15
        ssrd_mj = ds['ssrd'] / 1_000_000
        lat = ds['t2m'].coords['y']
        pet = pyet.jensen_haise(t2m_c, ssrd_mj, lat=lat)
        ds['pet'] = pet

        ds = ds.interp_like(dem)
        for var in ds.data_vars:
            ds[var] = ds[var].clip(min=0)

        ds = remove_sea_areas(ds, land_sea_mask)
        combined_datasets[my] = ds

    logger.info("All datasets processed with PET and masking")

    # Output directory and metadata
    output_dir = Path("/mnt/CEPH_PROJECTS/InterTwin/Climate_Downscaling/PAPER/v1/SEAS5/")
    output_dir.mkdir(parents=True, exist_ok=True)

    variable_units = {
        't2m': 'K',
        'tp': 'mm/day',
        'ssrd': 'J/m²/day',
        'pet': 'mm/day'
    }

    variable_names = {
        't2m': 'daily mean 2m_temperature',
        'tp': 'daily accumulated total precipitation',
        'ssrd': 'daily accumulated surface solar radiation downward',
        'pet': 'potential evapotranspiration calculated using jensen haise method using pyet package'
    }

    for my, ds in combined_datasets.items():
        zarr_path = output_dir / f"SEAS5_downscaled_{my}.zarr"
        logger.info(f"Saving downscaled dataset: {zarr_path}")

        for i, var_name in enumerate(ds.data_vars):
            logger.info(f"  Saving variable {i+1}/{len(ds.data_vars)}: {var_name}")
            var_ds = ds[[var_name]]
            var_ds[var_name].attrs['units'] = variable_units[var_name]
            var_ds[var_name].attrs['long_name'] = variable_names[var_name]
            mode = "w" if i == 0 else "a"
            var_ds.to_zarr(zarr_path, mode=mode, compute=True)

        logger.info(f"Completed saving {my}")

    logger.info("All datasets saved. Processing complete.")


if __name__ == "__main__":
    main()
