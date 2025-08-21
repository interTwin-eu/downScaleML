#!/usr/bin/env python3
import argparse
import xarray as xr
import matplotlib.pyplot as plt
from dask.distributed import LocalCluster, Client
import psutil
from openeo.local import LocalConnection
from downscaleml.core.core import match_to_mid_resolution, encode_doys
from typing import Dict, Tuple, Optional  # Updated import
import os
from dotenv import load_dotenv


def configure_cluster() -> Client:
    """Configure Dask cluster based on available resources."""
    total_cores = psutil.cpu_count(logical=True)
    n_workers = max(1, total_cores // 2)
    threads_per_worker = max(1, total_cores // n_workers)

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        worker_dashboard_address=False,
        diagnostics_port=None,
    )
    return Client(cluster)


def load_process_data(
    conn: LocalConnection,
    stac_url: str,
    spatial_extent: Dict[str, float],
    temporal_extent: Optional[Tuple[str, str]] = None,
    bands: Optional[list] = None,
) -> xr.Dataset:
    """Load and process data from STAC endpoint."""
    builder = conn.load_stac(url=stac_url, spatial_extent=spatial_extent)
    if temporal_extent:
        builder = builder.temporal_extent(temporal_extent)
    if bands:
        builder = builder.bands(bands)
    return builder.execute().to_dataset(dim="bands")


def process_dem(
    conn: LocalConnection, spatial_extent: Dict[str, float], use_xy_names: bool = False
) -> xr.Dataset:
    """Load and prepare DEM data."""
    dem = load_process_data(
        conn,
        "https://stac.intertwin.fedcloud.eu/collections/EMO1_DEM",
        spatial_extent,
        bands=["dem"],
    )
    dem_ds = (
        dem.to_dataset(dim="bands")["dem"].to_dataset().isel(time=0).drop_vars("time")
    )
    if use_xy_names:
        return dem_ds.rename({"lat": "y", "lon": "x"})
    return dem_ds


def process_era5(output_dir: str, spatial_extent: Dict[str, float]) -> None:
    """Process ERA5 data pipeline."""
    client = configure_cluster()
    conn = LocalConnection("./")

    try:
        print("Processing ERA5 data...")
        dem = process_dem(conn, spatial_extent, use_xy_names=False)

        era5_single = load_process_data(
            conn,
            "https://stac.intertwin.fedcloud.eu/collections/ERA5_T2M_SSRD_TP",
            spatial_extent,
            temporal_extent=("2000-01-01", "2020-12-31"),
        )

        era5_pressure = load_process_data(
            conn,
            "https://stac.intertwin.fedcloud.eu/collections/ERA5_PRESSURE",
            spatial_extent,
            temporal_extent=("2000-01-01", "2020-12-31"),
        )

        era5 = xr.merge([era5_single, era5_pressure])
        era5_interp = match_to_mid_resolution(era5, dem)
        encode_doys(era5_interp, inplace=True)

        era5_interp.chunk({"time": 1000, "lat": 50, "lon": 50}).to_zarr(
            f"{output_dir}/ERA5_to_latent.zarr", mode="w", compute=True
        )
        print("ERA5 processing completed successfully.")

    finally:
        client.close()


def process_seas5(
    output_dir: str, spatial_extent: Dict[str, float], month_year: Optional[str] = None
) -> None:
    """Process SEAS5 data pipeline."""
    client = configure_cluster()
    conn = LocalConnection("./")

    try:
        print("Processing SEAS5 data...")
        dem = process_dem(conn, spatial_extent, use_xy_names=True)

        months = [
            "AUGUST",
            "SEPTEMBER",
            "OCTOBER",
            "NOVEMBER",
            "DECEMBER",
            "JANUARY",
            "FEBRUARY",
            "MARCH",
            "APRIL",
            "MAY",
            "JUNE",
            "JULY",
        ]

        # Filter months if month_year is specified
        if month_year:
            month, year = month_year.split("_")
            months = [month.upper()]
            years = [year]
        else:
            years = ["2021", "2022"]

        for month in months:
            for year in years:
                # Skip invalid combinations
                if (
                    month in ["AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"]
                    and year != "2021"
                ):
                    continue
                if (
                    month
                    not in ["AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"]
                    and year != "2022"
                ):
                    continue

                print(f"Processing {month} {year}...")

                single = load_process_data(
                    conn,
                    f"https://stac.intertwin.fedcloud.eu/collections/SINGLE_LEVELS_DAILY_SEAS5_{month}_{year}",
                    spatial_extent,
                )

                pressure = load_process_data(
                    conn,
                    f"https://stac.intertwin.fedcloud.eu/collections/PRESSURE_LEVELS_DAILY_SEAS5_{month}_{year}",
                    spatial_extent,
                )

                seas5 = xr.merge([single, pressure]).drop_vars("pet", errors="ignore")
                seas5_interp = match_to_mid_resolution(seas5, dem, "y", "x")
                encode_doys(seas5_interp, inplace=True)

                seas5_interp.chunk(
                    {"time": 1000, "lat": 50, "lon": 50, "number": 1}
                ).to_zarr(
                    f"{output_dir}/SEAS5_to_latent_{month}_{year}.zarr",
                    mode="w",
                    compute=True,
                )

            print("SEAS5 processing completed successfully.")

    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description="Climate Data Preprocessing Pipeline")
    parser.add_argument("--era5", action="store_true", help="Process ERA5 data")
    parser.add_argument("--seas5", action="store_true", help="Process SEAS5 data")
    parser.add_argument(
        "--month-year",
        type=str,
        help="Process specific month_year for SEAS5 (e.g., AUGUST_2021)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/CEPH_PROJECTS/InterTwin/Climate_Downscaling/EMO1_DOWNSCALING/stage_1",
        help="Output directory for processed data",
    )
    args = parser.parse_args()

    # Load environment variables from .env file
    load_dotenv()

    # Access AWS credentials
    access_key = os.getenv("ACCESS_KEY")
    secret_key = os.getenv("SECRET_KEY")

    spatial_extent = {"west": 2, "east": 20, "south": 40, "north": 52}

    if not (args.era5 or args.seas5):
        print("Please specify at least one dataset to process (--era5 and/or --seas5)")
        return

    if args.era5:
        process_era5(args.output_dir, spatial_extent)

    if args.seas5:
        process_seas5(args.output_dir, spatial_extent, args.month_year)


if __name__ == "__main__":
    main()
