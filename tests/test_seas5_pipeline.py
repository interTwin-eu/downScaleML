# tests/test_seas5_data_loading.py
import os
import pytest
import xarray as xr
from dask.distributed import LocalCluster, Client
from openeo.local import LocalConnection
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for testing
STAC_URLS = {
    "SEAS5_SINGLE": "https://stac.intertwin.fedcloud.eu/collections/SINGLE_LEVELS_DAILY_SEAS5_{init}",
    "SEAS5_PRESSURE": "https://stac.intertwin.fedcloud.eu/collections/PRESSURE_LEVELS_DAILY_SEAS5_{init}",
    "EMO1_DEM": "https://stac.intertwin.fedcloud.eu/collections/EMO1_DEM"
}

@pytest.fixture(scope="module")
def dask_client():
    """Fixture to manage Dask cluster lifecycle"""
    cluster = LocalCluster(
        n_workers=14,
        threads_per_worker=1,
        worker_dashboard_address=False,
        diagnostics_port=None
    )
    client = Client(cluster)
    logger.info(f"Dask dashboard available at: {client.dashboard_link}")
    yield client
    client.close()
    cluster.close()

@pytest.fixture(scope="module")
def test_parameters():
    """Shared test parameters with reduced scope"""
    return {
        "init": "AUGUST_2021",  # Initialization date for SEAS5
        "spatial": {"west": 11.0, "east": 11.5, "south": 46.0, "north": 46.5},
        "temporal": ["2021-08-01", "2021-08-03"],
        "bands": {
            "single": ["ssrd", "t2m", "tp"],
            "pressure": ["t_850"],
            "dem": ["dem"]
        },
        "processing_bands": ["sin_doy", "cos_doy"],  # Expected output bands from sin_cos_doy
        "raster_stac": {
            "uuid": "pytests_100",
            "collection_url": "https://stac.intertwin.fedcloud.eu/collections/",
            "description": "Testing raster2stac from client",
            "keywords": ["interTwin", "Zarr", "test"],
            "s3_config": {
                "endpoint_url": "https://objectstore.eodc.eu:2222",
                "bucket_name": "rucio",
                "file_prefix": "interTwin_EURAC/"
            }
        }
    }

def test_seas5_processing_pipeline(dask_client, test_parameters):
    """Test complete SEAS5 data processing pipeline"""
    try:
        # Initialize connection
        local_conn = LocalConnection("./")
        logger.info("OpenEO local connection established")

        # Load SEAS5 datasets
        seas5_single = local_conn.load_stac(
            url=STAC_URLS["SEAS5_SINGLE"].format(init=test_parameters["init"]),
            spatial_extent=test_parameters["spatial"],
            temporal_extent=test_parameters["temporal"],
            bands=test_parameters["bands"]["single"]
        )
        
        seas5_pressure = local_conn.load_stac(
            url=STAC_URLS["SEAS5_PRESSURE"].format(init=test_parameters["init"]),
            spatial_extent=test_parameters["spatial"],
            temporal_extent=test_parameters["temporal"],
            bands=test_parameters["bands"]["pressure"]
        )

        # Load DEM
        dem = local_conn.load_stac(
            url=STAC_URLS["EMO1_DEM"],
            spatial_extent=test_parameters["spatial"],
            bands=test_parameters["bands"]["dem"]
        )
        logger.info("All datasets loaded successfully")

        # Processing pipeline
        seas5_cube = seas5_single.merge_cubes(seas5_pressure)
        seas5_remap = seas5_cube.resample_cube_spatial(dem, method="bilinear")
        
        # Temporal resampling and dimension renaming
        dem_expanded = dem.resample_cube_temporal(seas5_remap)
        dem_expanded = dem_expanded.rename_dimension(target="y", source="lat")
        dem_expanded = dem_expanded.rename_dimension(target="x", source="lon")
        
        # Merge with DEM
        seas5cube = seas5_remap.merge_cubes(dem_expanded)
        logger.info("SEAS5 cube with DEM merged successfully")

        output_path = f"/app/test_data/"
        #output_path = f"/home/sdhinakaran/test/"
        zarr_path = os.path.join(output_path, f"TEST_CUBE_SEAS5_{test_parameters['raster_stac']['uuid']}/TEST_CUBE_SEAS5_{test_parameters['raster_stac']['uuid']}.zarr")
        
        # Apply sin_cos_doy processing and merge results
        processed = seas5cube.process("sin_cos_doy", data=seas5cube)
        merged_seas5_cube = seas5cube.merge_cubes(processed)
        seas_r2s = merged_seas5_cube.process(
            "raster2stac",
            data=merged_seas5_cube,
            item_id=f"TEST_CUBE_SEAS5_{test_parameters['raster_stac']['uuid']}",
            collection_url=test_parameters["raster_stac"]["collection_url"],
            description=test_parameters["raster_stac"]["description"],
            write_collection_assets=True,
            keywords=test_parameters["raster_stac"]["keywords"],
            s3_upload=False,
            s3_endpoint_url=test_parameters["raster_stac"]["s3_config"]["endpoint_url"],
            bucket_name=test_parameters["raster_stac"]["s3_config"]["bucket_name"],
            bucket_file_prefix=test_parameters["raster_stac"]["s3_config"]["file_prefix"],
            post_to_stac=True,
            output_folder=output_path
        )
        print(f"[DEBUG] Writing output to: {output_path}")
        
        final_result = seas_r2s.execute()
        logger.info("Final merged SEAS5 cube with processing results")

        assert os.path.exists(zarr_path)
        logger.info(f".zarr output exists at {zarr_path}")

        logger.info("SEAS5 pipeline completed successfully with all processing steps")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise