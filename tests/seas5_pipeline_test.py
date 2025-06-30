# tests/test_seas5_data_loading.py
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
        n_workers=2,
        threads_per_worker=1,
        memory_limit='2GB',
        silence_logs=logging.ERROR,
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
        "processing_bands": ["sin_doy", "cos_doy"]  # Expected output bands from sin_cos_doy
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
        
        # Apply sin_cos_doy processing and merge results
        processed = seas5cube.process("sin_cos_doy", data=seas5cube)
        merged_seas5_cube = seas5cube.merge_cubes(processed)
        final_result = merged_seas5_cube.execute()
        logger.info("Final merged SEAS5 cube with processing results")

        # Validation
        dataset_result = final_result.to_dataset(dim="bands")
        logger.info(f"Final SEAS5 dataset: {dataset_result}")

        # Core assertions
        assert isinstance(dataset_result, xr.Dataset)
        
        # Check all original bands are present
        expected_original_bands = (
            test_parameters["bands"]["single"] +
            test_parameters["bands"]["pressure"] +
            test_parameters["bands"]["dem"]
        )
        
        # Check processing output bands are present
        expected_bands = expected_original_bands + test_parameters["processing_bands"]
        
        # Verify all expected bands exist in the result
        assert all(b in dataset_result.data_vars for b in expected_bands)
        
        # Verify dimension renaming was successful
        assert "x" in dataset_result.dims
        assert "y" in dataset_result.dims
        assert "lat" not in dataset_result.dims
        assert "lon" not in dataset_result.dims
        
        # Verify temporal dimension
        assert len(dataset_result.time) == 2  # 3 days in temporal extent
        
        logger.info("SEAS5 pipeline completed successfully with all processing steps")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise