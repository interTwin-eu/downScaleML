# tests/test_era5_data_loading.py
import pytest
import xarray as xr
from dask.distributed import LocalCluster, Client
from openeo.local import LocalConnection
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        "spatial": {"west": 11.0, "east": 11.5, "south": 46.0, "north": 46.5},
        "temporal": ["2018-01-01", "2018-01-03"],
        "bands": {
            "era5": ["t2m", "ssrd", "tp"],
            "pressure": ["t_850"],
            "dem": ["dem"],
            "emo1": ["ta24"]
        },
        "stac_urls": {
            "ERA5_T2M_SSRD_TP": "https://stac.intertwin.fedcloud.eu/collections/ERA5_T2M_SSRD_TP",
            "ERA5_PRESSURE": "https://stac.intertwin.fedcloud.eu/collections/ERA5_PRESSURE",
            "EMO1_DEM": "https://stac.intertwin.fedcloud.eu/collections/EMO1_DEM",
            "EMO1_TA24_PR_RG_PET_DAILY": "https://stac.intertwin.fedcloud.eu/collections/EMO1_TA24_PR_RG_PET_DAILY"
        },
        "processing_bands": ["sin_doy", "cos_doy"]  # Expected output bands from sin_cos_doy
    }

def test_complete_processing_pipeline(dask_client, test_parameters):
    """Test complete data processing pipeline from loading to final merged result"""
    try:
        # Initialize connection
        local_conn = LocalConnection("./")
        logger.info("OpenEO local connection established")

        # Load datasets (reduced scope)
        era5_single = local_conn.load_stac(
            url=test_parameters["stac_urls"]["ERA5_T2M_SSRD_TP"],
            spatial_extent=test_parameters["spatial"],
            temporal_extent=test_parameters["temporal"],
            bands=test_parameters["bands"]["era5"]
        )
        
        era5_pressure = local_conn.load_stac(
            url=test_parameters["stac_urls"]["ERA5_PRESSURE"],
            spatial_extent=test_parameters["spatial"],
            temporal_extent=test_parameters["temporal"],
            bands=test_parameters["bands"]["pressure"]
        )
        
        emo1 = local_conn.load_stac(
            url=test_parameters["stac_urls"]["EMO1_TA24_PR_RG_PET_DAILY"],
            bands=test_parameters["bands"]["emo1"],
            spatial_extent=test_parameters["spatial"],
            temporal_extent=test_parameters["temporal"],
        )

        dem = local_conn.load_stac(
            url=test_parameters["stac_urls"]["EMO1_DEM"],
            spatial_extent=test_parameters["spatial"],
            bands=test_parameters["bands"]["dem"]
        )
        logger.info("All datasets loaded successfully")

        # Processing pipeline
        era5_cube = era5_single.merge_cubes(era5_pressure)
        remap = era5_cube.resample_cube_spatial(dem, method="bilinear")
        dem_expanded = dem.resample_cube_temporal(remap)
        cube = remap.merge_cubes(dem_expanded)
        
        # EMO1 renaming and recube
        emo1_renamed = emo1.rename_labels(
            dimension="bands",
            target=["target_dataset"],
            source=test_parameters["bands"]["emo1"]
        )
        recube = cube.merge_cubes(emo1_renamed)
        logger.info("Recube created successfully")
        
        # Apply sin_cos_doy processing
        processed = recube.process("sin_cos_doy", data=recube)
        
        # Final merge step
        merged = recube.merge_cubes(processed)
        final_result = merged.execute()
        logger.info("Final merge completed successfully")
        
        # Validation
        dataset_result = final_result.to_dataset(dim="bands")
        logger.info(f"Final merged dataset: {dataset_result}")

        # Core assertions
        assert isinstance(dataset_result, xr.Dataset)
        
        # Check all original bands are present
        expected_original_bands = (
            test_parameters["bands"]["era5"] +
            test_parameters["bands"]["pressure"] +
            test_parameters["bands"]["dem"] +
            ["target_dataset"]
        )
        
        # Check processing output bands are present
        expected_bands = expected_original_bands + test_parameters["processing_bands"]
        
        # Verify all expected bands exist in the result
        assert all(b in dataset_result.data_vars for b in expected_bands)
        
        # Verify no duplicate bands
        assert len(dataset_result.data_vars) == len(expected_bands)
        
        # Check temporal dimension
        assert "time" in dataset_result.dims
        assert len(dataset_result.time) > 0
        
        logger.info("Pipeline completed successfully with all processing steps")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise