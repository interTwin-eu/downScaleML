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
        silence_logs=logging.ERROR
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
        }
    }

def test_data_processing_pipeline(dask_client, test_parameters):
    """Test complete data loading and processing pipeline including recube and EMO1 renaming"""
    try:
        # Initialize connection
        local_conn = LocalConnection("./")
        logger.info("OpenEO local connection established")

        # Load datasets (reduced scope)
        datasets = {
            "era5": local_conn.load_stac(
                url="https://stac.intertwin.fedcloud.eu/collections/ERA5_T2M_SSRD_TP",
                spatial_extent=test_parameters["spatial"],
                temporal_extent=test_parameters["temporal"],
                bands=test_parameters["bands"]["era5"]
            ),
            "pressure": local_conn.load_stac(
                url="https://stac.intertwin.fedcloud.eu/collections/ERA5_PRESSURE",
                spatial_extent=test_parameters["spatial"],
                temporal_extent=test_parameters["temporal"],
                bands=test_parameters["bands"]["pressure"]
            ),
            "dem": local_conn.load_stac(
                url="https://stac.intertwin.fedcloud.eu/collections/EMO1_DEM",
                spatial_extent=test_parameters["spatial"],
                bands=test_parameters["bands"]["dem"]
            ),
            "emo1": local_conn.load_stac(
                url="https://stac.intertwin.fedcloud.eu/collections/EMO1_TA24_PR_RG_PET_DAILY",
                spatial_extent=test_parameters["spatial"],
                temporal_extent=test_parameters["temporal"],
                bands=test_parameters["bands"]["emo1"]
            )
        }
        logger.info("All datasets loaded successfully")

        # Processing pipeline
        merged_era5 = datasets["era5"].merge_cubes(datasets["pressure"])
        resampled = merged_era5.resample_cube_spatial(datasets["dem"], method="bilinear")
        dem_temporal = datasets["dem"].resample_cube_temporal(resampled)
        intermediate_cube = resampled.merge_cubes(dem_temporal)
        
        # EMO1 renaming and recube
        emo1_renamed = datasets["emo1"].rename_labels(
            dimension="bands",
            target=["target_dataset"],
            source=test_parameters["bands"]["emo1"]
        )
        logger.info("Renamed EMO1 dataset bands")
        
        final_cube = intermediate_cube.merge_cubes(emo1_renamed)
        result = final_cube.execute()
        
        # Validation
        dataset_result = result.to_dataset(dim="bands")
        logger.info(f"Resulting dataset: {dataset_result}")

        # Core assertions
        assert isinstance(dataset_result, xr.Dataset)
        expected_bands = (
            test_parameters["bands"]["era5"] +
            test_parameters["bands"]["pressure"] +
            test_parameters["bands"]["dem"] +
            ["target_dataset"]
        )
        assert all(b in dataset_result.data_vars for b in expected_bands)
        
        # Additional assertions for recube
        assert "target_dataset" in dataset_result.data_vars
        assert len(dataset_result.data_vars) == len(expected_bands)

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise