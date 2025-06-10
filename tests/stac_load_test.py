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
            "dem": ["dem"]
        }
    }

def test_data_processing_pipeline(dask_client, test_parameters):
    """Test complete data loading and processing pipeline"""
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
            )
        }
        logger.info("All datasets loaded successfully")

        # Processing pipeline
        merged_era5 = datasets["era5"].merge_cubes(datasets["pressure"])
        resampled = merged_era5.resample_cube_spatial(datasets["dem"], method="bilinear")
        dem_temporal = datasets["dem"].resample_cube_temporal(resampled)
        final_cube = resampled.merge_cubes(dem_temporal).execute()

        # Validation
        result = final_cube.to_dataset(dim="bands")
        logger.info(f"Resulting dataset: {result}")

        # Core assertions
        assert isinstance(result, xr.Dataset)
        assert all(b in result.data_vars for b in test_parameters["bands"]["era5"] + 
                  test_parameters["bands"]["pressure"] + 
                  test_parameters["bands"]["dem"])
        assert result.dims["time"] == 3  # 3 days of data

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise