import os
import pytest
import xarray as xr
import numpy as np
import joblib
from pathlib import Path
from dotenv import load_dotenv
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
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
        n_workers=13,
        threads_per_worker=1,
        memory_limit="3GB",
        silence_logs=logging.ERROR,
        worker_dashboard_address=False,
        diagnostics_port=None,
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
        "temporal": ["2018-01-01", "2018-02-01"],
        "seas5_temporal": ["2021-08-01", "2021-08-03"],
        "bands": {
            "era5": ["t2m", "ssrd", "tp"],
            "pressure": ["t_850"],
            "dem": ["dem"],
            "emo1": ["ta24"],
        },
        "stac_urls": {
            "ERA5_T2M_SSRD_TP": "https://stac.intertwin.fedcloud.eu/collections/ERA5_T2M_SSRD_TP",
            "ERA5_PRESSURE": "https://stac.intertwin.fedcloud.eu/collections/ERA5_PRESSURE",
            "EMO1_DEM": "https://stac.intertwin.fedcloud.eu/collections/EMO1_DEM",
            "EMO1_TA24_PR_RG_PET_DAILY": "https://stac.intertwin.fedcloud.eu/collections/EMO1_TA24_PR_RG_PET_DAILY",
            "SEAS5_SINGLE": "https://stac.intertwin.fedcloud.eu/collections/SINGLE_LEVELS_DAILY_SEAS5_AUGUST_2021",
            "SEAS5_PRESSURE": "https://stac.intertwin.fedcloud.eu/collections/PRESSURE_LEVELS_DAILY_SEAS5_AUGUST_2021",
        },
        "processing_bands": [
            "sin_doy",
            "cos_doy",
        ],  # Expected output bands from sin_cos_doy
        "raster_stac": {
            "uuid": "pytest_101",
            "collection_url": "https://stac.intertwin.fedcloud.eu/collections/",
            "description": "Testing ERA5 raster2stac from client",
            "keywords": ["interTwin", "ERA5", "Zarr", "test"],
            "seas5_uuid": "pytests_101",
            "s3_config": {
                "endpoint_url": "https://objectstore.eodc.eu:2222",
                "bucket_name": "rucio",
                "file_prefix": "interTwin_EURAC/",
            },
        },
    }


def test_era5_processing_pipeline(dask_client, test_parameters):
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
            bands=test_parameters["bands"]["era5"],
        )

        era5_pressure = local_conn.load_stac(
            url=test_parameters["stac_urls"]["ERA5_PRESSURE"],
            spatial_extent=test_parameters["spatial"],
            temporal_extent=test_parameters["temporal"],
            bands=test_parameters["bands"]["pressure"],
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
            bands=test_parameters["bands"]["dem"],
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
            source=test_parameters["bands"]["emo1"],
        )
        recube = cube.merge_cubes(emo1_renamed)
        logger.info("Recube created successfully")

        # Apply sin_cos_doy processing
        processed = recube.process("sin_cos_doy", data=recube)

        # Final merge step
        merged = recube.merge_cubes(processed)
        merged = merged.rename_dimension(target="y", source="lat")
        merged = merged.rename_dimension(target="x", source="lon")

        # Prepare output path
        output_path = f"/app/test_data/"
        zarr_path = os.path.join(
            output_path,
            f"TEST_CUBE_ERA5_{test_parameters['raster_stac']['uuid']}/TEST_CUBE_ERA5_{test_parameters['raster_stac']['uuid']}.zarr",
        )

        # Apply raster2stac processing
        era5_r2s = merged.process(
            "raster2stac",
            data=merged,
            item_id=f"TEST_CUBE_ERA5_{test_parameters['raster_stac']['uuid']}",
            collection_url=test_parameters["raster_stac"]["collection_url"],
            description=test_parameters["raster_stac"]["description"],
            write_collection_assets=True,
            keywords=test_parameters["raster_stac"]["keywords"],
            s3_upload=False,
            post_to_stac=True,
            output_folder=output_path,
        )

        final_result = era5_r2s.execute()
        logger.info("Final merge completed successfully")

        # Validation
        assert os.path.exists(zarr_path)
        logger.info(f".zarr output exists at {zarr_path}")

        dataset_result = final_result.to_dataset(dim="bands")
        logger.info(f"Final merged dataset: {dataset_result}")

        # Core assertions
        assert isinstance(dataset_result, xr.Dataset)

        # Check all original bands are present
        expected_original_bands = (
            test_parameters["bands"]["era5"]
            + test_parameters["bands"]["pressure"]
            + test_parameters["bands"]["dem"]
            + ["target_dataset"]
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


def test_seas5_processing_pipeline(dask_client, test_parameters):
    """Test complete SEAS5 data processing pipeline"""
    try:
        # Initialize connection
        local_conn = LocalConnection("./")
        logger.info("OpenEO local connection established")

        # Load SEAS5 datasets
        seas5_single = local_conn.load_stac(
            url=test_parameters["stac_urls"]["SEAS5_SINGLE"],
            spatial_extent=test_parameters["spatial"],
            temporal_extent=test_parameters["seas5_temporal"],
            bands=test_parameters["bands"][
                "era5"
            ],  # Using same bands as ERA5 for consistency
        )

        seas5_pressure = local_conn.load_stac(
            url=test_parameters["stac_urls"]["SEAS5_PRESSURE"],
            spatial_extent=test_parameters["spatial"],
            temporal_extent=test_parameters["seas5_temporal"],
            bands=test_parameters["bands"]["pressure"],
        )

        # Load DEM
        dem = local_conn.load_stac(
            url=test_parameters["stac_urls"]["EMO1_DEM"],
            spatial_extent=test_parameters["spatial"],
            bands=test_parameters["bands"]["dem"],
        )
        logger.info("All SEAS5 datasets loaded successfully")

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
        # output_path = f"/home/sdhinakaran/test/"
        zarr_path = os.path.join(
            output_path,
            f"TEST_CUBE_SEAS5_{test_parameters['raster_stac']['seas5_uuid']}/TEST_CUBE_SEAS5_{test_parameters['raster_stac']['seas5_uuid']}.zarr",
        )

        # Apply sin_cos_doy processing and merge results
        processed = seas5cube.process("sin_cos_doy", data=seas5cube)
        merged_seas5_cube = seas5cube.merge_cubes(processed)

        # Apply raster2stac processing
        seas_r2s = merged_seas5_cube.process(
            "raster2stac",
            data=merged_seas5_cube,
            item_id=f"TEST_CUBE_SEAS5_{test_parameters['raster_stac']['seas5_uuid']}",
            collection_url=test_parameters["raster_stac"]["collection_url"],
            description=test_parameters["raster_stac"]["description"],
            write_collection_assets=True,
            keywords=test_parameters["raster_stac"]["keywords"],
            s3_upload=False,
            s3_endpoint_url=test_parameters["raster_stac"]["s3_config"]["endpoint_url"],
            bucket_name=test_parameters["raster_stac"]["s3_config"]["bucket_name"],
            bucket_file_prefix=test_parameters["raster_stac"]["s3_config"][
                "file_prefix"
            ],
            post_to_stac=True,
            output_folder=output_path,
        )

        final_result = seas_r2s.execute()
        logger.info("Final merged SEAS5 cube with processing results")

        assert os.path.exists(zarr_path)
        logger.info(f".zarr output exists at {zarr_path}")

        dataset_result = final_result.to_dataset(dim="bands")
        logger.info(f"Final merged SEAS5 dataset: {dataset_result}")

        # Core assertions
        assert isinstance(dataset_result, xr.Dataset)
        assert "number" in dataset_result.dims  # SEAS5 specific dimension
        assert len(dataset_result.number) > 0  # Should have ensemble members

        logger.info("SEAS5 pipeline completed successfully with all processing steps")

    except Exception as e:
        logger.error(f"SEAS5 test failed: {str(e)}")
        raise


def sort_features_by_name(ds):
    return ds[sorted(ds.data_vars)]


def temporal_split(ds):
    time_coord = ds["time"]
    return ds.sel(time=time_coord[:-1]), ds.sel(time=time_coord[-1:])


@pytest.mark.integration
def test_end_to_end_pixel_model_pipeline(dask_client, test_parameters):
    """Test end-to-end pixel-based modeling pipeline with both ERA5 and SEAS5"""
    load_dotenv()
    access_key = os.getenv("ACCESS_KEY")
    secret_key = os.getenv("SECRET_KEY")

    STAC_URLS = {
        "train_xy": f"https://stac.intertwin.fedcloud.eu/collections/TEST_CUBE_ERA5_{test_parameters['raster_stac']['uuid']}",
        "test_x": f"https://stac.intertwin.fedcloud.eu/collections/TEST_CUBE_SEAS5_{test_parameters['raster_stac']['seas5_uuid']}",
    }
    spatial_extent = test_parameters["spatial"]
    target_var = "target_dataset"
    output_dir = Path("/app/test_data/results/")
    output_dir.mkdir(exist_ok=True)

    # Load STAC via openEO
    local_conn = LocalConnection("./")
    train_xy = (
        local_conn.load_stac(
            url=STAC_URLS["train_xy"],
            spatial_extent=spatial_extent,
            temporal_extent=test_parameters["temporal"],
        )
        .execute()
        .to_dataset(dim="bands")
    )

    test_x = (
        local_conn.load_stac(
            url=STAC_URLS["test_x"],
            spatial_extent=spatial_extent,
            temporal_extent=test_parameters["seas5_temporal"],
        )
        .execute()
        .to_dataset(dim="bands")
    )

    # Train/test split
    y = train_xy[[target_var]]
    X = sort_features_by_name(train_xy.drop_vars(target_var))
    X_train, X_test = temporal_split(X)
    y_train, _ = temporal_split(y)

    # Convert to numpy
    X_train = (
        X_train.astype(np.float32).to_array().values
    )  # shape: (features, time, y, x)
    y_train = y_train.astype(np.float32).to_array().values  # shape: (1, time, y, x)
    y_coords = train_xy.y
    x_coords = train_xy.x

    models = {}
    for i in range(len(y_coords)):
        for j in range(len(x_coords)):
            X_pixel = X_train[:, :, i, j].T
            y_pixel = y_train[0, :, i, j].ravel()

            valid = ~np.isnan(X_pixel).any(axis=1) & ~np.isnan(y_pixel)
            X_pixel = X_pixel[valid]
            y_pixel = y_pixel[valid]
            if len(X_pixel) < 10:
                continue

            scaler = StandardScaler().fit(X_pixel)
            model = LGBMRegressor(verbose=-1)
            model.fit(scaler.transform(X_pixel), y_pixel)
            models[(i, j)] = (model, scaler)

    # Save models
    model_file = output_dir / f"{target_var}_models_scalers.joblib"
    joblib.dump(
        {"models": models, "y_coords": y_coords, "x_coords": x_coords}, model_file
    )
    assert model_file.exists()

    # Predict on test
    X_test_time = X_test.time
    X_test_np = X_test.astype(np.float32).to_array().values
    test_preds = np.full((len(X_test_time), len(y_coords), len(x_coords)), np.nan)

    for (i, j), (model, scaler) in models.items():
        X = X_test_np[:, :, i, j].T
        test_preds[:, i, j] = model.predict(scaler.transform(X))

    test_ds = xr.Dataset(
        {target_var: (("time", "y", "x"), test_preds)},
        coords={"time": X_test_time, "y": y_coords, "x": x_coords},
    )
    test_out = output_dir / f"{target_var}_test_predictions.zarr"
    test_ds.to_zarr(test_out, mode="w")
    assert test_out.exists()

    # SEAS5 forecasts
    test_x = sort_features_by_name(test_x)
    seas5_np = test_x.to_array().values  # shape: (features, time, number, y, x)

    seas5_preds = np.full(
        (len(test_x.time), len(test_x.number), len(test_x.y), len(test_x.x)), np.nan
    )

    for m in range(len(test_x.number)):
        for (i, j), (model, scaler) in models.items():
            if i >= len(test_x.y) or j >= len(test_x.x):
                continue
            X = seas5_np[:, :, m, i, j].T
            seas5_preds[:, m, i, j] = model.predict(scaler.transform(X))

    seas5_ds = xr.Dataset(
        {target_var: (("time", "number", "y", "x"), seas5_preds)},
        coords={
            "time": test_x.time,
            "number": test_x.number,
            "y": test_x.y,
            "x": test_x.x,
        },
    )
    seas5_out = output_dir / f"{target_var}_seas5_forecast.zarr"
    seas5_ds.to_zarr(seas5_out, mode="w")
    assert seas5_out.exists()

    logger.info("End-to-end pipeline ran successfully with both ERA5 and SEAS5 outputs")
