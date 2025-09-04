import os
import glob
import logging
from pathlib import Path
import joblib
import numpy as np
import pytest
import xarray as xr
from dotenv import load_dotenv
from dask.distributed import LocalCluster, Client
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from openeo.local import LocalConnection

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants shared across tests
# ──────────────────────────────────────────────────────────────────────────────
UUID = "pytest_100"  # Single source of truth for both tests
OUTPUT_ROOT = Path(
    "/app/test_data"
).resolve()  # pre-created in CI; still ensure existence
COLL_DIR = OUTPUT_ROOT / f"TEST_CUBE_ERA5_{UUID}"
EXPECTED_ZARR = COLL_DIR / f"TEST_CUBE_ERA5_{UUID}.zarr"
EXPECTED_COLLECTION = COLL_DIR / "collection.json"

SPATIAL = {"west": 11.0, "east": 11.5, "south": 46.0, "north": 46.5}
TEMPORAL = ["2018-01-01", "2018-02-01"]

BANDS = {
    "era5": ["t2m", "ssrd", "tp"],
    "pressure": ["t_850"],
    "dem": ["dem"],
    "emo1": ["ta24"],
}

STAC_URLS = {
    "ERA5_T2M_SSRD_TP": "https://stac.intertwin.fedcloud.eu/collections/ERA5_T2M_SSRD_TP",
    "ERA5_PRESSURE": "https://stac.intertwin.fedcloud.eu/collections/ERA5_PRESSURE",
    "EMO1_DEM": "https://stac.intertwin.fedcloud.eu/collections/EMO1_DEM",
    "EMO1_TA24_PR_RG_PET_DAILY": "https://stac.intertwin.fedcloud.eu/collections/EMO1_TA24_PR_RG_PET_DAILY",
}
PROCESSING_BANDS = ["sin_doy", "cos_doy"]
COLLECTION_URL_PREFIX = "https://stac.intertwin.fedcloud.eu/collections/"
DESCRIPTION = "Testing ERA5 raster2stac from client"
KEYWORDS = ["interTwin", "ERA5", "Zarr", "test"]


# ──────────────────────────────────────────────────────────────────────────────
# Pytest fixtures
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def dask_client():
    """Fixture to manage a small Dask cluster for the module."""
    cluster = LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        memory_limit="2GB",
        silence_logs=logging.ERROR,
        worker_dashboard_address=False,
        diagnostics_port=None,
    )
    client = Client(cluster)
    logger.info(f"Dask dashboard: {client.dashboard_link}")
    yield client
    client.close()
    cluster.close()


@pytest.fixture(scope="module")
def ensure_output_root():
    """Ensure output root exists and is writable."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    assert OUTPUT_ROOT.exists(), f"Missing {OUTPUT_ROOT}"
    assert os.access(OUTPUT_ROOT, os.W_OK), f"Not writable: {OUTPUT_ROOT}"
    logger.info(f"OUTPUT_ROOT = {OUTPUT_ROOT}")
    return OUTPUT_ROOT


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _glob_first(patterns):
    """Return the first match for given pattern(s); patterns can be str or list[str]."""
    if isinstance(patterns, str):
        patterns = [patterns]
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return Path(hits[0]).resolve()
    return None


def _discover_written_zarr():
    """
    Try the expected Zarr path first; fall back to a global search to handle
    implementations that ignore or reinterpret output_folder.
    """
    if EXPECTED_ZARR.exists():
        return EXPECTED_ZARR

    # Fallback: search widely
    pat1 = f"/**/TEST_CUBE_ERA5_{UUID}/TEST_CUBE_ERA5_{UUID}.zarr"
    pat2 = f"/**/TEST_CUBE_ERA5_{UUID}/*.zarr"
    found = _glob_first([pat1, pat2])
    if found:
        logger.info(f"Discovered Zarr at: {found}")
        return found

    return None


def _discover_collection_json():
    if EXPECTED_COLLECTION.exists():
        return EXPECTED_COLLECTION

    pat = f"/**/TEST_CUBE_ERA5_{UUID}/collection.json"
    found = _glob_first(pat)
    if found:
        logger.info(f"Discovered collection.json at: {found}")
        return found
    return None


def _sort_features_by_name(ds: xr.Dataset) -> xr.Dataset:
    return ds[sorted(ds.data_vars)]


def _temporal_split(ds: xr.Dataset):
    time_coord = ds["time"]
    return ds.sel(time=time_coord[:-1]), ds.sel(time=time_coord[-1:])


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────
def test_complete_processing_pipeline(dask_client, ensure_output_root):
    """
    Build ERA5/pressure/DEM/EMO1 cube, compute sin/cos DOY, and write STAC+Zarr via raster2stac.
    Validates presence of expected bands and confirms the write location.
    """
    # 1) Load via openEO local
    local_conn = LocalConnection("./")
    logger.info("OpenEO local connection established")

    era5_single = local_conn.load_stac(
        url=STAC_URLS["ERA5_T2M_SSRD_TP"],
        spatial_extent=SPATIAL,
        temporal_extent=TEMPORAL,
        bands=BANDS["era5"],
    )

    era5_pressure = local_conn.load_stac(
        url=STAC_URLS["ERA5_PRESSURE"],
        spatial_extent=SPATIAL,
        temporal_extent=TEMPORAL,
        bands=BANDS["pressure"],
    )

    emo1 = local_conn.load_stac(
        url=STAC_URLS["EMO1_TA24_PR_RG_PET_DAILY"],
        bands=BANDS["emo1"],
        spatial_extent=SPATIAL,
        temporal_extent=TEMPORAL,
    )

    dem = local_conn.load_stac(
        url=STAC_URLS["EMO1_DEM"],
        spatial_extent=SPATIAL,
        bands=BANDS["dem"],
    )

    logger.info("All datasets loaded successfully")

    # 2) Processing chain
    era5_cube = era5_single.merge_cubes(era5_pressure)
    remap = era5_cube.resample_cube_spatial(dem, method="bilinear")
    dem_expanded = dem.resample_cube_temporal(remap)
    cube = remap.merge_cubes(dem_expanded)

    emo1_renamed = emo1.rename_labels(
        dimension="bands",
        target=["target_dataset"],
        source=BANDS["emo1"],
    )
    recube = cube.merge_cubes(emo1_renamed)
    logger.info("Recube created successfully")

    processed = recube.process("sin_cos_doy", data=recube)
    merged = recube.merge_cubes(processed)
    merged = merged.rename_dimension(target="y", source="lat")
    merged = merged.rename_dimension(target="x", source="lon")

    # 3) raster2stac write (cwd-safe)
    output_folder = str(OUTPUT_ROOT)
    prev_cwd = os.getcwd()
    try:
        os.chdir(output_folder)
        logger.info(f"Changed CWD to: {os.getcwd()} (output_folder={output_folder})")

        era5_r2s = merged.process(
            "raster2stac",
            data=merged,
            item_id=f"TEST_CUBE_ERA5_{UUID}",
            collection_url=COLLECTION_URL_PREFIX,
            description=DESCRIPTION,
            write_collection_assets=True,
            keywords=KEYWORDS,
            s3_upload=False,
            post_to_stac=True,
            output_folder=".",  # ensure relative to /app/test_data
        )
        final_result = era5_r2s.execute()
    finally:
        os.chdir(prev_cwd)

    # 4) Validate write location
    zarr_path = _discover_written_zarr()
    assert zarr_path and zarr_path.exists(), f"Zarr not found for UUID={UUID}"
    logger.info(f".zarr output exists at {zarr_path}")

    # 5) Validate dataset content returned by process (if any)
    dataset_result = final_result.to_dataset(dim="bands")
    assert isinstance(dataset_result, xr.Dataset)

    expected_original = (
        BANDS["era5"] + BANDS["pressure"] + BANDS["dem"] + ["target_dataset"]
    )
    expected_bands = expected_original + PROCESSING_BANDS

    assert all(b in dataset_result.data_vars for b in expected_bands), (
        f"Missing bands. Have: {list(dataset_result.data_vars)}; "
        f"expected superset: {expected_bands}"
    )
    assert len(dataset_result.data_vars) == len(expected_bands)
    assert "time" in dataset_result.dims and len(dataset_result.time) > 0

    logger.info(
        "Pipeline completed successfully with all processing steps & validations"
    )


@pytest.mark.integration
def test_end_to_end_pixel_model_pipeline(dask_client, ensure_output_root):
    """
    Read the locally written STAC from the previous test, build a training set,
    fit per-pixel LGBM models, save scalers+models, and write test predictions to Zarr.
    """
    load_dotenv()

    # Prefer local STAC produced in test 1
    collection_json = _discover_collection_json()
    assert collection_json and collection_json.exists(), (
        f"Local STAC collection.json not found for UUID={UUID}. "
        f"Expected something like {EXPECTED_COLLECTION}"
    )

    spatial_extent = {"west": 11, "east": 11.5, "south": 46, "north": 46.5}
    target_var = "target_dataset"

    # Load STAC via openEO local runner
    local_conn = LocalConnection("./")
    train_xy = (
        local_conn.load_stac(
            url=str(collection_json),
            spatial_extent=spatial_extent,
            temporal_extent=TEMPORAL,
        )
        .execute()
        .to_dataset(dim="bands")
    )

    # Train/test split
    y = train_xy[[target_var]]
    X = _sort_features_by_name(train_xy.drop_vars(target_var))
    X_train, X_test = _temporal_split(X)
    y_train, _ = _temporal_split(y)

    # Convert to numpy
    X_train_np = X_train.astype(np.float32).to_array().values  # (features, time, y, x)
    y_train_np = y_train.astype(np.float32).to_array().values  # (1, time, y, x)
    y_coords = train_xy.y
    x_coords = train_xy.x

    models = {}
    for i in range(len(y_coords)):
        for j in range(len(x_coords)):
            X_pixel = X_train_np[:, :, i, j].T  # (time, features)
            y_pixel = y_train_np[0, :, i, j].ravel()

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
    results_dir = OUTPUT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    model_file = results_dir / f"{target_var}_models_scalers.joblib"
    joblib.dump(
        {"models": models, "y_coords": y_coords, "x_coords": x_coords}, model_file
    )
    assert model_file.exists(), "Model file was not written"

    # Predict on test
    X_test_time = X_test.time
    X_test_np = X_test.astype(np.float32).to_array().values
    test_preds = np.full(
        (len(X_test_time), len(y_coords), len(x_coords)), np.nan, dtype=np.float32
    )

    for (i, j), (model, scaler) in models.items():
        Xp = X_test_np[:, :, i, j].T  # (time, features)
        test_preds[:, i, j] = model.predict(scaler.transform(Xp))

    test_ds = xr.Dataset(
        {target_var: (("time", "y", "x"), test_preds)},
        coords={"time": X_test_time, "y": y_coords, "x": x_coords},
    )
    test_out = results_dir / f"{target_var}_test_predictions.zarr"
    test_ds.to_zarr(test_out, mode="w")
    assert test_out.exists(), "Prediction Zarr was not written"

    logger.info("End-to-end pixel model pipeline completed and artifacts stored.")
