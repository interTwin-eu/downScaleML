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


def sort_features_by_name(ds):
    return ds[sorted(ds.data_vars)]


def temporal_split(ds):
    time_coord = ds["time"]
    return ds.sel(time=time_coord[:-1]), ds.sel(time=time_coord[-1:])


@pytest.mark.integration
def test_end_to_end_pixel_model_pipeline():
    load_dotenv()
    access_key = os.getenv("ACCESS_KEY")
    secret_key = os.getenv("SECRET_KEY")

    STAC_URLS = {
        "train_xy": "https://stac.intertwin.fedcloud.eu/collections/TEST_CUBE_ERA5_pytest_100",
        "test_x": "https://stac.intertwin.fedcloud.eu/collections/TEST_CUBE_SEAS5_pytests_100",
    }
    spatial_extent = {"west": 11, "east": 11.5, "south": 46, "north": 46.5}
    target_var = "target_dataset"
    output_dir = Path("/app/test_data/results/")
    output_dir.mkdir(exist_ok=True)

    # Dask cluster
    cluster = LocalCluster(
        n_workers=7, threads_per_worker=2, memory_limit="3GB", dashboard_address=None
    )
    client = Client(cluster)

    # Load STAC via openEO
    local_conn = LocalConnection("./")
    train_xy = (
        local_conn.load_stac(
            url=STAC_URLS["train_xy"],
            spatial_extent=spatial_extent,
            temporal_extent=["2018-01-01", "2018-02-01"],
        )
        .execute()
        .to_dataset(dim="bands")
    )

    test_x = (
        local_conn.load_stac(
            url=STAC_URLS["test_x"],
            spatial_extent=spatial_extent,
            temporal_extent=["2021-08-01", "2021-08-02"],
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

    print("End-to-end pipeline ran successfully and outputs are stored.")

    client.close()
    cluster.close()
