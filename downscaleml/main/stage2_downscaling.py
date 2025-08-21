import argparse
import xarray as xr
import numpy as np
from pathlib import Path
import logging
from lightgbm import LGBMRegressor, log_evaluation
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm
import sys


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Climate Downscaling Pipeline")

    parser.add_argument(
        "--target_var",
        type=str,
        required=True,
        help="Target variable name (t2m, ssrd, tp)",
    )
    parser.add_argument(
        "--x_path",
        type=str,
        required=True,
        help="Path to features Zarr (contains all years)",
    )
    parser.add_argument(
        "--y_path",
        type=str,
        required=True,
        help="Path to target variable Zarr (contains all years)",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=None,
        help="Optional path to additional external test features Zarr",
    )
    parser.add_argument(
        "--seas5_paths",
        type=str,
        nargs="*",
        default=[],
        help="Paths to SEAS5 forecast Zarrs (provide all 12 if needed)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for results"
    )

    return parser.parse_args()


def temporal_split(ds):
    """Split dataset into training (2000-2017) and testing (2018-2020) periods"""
    train = ds.sel(time=slice("2000", "2017"))
    test = ds.sel(time=slice("2018", "2020"))
    return train, test


def sort_features_by_name(ds):
    """Sort the variables in the dataset by their names"""
    sorted_vars = sorted(ds.data_vars)
    return ds[sorted_vars]


def setup_logging(output_dir: Path, target_var: str):
    """Configure logging to both file and console"""
    log_file = output_dir / f"{target_var}_downscaling.log"

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Add both handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main():
    args = parse_arguments()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir, args.target_var)

    try:
        logger.info(f"Starting pipeline for {args.target_var}")

        # 1. Load and prepare data
        logger.info("Loading datasets")
        X = (
            xr.open_zarr(args.x_path)
            .sel(lat=slice(46, 47), lon=slice(10, 12))
            .compute()
        )
        y = (
            xr.open_zarr(args.y_path)[args.target_var]
            .sel(lat=slice(46, 47), lon=slice(10, 12))
            .compute()
        )

        X = X[[args.target_var, "dem", "cos_doy", "sin_doy"]]

        X = X.rename({"lat": "y", "lon": "x"})
        y = y.rename({"lat": "y", "lon": "x"})

        # Sort features by variable names
        X = sort_features_by_name(X)

        # Always perform temporal split for training and temporal test
        logger.info("Splitting dataset into train/test periods")
        X_train, X_test_temporal = temporal_split(X)
        y_train, y_test_temporal = temporal_split(y)

        # Load external test data if provided
        if args.test_path:
            logger.info("Loading additional external test dataset")
            X_test_external = (
                xr.open_zarr(args.test_path)
                .sel(lat=slice(46, 47), lon=slice(10, 12))
                .compute()
            )
            X_test_external = X_test_external[
                [args.target_var, "dem", "cos_doy", "sin_doy"]
            ]
            X_test_external = X_test_external.rename({"lat": "y", "lon": "x"})
            X_test_external = sort_features_by_name(X_test_external)

        # 2. Train models for each pixel
        logger.info("Training pixel models")
        models = {}
        scalers = {}

        # Initialize progress bar for training
        total_pixels = len(X_train.y) * len(X_train.x)
        with tqdm(total=total_pixels, desc="Training models") as pbar:
            for i in range(len(X_train.y)):
                for j in range(len(X_train.x)):
                    X_pixel = X_train.isel(y=i, x=j).to_array().values.T
                    y_pixel = y_train.isel(y=i, x=j).values

                    valid = ~np.isnan(X_pixel).any(axis=1) & ~np.isnan(y_pixel)
                    X_pixel = X_pixel[valid]
                    y_pixel = y_pixel[valid]

                    if len(X_pixel) < 10:
                        pbar.update(1)
                        continue

                    scaler = StandardScaler().fit(X_pixel)
                    # Configure LGBM to be silent and use tqdm callback
                    model = LGBMRegressor(verbose=-1)
                    model.fit(
                        scaler.transform(X_pixel),
                        y_pixel,
                    )

                    models[(i, j)] = (model, scaler)
                    pbar.update(1)

        logger.info(f"Trained {len(models)} pixel models")

        # 4. Predict external test period if provided
        if args.test_path:
            logger.info("Predicting external test period")
            test_external_preds = np.full(
                (len(X_test_external.time), len(X_train.y), len(X_train.x)), np.nan
            )

            # Progress bar for external test prediction
            with tqdm(
                total=len(models), desc="Predicting external test period"
            ) as pbar:
                for (i, j), (model, scaler) in models.items():
                    X = X_test_external.isel(y=i, x=j).to_array().values.T
                    test_external_preds[:, i, j] = model.predict(scaler.transform(X))
                    pbar.update(1)

            # Save external test predictions
            test_external_ds = xr.Dataset(
                {args.target_var: (("time", "y", "x"), test_external_preds)},
                coords={"time": X_test_external.time, "y": X_train.y, "x": X_train.x},
            )
            test_external_out = (
                output_dir / f"{args.target_var}_test_external_predictions.zarr"
            )
            test_external_ds.to_zarr(test_external_out, mode="w")
            logger.info(f"Saved external test predictions to {test_external_out}")

        else:

            # 3. Predict temporal test period (2018-2020)
            logger.info("Predicting temporal test period")
            test_temporal_preds = np.full(
                (len(X_test_temporal.time), len(X_train.y), len(X_train.x)), np.nan
            )

            # Progress bar for test prediction
            with tqdm(
                total=len(models), desc="Predicting temporal test period"
            ) as pbar:
                for (i, j), (model, scaler) in models.items():
                    X = X_test_temporal.isel(y=i, x=j).to_array().values.T
                    test_temporal_preds[:, i, j] = model.predict(scaler.transform(X))
                    pbar.update(1)

            # Save temporal test predictions
            test_temporal_ds = xr.Dataset(
                {args.target_var: (("time", "y", "x"), test_temporal_preds)},
                coords={"time": X_test_temporal.time, "y": X_train.y, "x": X_train.x},
            )
            test_temporal_out = (
                output_dir / f"{args.target_var}_test_temporal_predictions.zarr"
            )
            test_temporal_ds.to_zarr(test_temporal_out, mode="w")
            logger.info(f"Saved temporal test predictions to {test_temporal_out}")

        # 5. Process SEAS5 forecasts only if provided
        if args.seas5_paths:
            logger.info(f"Processing {len(args.seas5_paths)} SEAS5 forecasts")
            for idx, seas5_path in enumerate(args.seas5_paths):
                seas5 = (
                    xr.open_zarr(seas5_path)
                    .sel(y=slice(46, 47), x=slice(10, 12))
                    .compute()
                )
                # Sort SEAS5 features by name to match training data
                seas5 = sort_features_by_name(seas5)
                seas5_preds = np.full(
                    (
                        len(seas5.time),
                        len(seas5.number),
                        len(X_train.y),
                        len(X_train.x),
                    ),
                    np.nan,
                )

                # Progress bar for SEAS5 prediction
                total_models = len(models) * len(seas5.number)
                with tqdm(total=total_models, desc=f"SEAS5 forecast {idx+1}") as pbar:
                    for m in range(len(seas5.number)):
                        for (i, j), (model, scaler) in models.items():
                            X = seas5.isel(y=i, x=j, number=m).to_array().values.T
                            seas5_preds[:, m, i, j] = model.predict(scaler.transform(X))
                            pbar.update(1)

                seas5_ds = xr.Dataset(
                    {args.target_var: (("time", "number", "y", "x"), seas5_preds)},
                    coords={
                        "time": seas5.time,
                        "number": seas5.number,
                        "y": X_train.y,
                        "x": X_train.x,
                    },
                )
                seas5_out = output_dir / f"{args.target_var}_seas5_{idx+1}.zarr"
                seas5_ds.to_zarr(seas5_out, mode="w")
                logger.info(f"Saved SEAS5 forecast {idx+1} to {seas5_out}")
        else:
            logger.info("No SEAS5 paths provided - skipping SEAS5 processing")

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
