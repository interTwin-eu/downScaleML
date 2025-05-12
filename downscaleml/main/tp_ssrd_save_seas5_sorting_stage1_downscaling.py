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
import os
from pathlib import Path
from datetime import datetime
import gc
import joblib

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Climate Downscaling Pipeline')
    
    parser.add_argument('--target_var', type=str, required=True,
                       help='Target variable name (t2m, ssrd, tp)')
    parser.add_argument('--x_path', type=str, required=True,
                       help='Path to features Zarr (contains all years)')
    parser.add_argument('--y_path', type=str, required=True,
                       help='Path to target variable Zarr (contains all years)')
    parser.add_argument('--seas5_paths', type=str, nargs='*', default=[],
                       help='Paths to SEAS5 forecast Zarrs (either a directory or specific .zarr files)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--use-pretrained', action='store_true',
                       help='Use pretrained models if available (skip training)')
    
    return parser.parse_args()

def get_seas5_zarr_paths(seas5_paths):
    """Process SEAS5 paths - if directory given, list all .zarr files sorted by month_year, otherwise use provided files"""
    if not seas5_paths:
        return []
    
    # If only one path is given and it's a directory
    if len(seas5_paths) == 1 and os.path.isdir(seas5_paths[0]):
        directory = Path(seas5_paths[0])
        zarr_files = list(directory.glob('*.zarr'))
        if not zarr_files:
            raise ValueError(f"No .zarr files found in directory: {directory}")
        
        # Sort files by month and year
        def get_month_year_key(file_path):
            # Extract the month_year part from the filename
            month_year_str = file_path.stem.split('_')[-2:]  # Gets last two parts
            month_year_str = ' '.join(month_year_str)  # Handle both space and underscore
            
            # Clean up any remaining underscores and standardize format
            month_year_str = month_year_str.replace('_', ' ')
            
            # Parse to datetime object for proper sorting
            try:
                return datetime.strptime(month_year_str, '%B %Y')
            except ValueError:
                # Try alternative formats if needed
                return datetime.strptime(month_year_str.lower().capitalize(), '%B %Y')
        
        zarr_files_sorted = sorted(zarr_files, key=get_month_year_key)
        return [str(f) for f in zarr_files_sorted]
    
    # Otherwise use the provided paths as-is
    return seas5_paths

def feature_engineering_ssrd(X):
    """Feature engineering for ssrd variable"""
    X = X.drop_vars(["tp", "q_850", "u_850", "v_850", "z_850"])
    X['ssrd_lag1'] = X['ssrd'].shift(time=1)  # Previous day
    X['ssrd_lag2'] = X['ssrd'].shift(time=2)  # Day before yesterday
    
    # Moving averages
    X['ssrd_ma3'] = X['ssrd'].rolling(time=3, min_periods=1, center=False).mean()
    X['ssrd_ma7'] = X['ssrd'].rolling(time=7, min_periods=1, center=False).mean()

    # Interaction terms
    X['t2m_t850_interaction'] = X['t2m'] * X['t_850']
    X['t2m_dem_ratio'] = X['t2m'] / (X['dem'] + 1e-10)

    # Drop unused variables
    X = X.drop_vars(["dem", "t2m", "t_850"])
    return X

def feature_engineering_t2m(X):
    """Feature engineering for t2m variable"""
    
    return X

def feature_engineering_tp(X):
    """Feature engineering for tp variable"""
    # Keep relevant variables
    X = X.drop_vars(["ssrd", "t2m", "q_850", "u_850", "v_850", "z_850"])
    
    # Lag features
    X['tp_lag1'] = X['tp'].shift(time=1)
    X['tp_lag2'] = X['tp'].shift(time=2)
    
    # Moving averages
    X['tp_ma3'] = X['tp'].rolling(time=3, min_periods=1, center=False).mean()
    X['tp_ma7'] = X['tp'].rolling(time=7, min_periods=1, center=False).mean()
    
    # Interaction with topography
    X['tp_dem_ratio'] = X['tp'] / (X['dem'] + 1e-10)
    
    # Drop unused variables
    X = X.drop_vars(["dem"])
    return X

def apply_feature_engineering(X, target_var):
    """Apply appropriate feature engineering based on target variable"""
    if target_var == 'ssrd':
        return feature_engineering_ssrd(X)
    elif target_var == 't2m':
        return feature_engineering_t2m(X)
    elif target_var == 'tp':
        return feature_engineering_tp(X)
    else:
        raise ValueError(f"Unknown target variable: {target_var}")

def temporal_split(ds):
    """Split dataset into training (2000-2017) and testing (2018-2020) periods"""
    train = ds.sel(time=slice('2000', '2017'))
    test = ds.sel(time=slice('2018', '2020'))
    return train, test

def sort_features_by_name(ds):
    """Sort the variables in the dataset by their names"""
    sorted_vars = sorted(ds.data_vars)
    return ds[sorted_vars]

def setup_logging(output_dir: Path, target_var: str):
    """Configure logging to both file and console"""
    log_file = output_dir / f'{target_var}_downscaling.log'
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
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
        logger.info(f'Starting pipeline for {args.target_var}')
        
        # Process SEAS5 paths
        seas5_paths = get_seas5_zarr_paths(args.seas5_paths)
        
        # 1. Load and split data
        logger.info('Loading and splitting datasets')
        X = xr.open_zarr(args.x_path).sel(lat=slice(46, 46.5), lon=slice(10, 10.5)).compute()
        X = apply_feature_engineering(X, args.target_var)
        
        y = xr.open_zarr(args.y_path)[args.target_var].sel(lat=slice(46, 46.5), lon=slice(10, 10.5)).compute()

        X = X.rename({"lat": "y", "lon": "x"})
        y = y.rename({"lat": "y", "lon": "x"})

        X = X.fillna(0)
        y = y.fillna(0)

        X = sort_features_by_name(X)

        X_train, X_test = temporal_split(X)
        y_train, _ = temporal_split(y)

        y_coords = X_train.y
        x_coords = X_train.x        

        # Check for pretrained models
        model_file = output_dir / f'{args.target_var}_models_scalers_v3.joblib'
        
        if args.use_pretrained and model_file.exists():
            # Load pretrained models
            logger.info(f'Loading pretrained models from {model_file}')
            model_store = joblib.load(model_file)
            models = model_store['models']
            y_coords = model_store['y_coords']
            x_coords = model_store['x_coords']
            logger.info(f'Loaded {len(models)} pretrained models')
        else:
            if args.use_pretrained:
                logger.warning(f'Pretrained model file not found at {model_file}, training new models')
            
            # 2. Train models for each pixel
            logger.info('Training pixel models')
            models = {}
            
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
                        model = LGBMRegressor(verbose=-1)
                        model.fit(
                            scaler.transform(X_pixel), 
                            y_pixel,
                        )
                        
                        models[(i,j)] = (model, scaler)
                        pbar.update(1)
                    
            logger.info(f'Trained {len(models)} pixel models')
            
            # Save all models and scalers
            model_store = {
                'models': models,
                'y_coords': y_coords,
                'x_coords': x_coords
            }
            joblib.dump(model_store, model_file)
            logger.info(f'Saved models and scalers to {model_file}')


        # 3. Predict test period (2018-2020)
        logger.info('Predicting test period')
        test_preds = np.full((len(X_test.time), len(y_coords), len(x_coords)), np.nan)
        
        # Progress bar for test prediction
        with tqdm(total=len(models), desc="Predicting test period") as pbar:
            for (i,j), (model, scaler) in models.items():
                X = X_test.isel(y=i, x=j).to_array().values.T
                test_preds[:, i, j] = model.predict(scaler.transform(X))
                pbar.update(1)
        
        # Save test predictions
        test_ds = xr.Dataset(
            {args.target_var: (('time', 'y', 'x'), test_preds)},
            coords={'time': X_test.time, 'y': y_coords, 'x': x_coords}
        )
        test_out = output_dir / f'{args.target_var}_test_predictions_v3.zarr'
        test_ds.to_zarr(test_out, mode='w')
        logger.info(f'Saved test predictions to {test_out}')

        del X, y, X_train, X_test, y_train
        gc.collect()
        
        # 4. Process SEAS5 forecasts only if provided
        # 4. Process SEAS5 forecasts only if provided
        if seas5_paths:
            logger.info(f'Processing {len(seas5_paths)} SEAS5 forecasts')
            for idx, seas5_path in enumerate(seas5_paths):
                # Extract month_year from the filename
                month_year = Path(seas5_path).stem.split('_')[-2:]  # Gets last two parts
                month_year = '_'.join(month_year).replace('_', ' ')  # Convert to space-separated
                month_year = month_year.replace('_', ' ')  # Ensure space separator
                
                seas5 = xr.open_zarr(seas5_path).sel(y=slice(46, 46.5), x=slice(10, 10.5)).compute()
                seas5 = apply_feature_engineering(seas5, args.target_var)

                seas5 = seas5.drop_vars(["tp", "q_850", "u_850", "v_850", "z_850"])
                seas5['ssrd_lag1'] = seas5['ssrd'].shift(time=1)  # Previous day
                seas5['ssrd_lag2'] = seas5['ssrd'].shift(time=2)  # Day before yesterday
                
                # 3-day moving average (centered=False to use past values only)
                seas5['ssrd_ma3'] = seas5['ssrd'].rolling(time=3, min_periods=1, center=False).mean()
                
                # 7-day moving average
                seas5['ssrd_ma7'] = seas5['ssrd'].rolling(time=7, min_periods=1, center=False).mean()

                # After loading SEAS5 data
                seas5 = seas5.fillna(0)
                seas5 = sort_features_by_name(seas5)
                
                # Get the number of y and x points in the SEAS5 data
                num_y = len(seas5.y)
                num_x = len(seas5.x)
                
                seas5_preds = np.full(
                    (len(seas5.time), len(seas5.number), num_y, num_x),
                    np.nan
                )
                
                # Progress bar for SEAS5 prediction
                total_models = len(models) * len(seas5.number)
                with tqdm(total=total_models, desc=f"SEAS5 forecast {month_year}") as pbar:
                    for m in range(len(seas5.number)):
                        for (i,j), (model, scaler) in models.items():
                            if i >= num_y or j >= num_x:
                                continue  # Skip if model coordinates are out of bounds
                            X = seas5.isel(y=i, x=j, number=m).to_array().values.T
                            seas5_preds[:, m, i, j] = model.predict(scaler.transform(X))
                            pbar.update(1)
                
                seas5_ds = xr.Dataset(
                    {args.target_var: (('time', 'number', 'y', 'x'), seas5_preds)},
                    coords={
                        'time': seas5.time,
                        'number': seas5.number,
                        'y': seas5.y,  # Use SEAS5's y coordinates
                        'x': seas5.x   # Use SEAS5's x coordinates
                    }
                )
                seas5_out = output_dir / f'{args.target_var}_seas5_{month_year}.zarr'
                seas5_ds.to_zarr(seas5_out, mode='w')
                logger.info(f'Saved SEAS5 forecast {month_year} to {seas5_out}')
                del seas5_ds, seas5_preds, seas5 
                gc.collect()
        else:
            logger.info('No SEAS5 paths provided - skipping SEAS5 processing')
        
        logger.info('Pipeline completed successfully')
        
    except Exception as e:
        logger.error(f'Pipeline failed: {str(e)}')
        raise

if __name__ == '__main__':
    main()