# builtins
import sys
import os
import time
import logging
from datetime import timedelta
from logging.config import dictConfig
import numpy as np
import datetime
import pathlib
import pandas as pd
import joblib
from dask.distributed import Client
import optuna
import distributed
from optuna.integration.dask import DaskStorage
from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import joblib

# externals|
import xarray as xr

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
import lightgbm as lgb
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from downscaleml.core.hpo import objective

# locals
from downscaleml.core.dataset import ERA5Dataset, NetCDFDataset, EoDataset, DatasetStacker, SummaryStatsCalculator, doy_encoding

from downscaleml.core.clustering import ClusteringWorkflow

from downscaleml.main.config_reanalysis_dev import (ERA5_PATH, OBS_PATH, DEM_PATH, MODEL_PATH, TARGET_PATH, 
                                     NET, ERA5_PLEVELS, ERA5_PREDICTORS, PREDICTAND,
                                     CALIB_PERIOD, VALID_PERIOD, DOY, NORM,
                                     OVERWRITE, DEM, DEM_FEATURES, STRATIFY, WET_DAY_THRESHOLD,
                                     VALID_SIZE, start_year, end_year, CHUNKS)

#from downscaleml.main.inputoutput import ()

from downscaleml.core.constants import (ERA5_P_VARIABLES, ERA5_P_VARIABLES_SHORTCUT, ERA5_P_VARIABLE_NAME,
                                        ERA5_S_VARIABLES, ERA5_S_VARIABLES_SHORTCUT, ERA5_S_VARIABLE_NAME,
                                        ERA5_VARIABLES, ERA5_VARIABLE_NAMES, ERA5_PRESSURE_LEVELS,
                                        PREDICTANDS, ERA5_P_VARIABLES, ERA5_S_VARIABLES)

from downscaleml.core.utils import NAMING_Model, normalize, search_files, LogConfig
from downscaleml.core.logging import log_conf
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# module level logger
LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':

    client = Client()

    # initialize timing
    start_time = time.monotonic()
        
    # initialize network filename
    state_file = NAMING_Model.state_file(
        NET, PREDICTAND, ERA5_PREDICTORS, ERA5_PLEVELS, WET_DAY_THRESHOLD, dem=DEM,
        dem_features=DEM_FEATURES, doy=DOY, stratify=STRATIFY)
    
    state_file = MODEL_PATH.joinpath(PREDICTAND, state_file)
    target = TARGET_PATH.joinpath(PREDICTAND)

    # check if output path exists
    if not target.exists():
        target.mkdir(parents=True, exist_ok=True)
    # initialize logging
    log_file = state_file.with_name(state_file.name + "_log.txt")
    
    if log_file.exists():
        log_file.unlink()
    dictConfig(log_conf(log_file))

    # check if target dataset already exists
    target = target.joinpath(state_file.name + '.nc')
    if target.exists() and not OVERWRITE:
        LogConfig.init_log('{} already exists.'.format(target))
        sys.exit()

    LogConfig.init_log('Initializing downscaling for period: {}'.format(
        ' - '.join([str(CALIB_PERIOD[0]), str(CALIB_PERIOD[-1])])))

    # initialize ERA5 predictor dataset
    LogConfig.init_log('Initializing ERA5 predictors.')
    Era5 = ERA5Dataset(ERA5_PATH.joinpath('ERA5'), ERA5_PREDICTORS,
                       plevels=ERA5_PLEVELS)
    Era5_ds = Era5.merge(chunks=CHUNKS)
    Era5_ds = Era5_ds.rename({'lon': 'x','lat': 'y'})
    
    # initialize OBS predictand dataset
    LogConfig.init_log('Initializing observations for predictand: {}'
                       .format(PREDICTAND))

    # read in-situ gridded observations
    Obs_ds = search_files(OBS_PATH.joinpath(PREDICTAND), '.nc$').pop()
    Obs_ds = xr.open_dataset(Obs_ds)
    Obs_ds = Obs_ds.rename({'lon': 'x','lat': 'y'})

    # whether to use digital elevation model
    if DEM:
        # digital elevation model: Copernicus EU-Dem v1.1
        dem = search_files(DEM_PATH, '^interTwin_dem.nc$').pop()

        # read elevation and compute slope and aspect
        dem = ERA5Dataset.dem_features(
            dem, {'y': Era5_ds.y, 'x': Era5_ds.x},
            add_coord={'time': Era5_ds.time})

        # check whether to use slope and aspect
        if not DEM_FEATURES:
            dem = dem.drop_vars(['slope', 'aspect']).chunk(Era5_ds.chunks)

        # add dem to set of predictor variables
        dem = dem.chunk(Era5_ds.chunks)
        Era5_ds = xr.merge([Era5_ds, dem])


    # initialize training data
    LogConfig.init_log('Initializing training data.')

    # split calibration period into training and validation period
    if PREDICTAND == 'pr' and STRATIFY:
        # stratify training and validation dataset by number of
        # observed wet days for precipitation
        wet_days = (Obs_ds.sel(time=CALIB_PERIOD).mean(dim=('y', 'x'))
                    >= WET_DAY_THRESHOLD).to_array().values.squeeze()
        train, valid = train_test_split(
            CALIB_PERIOD, stratify=wet_days, test_size=VALID_SIZE)

        # sort chronologically
        train, valid = sorted(train), sorted(valid)
        Era5_train, Obs_train = Era5_ds.sel(time=train), Obs_ds.sel(time=train)
        Era5_valid, Obs_valid = Era5_ds.sel(time=valid), Obs_ds.sel(time=valid)
    else:
        LogConfig.init_log('We are not calculating Stratified Precipitation based on Wet Days here!')

    # training and validation dataset
    Era5_train, Obs_train = Era5_ds.sel(time=CALIB_PERIOD), Obs_ds.sel(time=CALIB_PERIOD)
    Era5_valid, Obs_valid = Era5_ds.sel(time=VALID_PERIOD), Obs_ds.sel(time=VALID_PERIOD)

    # whether to use digital elevation model
    if DEM:
        # digital elevation model: Copernicus EU-Dem v1.1
        dem = search_files(DEM_PATH, '^interTwin_dem.nc$').pop()

        # read elevation and compute slope and aspect
        dem = ERA5Dataset.dem_features(
            dem, {'y': Obs_train.y, 'x': Obs_train.x},
            add_coord={'time': Obs_train.time})

        # check whether to use slope and aspect
        if not DEM_FEATURES:
            dem = dem.drop_vars(['slope', 'aspect']).chunk(Obs_train.chunks)

        # add dem to set of predictor variables
        dem = dem.chunk(Obs_train.chunks)
        Obs_train = xr.merge([Obs_train, dem])

    #statistics_dataset = calculate_statistics_to_dataset(Obs_train)
    LogConfig.init_log(f'To see : {Obs_train}')

    # Create an instance of the class
    stats_calculator = SummaryStatsCalculator(Obs_train)
    
    # Compute summary statistics
    statistics_dataset = stats_calculator.compute_summary_stats()

    LogConfig.init_log(f'Summary statistics loaded along the time dimension : {statistics_dataset}')

    lat_lon_flat = statistics_dataset.to_array().stack(points=('y', 'x')).T.values

    # Instantiate the workflow and run the full clustering process
    clustering = ClusteringWorkflow(data=lat_lon_flat, statistics_dataset=statistics_dataset)
    clustered_map = clustering.full_clustering_workflow()
    
    # Log the final clustered map
    LOGGER.info(f'Clustered map dataset here: {clustered_map}')


    Era5_train = doy_encoding(Era5_train, Obs_train, doy=DOY)
    Era5_valid = doy_encoding(Era5_valid, Obs_valid, doy=DOY)
    
    # Assuming Era5_train, Era5_valid, Obs_train, and Obs_valid are already defined xarray datasets
    stacker = DatasetStacker(Era5_train)
    predictors_train = stacker.stack_spatial_dimensions(compute=True)
    
    stacker = DatasetStacker(Era5_valid)
    predictors_valid = stacker.stack_spatial_dimensions(compute=True)
    
    stacker = DatasetStacker(Obs_train.drop_vars("elevation"))
    predictand_train = stacker.stack_spatial_dimensions()  # Computed later if needed
    
    stacker = DatasetStacker(Obs_valid)
    predictand_valid = stacker.stack_spatial_dimensions()  # Computed later if needed

    cluster_py = clustered_map.stack(spatial=('y', 'x')).to_numpy()
    cluster_dict = {}
    
    # Get unique values from the data array
    unique_values = np.unique(cluster_py)  # Use `.values` to get the raw NumPy array
    
    # Loop through each unique value and capture the indices
    for value in unique_values:
        # Get the indices where the value occurs in the DataArray using NumPy
        indices = np.argwhere(cluster_py == value)
        
        # Convert to list of tuples (x, y) and store in the dictionary
        cluster_dict[value] = indices.squeeze()
    
        LogConfig.init_log(f'Clusters and their shapes: {cluster_dict[value].shape}')


    # Processing clusters and optimizing the model for each cluster
    lgb_models = {}
    
    for cluster_id, indices in cluster_dict.items():
        LogConfig.init_log(f"Processing cluster {cluster_id} with {len(indices)} indices")

        # Gather all data for the current cluster
        cluster_data = predictors_train[indices, :, :]  # shape: (cluster_size, 1826, 15)
        cluster_target = predictand_train[indices, :, 0]
    
        cluster_data_test = predictors_valid[indices, :, :]
        cluster_target_test = predictand_valid[indices, :, 0]
        
        # Reshape the cluster data for Optuna optimization
        X_cluster = cluster_data.reshape(-1, 15)  # shape: (cluster_size * 1826, 15)
        y_cluster = cluster_target.flatten()
    
        # Reshape the cluster data for Optuna optimization
        test_X_cluster = cluster_data_test.reshape(-1, 15)  # shape: (cluster_size * 1826, 15)
        test_y_cluster = cluster_target_test.flatten()
    
        # Run Optuna optimization for the entire cluster
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        study.optimize(lambda trial: objective(trial, X_cluster, y_cluster, test_X_cluster, test_y_cluster), n_trials=50, n_jobs=-1)
        
        best_params = study.best_trial.params
        best_params.update({
        "verbose": -1,                 # Suppress output
        "early_stopping_rounds": 100,
        })

        LogConfig.init_log(f"Best parameters for cluster {cluster_id}: {best_params}")
    
        # Split the data into training and validation sets
        train_x, valid_x, train_y, valid_y = train_test_split(X_cluster, y_cluster, test_size=0.2, random_state=42)
        # Extract the best parameters from the Optuna study
    
        # Train the model on the entire dataset for the current cluster
        dtrain = lgb.Dataset(train_x, label=train_y)
        dvalid = lgb.Dataset(valid_x, label=valid_y, reference=dtrain)
    
        # Train the model with the best parameters
        model = lgb.train(
            best_params,
            train_set=dtrain,
            valid_sets=[dtrain, dvalid],
            valid_names=['train', 'valid']
        )
    
        # Predict on the test set
        preds = model.predict(test_X_cluster)
    
        # Compute RMSE for the test set
        rmse = mean_squared_error(test_y_cluster, preds, squared=False)
        LogConfig.init_log(f"RMSE for cluster {cluster_id}: {rmse}")
    
        # Store the trained model for the current cluster
        lgb_models[cluster_id] = model
        LogConfig.init_log(f"Model for cluster {cluster_id} trained and stored.")

    joblib.dump(lgb_models, 'trained_lgb_models_gpu_complete_nT50_test.pkl')
    LogConfig.init_log("Models Trained, Optimised and Saved Successfully.")

    prediction = np.ones(shape=(predictand_valid.shape[1], predictand_valid.shape[0], 1)) * np.nan

    # Iterate through clusters and process in batches
    for cluster_id, indices in cluster_dict.items():
        LogConfig.init_log(f"Processing cluster {cluster_id} with {len(indices)} indices")
            
        # Extract the data for the current batch from the numpy array
        X_unseen = predictors_valid[indices, :, :]  # shape: (batch_size, 1826, 15)
    
        truth = predictand_valid[indices, :, 0]
    
        LogConfig.init_log(X_unseen.shape, truth.shape)
        
        X = X_unseen.reshape(-1, 15)  # shape: (batch_size * 1826, 15)
        y = truth.flatten()
    
        LogConfig.init_log(X.shape, y.shape)
    
        # Make predictions using the trained model for the current cluster
        if cluster_id in lgb_models:
            preds = lgb_models[cluster_id].predict(X)
            LogConfig.init_log('Cluster ID: {:d} - score: R2: {:.2f}, MSE: {:.2f}, MAE: {:.2f}'.format(cluster_id, r2_score(y, preds), mean_squared_error(y, preds), mean_absolute_error(y, preds)))
    
             # Reshape the predictions and ground truth back to (batch_size, 1826)
            preds_unflattened = preds.reshape(len(indices), truth.shape[1])
    
            prediction[:, indices, 0] = preds_unflattened.T
            
        else:
            LogConfig.init_log(f"No trained model found for cluster {cluster_id}")
    
        LogConfig.init_log(f"Model for cluster {cluster_id} trained and stored.")
        
    prediction = prediction.reshape(Obs_valid.time.shape[0],Obs_valid.y.shape[0], Obs_valid.x.shape[0], 1).squeeze()
    
    # store predictions in xarray.Dataset
    predictions = xr.DataArray(data=prediction, dims=['time', 'y', 'x'], coords=dict(time=pd.date_range(Era5_valid.time.values[0],Era5_valid.time.values[-1], freq='D'), lat=Obs_valid.y, lon=Obs_valid.x))
    predictions = predictions.to_dataset(name=PREDICTAND)
    
    predictions = predictions.set_index(
        time='time',
        y='lat',
        x='lon'
    )
    
    predictions.to_netcdf("/home/sdhinakaran/eurac/downScaleML-refactoring/downscaleml/main/CLUSTERED_50T_tasmean_2016_downscaled_LGBM_GPU.nc")
    
    LogConfig.init_log('Prediction Saved!!! SMILE PLEASE')
    
    
        
    
    
        






