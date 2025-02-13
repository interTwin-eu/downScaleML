# builtins
import sys
import os
import time
import logging
from datetime import timedelta, datetime
from logging.config import dictConfig
import pathlib
import datetime

# data processing and ML libraries
import numpy as np
import pandas as pd
import joblib
import xarray as xr
import dask
from dask.distributed import LocalCluster, Client
import optuna
from optuna.integration.dask import DaskStorage
from optuna.integration import LightGBMPruningCallback

# Scikit-learn and model-related imports
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, silhouette_score
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.cluster import KMeans
import lightgbm as lgb
import xgboost as xgb
from torch.cuda.amp import GradScaler, autocast


# Visualization
import matplotlib.pyplot as plt

# downscaleml imports
from downscaleml.core.hpo import objective
from downscaleml.core.dataset import ERA5Dataset, NetCDFDataset, EoDataset, DatasetStacker, SummaryStatsCalculator, doy_encoding
from downscaleml.core.clustering import ClusteringWorkflow
from downscaleml.main.config_reanalysis_dev_stage_2 import (
    ERA5_PATH, OBS_PATH, DEM_PATH, MODEL_PATH, TARGET_PATH,
    NET, ERA5_PLEVELS, ERA5_PREDICTORS, PREDICTAND,
    CALIB_PERIOD, VALID_PERIOD, DOY, NORM,
    OVERWRITE, DEM, DEM_FEATURES, STRATIFY, WET_DAY_THRESHOLD,
    VALID_SIZE, start_year, end_year, CHUNKS, TEST_PERIOD, RESULT_PERIOD, SEAS5_STAGE2_ADIGE, SEAS5_CHUNKS_forecast
)
from downscaleml.core.constants import (
    ERA5_P_VARIABLES, ERA5_P_VARIABLES_SHORTCUT, ERA5_P_VARIABLE_NAME,
    ERA5_S_VARIABLES, ERA5_S_VARIABLES_SHORTCUT, ERA5_S_VARIABLE_NAME,
    ERA5_VARIABLES, ERA5_VARIABLE_NAMES, ERA5_PRESSURE_LEVELS,
    PREDICTANDS
)
from downscaleml.core.utils import NAMING_Model, normalize, search_files, LogConfig
from downscaleml.core.logging import log_conf


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torchmetrics import MeanMetric
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import os
from mlflow import MlflowClient
import mlflow.pytorch
import subprocess
from sklearn.metrics import r2_score

LOGGER = logging.getLogger(__name__)

class MeteorologicalDataset(Dataset):
    def __init__(self, X, y, window_size=5):
        """
        Initialize the dataset for sliding window time sequences.

        Parameters:
        X (Tensor): Input tensor of shape (lat, lon, time, features).
        y (Tensor): Target tensor of shape (lat, lon, time, features).
        window_size (int): Size of the time window (e.g., 5 for t-2 to t+2).
        """
        self.X = X  # Shape: (lat, lon, time, features)
        self.y = y  # Shape: (lat, lon, time, features)
        self.window_size = window_size
        self.half_window = window_size // 2
        self.lat, self.lon, self.time, self.features = X.shape

    def __len__(self):
        # Limit the range to ensure full windows can be formed
        return self.time - 2 * self.half_window

    def __getitem__(self, idx):
        # Center the window around `idx + self.half_window`
        center_idx = idx + self.half_window
        start_idx = center_idx - self.half_window
        end_idx = center_idx + self.half_window + 1

        # Extract the time window for input
        input_window = self.X[:, :, start_idx:end_idx, :]  # Shape: (lat, lon, time_window, features)
        input_window = input_window.permute(2, 3, 0, 1)  # Shape: (time_window, features, lat, lon)

        # Extract the target at the center time step
        target_sample = self.y[:, :, center_idx, :]  # Shape: (lat, lon, features)
        target_sample = target_sample.permute(2, 0, 1)  # Shape: (features, lat, lon)

        return input_window, target_sample


# Example usage:
def prepare_data(X_tensor, y_tensor, window_size, batch_size):
    dataset = MeteorologicalDataset(X_tensor, y_tensor, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)
    return dataloader

def stacker(xarray_dataset):
    # stack along the lat and lon dimensions
    stacked = xarray_dataset.stack()
    dask_arr = stacked.to_array().data
    xarray_dataset = dask_arr.T
    LogConfig.init_log('Shape is in (spatial, time, variables):{}'.format(xarray_dataset.shape))
    return xarray_dataset

def normalize_tensor(data: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a 4D tensor of shape (lat, lon, time, features) for each feature independently.

    Args:
        data (torch.Tensor): Input tensor of shape (lat, lon, time, features).

    Returns:
        torch.Tensor: Normalized tensor with the same shape as input, where each feature has mean ~0 and std ~1.
    """
    # Reshape to isolate features for normalization: (features, lat*lon*time)
    data_reshaped = data.permute(3, 0, 1, 2).reshape(data.shape[3], -1)

    # Compute mean and std along the flattened dimensions (excluding feature dimension)
    mean = data_reshaped.mean(dim=1, keepdim=True)  # Shape: (features, 1)
    std = data_reshaped.std(dim=1, keepdim=True)    # Shape: (features, 1)

    # Normalize: (data - mean) / std
    data_normalized_reshaped = (data_reshaped - mean) / std

    # Reshape back to the original shape
    data_normalized = data_normalized_reshaped.reshape(data.shape[3], *data.shape[:3]).permute(1, 2, 3, 0)

    return data_normalized


import torch
import torch.nn as nn

# Define the ResidualBlock class
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)

# Define the Generator class
class Generator(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.residual_blocks = nn.ModuleList([ResidualBlock(64) for _ in range(16)])
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        residual = x
        for block in self.residual_blocks:
            x = block(x) + residual  # Apply each block, maintaining the residual
        x = self.conv_out(x)
        x = x.mean(dim=0, keepdim=True)  # Reduce over the batch dimension
        return x
    
class MeteorologicalDataset_prediction(Dataset):
    def __init__(self, X, window_size=5):
        """
        Initialize the dataset for sliding window time sequences.

        Parameters:
        X (Tensor): Input tensor of shape (lat, lon, time, features).
        window_size (int): Size of the time window (e.g., 5 for t-2 to t+2).
        """
        self.X = X  # Shape: (lat, lon, time, features)
        self.window_size = window_size
        self.half_window = window_size // 2
        self.lat, self.lon, self.time, self.features = X.shape

    def __len__(self):
        # Limit the range to ensure full windows can be formed
        return self.time - 2 * self.half_window

    def __getitem__(self, idx):
        # Center the window around `idx + self.half_window`
        center_idx = idx + self.half_window
        start_idx = center_idx - self.half_window
        end_idx = center_idx + self.half_window + 1

        # Extract the time window for input
        input_window = self.X[:, :, start_idx:end_idx, :]  # Shape: (lat, lon, time_window, features)
        input_window = input_window.permute(2, 3, 0, 1)  # Shape: (time_window, features, lat, lon)

        return input_window

def prepare_data_prediction(X_tensor, window_size):
    dataset = MeteorologicalDataset_prediction(X_tensor, window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader

# Extend the temporal dimension
def extend_time(ds, days_before, days_after):
    time = ds.coords['time']
    start_date = pd.to_datetime(time.values[0])
    end_date = pd.to_datetime(time.values[-1])
    
    # Create new time coordinates
    new_start_dates = pd.date_range(start=start_date - pd.Timedelta(days=days_before), end=start_date - pd.Timedelta(days=1))
    new_end_dates = pd.date_range(start=end_date + pd.Timedelta(days=1), end=end_date + pd.Timedelta(days=days_after))
    
    # Extend the dataset
    before_ds = ds.isel(time=0).expand_dims(time=new_start_dates)
    after_ds = ds.isel(time=-1).expand_dims(time=new_end_dates)
    
    extended_ds = xr.concat([before_ds, ds, after_ds], dim='time')
    return extended_ds

if __name__ == '__main__':

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
        ' - '.join([str(RESULT_PERIOD[0]), str(RESULT_PERIOD[-1])])))


    Seas5 = ERA5Dataset(SEAS5_STAGE2_ADIGE.joinpath("SEAS5_STAGE2_ADIGE"), ERA5_PREDICTORS,
                       plevels=ERA5_PLEVELS)
    print(Seas5)
    Seas5_ds = Seas5.merge(chunks=SEAS5_CHUNKS_forecast)
    Seas5_ds = xr.unify_chunks(Seas5_ds)
    Seas5_ds = Seas5_ds[0]
    Seas5_ds = Seas5_ds.rename({'longitude': 'x','latitude': 'y'})
    
    # initialize OBS predictand dataset
    LogConfig.init_log('Initializing observations for predictand: {}'
                       .format(PREDICTAND))

    # read in-situ gridded observations
    Obs_ds = search_files(OBS_PATH.joinpath(PREDICTAND), '.nc$').pop()
    Obs_ds = xr.open_dataset(Obs_ds)
    Obs_ds = Obs_ds.rename({'longitude': 'x','latitude': 'y'})

    Era5_ds = None

    # whether to use digital elevation model
    if DEM:

        dem = None
        # digital elevation model: Copernicus EU-Dem v1.1
        dem_path = search_files(DEM_PATH, '^interTwin_dem_chelsa_stage2_adige.nc$').pop()

        dem_seas5 = ERA5Dataset.dem_features(
            dem_path, {'y': Seas5_ds.y, 'x': Seas5_ds.x},
            add_coord={'time': Seas5_ds.time})
        
        #LogConfig.init_log('{}'.format(Era5_ds.chunks))

        # check whether to use slope and aspect
        if not DEM_FEATURES:
            #dem = dem.drop_vars(['slope', 'aspect']).chunk(Era5_ds.chunks)
            dem_seas5_n = dem_seas5.merge(dem_seas5.expand_dims(number=Seas5_ds['number'])).chunk(Seas5_ds.chunks)
            dem_seas5_n = dem_seas5_n.drop_vars(['slope', 'aspect']).chunk(Seas5_ds.chunks)

        # add dem to set of predictor variables
        #dem = dem.chunk(Era5_ds.chunks)
        dem_seas5_n = dem_seas5_n.chunk(Seas5_ds.chunks)
        #Era5_ds = xr.merge([Era5_ds, dem])
        Seas5_ds = xr.merge([Seas5_ds, dem_seas5_n])

    # initialize training data
    LogConfig.init_log('Initializing training data.')

    dem_seas5 = doy_encoding(dem_seas5, doy=DOY)

    Seas5_ds = Seas5_ds.merge(dem_seas5.expand_dims(number=Seas5_ds['number'])).chunk(Seas5_ds.chunks)
    Seas5_ds = Seas5_ds.drop_vars(['slope', 'aspect'])
    Seas5_ds = Seas5_ds.transpose('time', 'y', 'x', 'number')
    Seas5_ds = Seas5_ds.fillna(0)
    Seas5_ds = Seas5_ds.astype(np.float32)
    Seas5_ds = extend_time(Seas5_ds, days_before=2, days_after=2)


    
    LogConfig.init_log(f'Predictors : {Seas5_ds}')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    batch_size = 1  # Define batch size as per your requirement
    window_size = 5

    # Load the entire model directly (not just the state dictionary)
    generator = torch.load('/home/sdhinakaran/models_adige/best_generator_pr.pth', map_location=device)

    # Move the model to the GPU
    generator.to(device)
    generator.eval()
    #Loop should starts here!
        
    for i in range(Seas5_ds.sizes["number"]):
        
        LogConfig.init_log('{} Member'.format(i))
        predictors_test = stacker(Seas5_ds.isel(number=i)).compute()  
        LogConfig.init_log(f'predictors_test shape : {predictors_test.shape}')
        X_tensor_test = torch.from_numpy(predictors_test)
        X_tensor_test = normalize_tensor(X_tensor_test)

        dataloader_valid = prepare_data_prediction(X_tensor_test, window_size=window_size)
        
        # Generate predictions
        predicted_outputs = []
        with torch.no_grad():  # No gradients needed during inference
            # Iterate through the DataLoader
            for j, input_window in enumerate(dataloader_valid):
                # Remove the extra batch dimension
                input_window = input_window.squeeze(0).to(device)  # Shape: [5, 10, 161, 96]
                predicted_output = generator(input_window)
                predicted_outputs.append(predicted_output)
                pass
        
        # Stack all predictions into a single tensor
        predicted_outputs = torch.cat(predicted_outputs, dim=0)
        
        # Convert predictions to numpy for easier handling
        predicted_outputs_np = predicted_outputs.cpu().numpy()
        
        # Verify the output shape
        LogConfig.init_log(f"Predicted Outputs Shape: {predicted_outputs_np.shape}")  # Expected: (366, 1, 161, 96)
        
        prediction = predicted_outputs_np.squeeze().transpose(0, 2, 1)
        
        prediction.shape
        LogConfig.init_log(f"Predicted Outputs Shape: {prediction.shape}")  # Expected: (366, 1, 161, 96)
        
        # store predictions in xarray.Dataset
        predictions = xr.DataArray(data=prediction, dims=['time', 'y', 'x'], coords=dict(time=pd.date_range(Seas5_ds.time.values[2],Seas5_ds.time.values[-3], freq='D'), lat=Seas5_ds.y, lon=Seas5_ds.x))

        predictions = predictions
        predictions = predictions.to_dataset(name=PREDICTAND)
        
        predictions = predictions.set_index(
            time='time',
            y='lat',
            x='lon'
        )
        
        predictions.to_netcdf("/mnt/CEPH_PROJECTS/InterTwin/Climate_Downscaling/two_stage/RESULTS_STAGE2_ADIGE/{}/SEAS5_ADIGE_PR_{}_CHELSA_2020_I_1KM.nc".format(PREDICTAND, str(i)))
        
    LogConfig.init_log('Prediction Saved!!! SMILE PLEASE')












