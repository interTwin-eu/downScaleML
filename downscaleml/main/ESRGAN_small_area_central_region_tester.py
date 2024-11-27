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
from dask.distributed import Client
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
from downscaleml.main.config_reanalysis_dev import (
    ERA5_PATH, OBS_PATH, DEM_PATH, MODEL_PATH, TARGET_PATH,
    NET, ERA5_PLEVELS, ERA5_PREDICTORS, PREDICTAND,
    CALIB_PERIOD, VALID_PERIOD, DOY, NORM,
    OVERWRITE, DEM, DEM_FEATURES, STRATIFY, WET_DAY_THRESHOLD,
    VALID_SIZE, start_year, end_year, CHUNKS, TEST_PERIOD
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

# module level logger
LOGGER = logging.getLogger(__name__)
client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
mlflow.set_tracking_uri('http://127.0.0.1:5000')
mlflow.set_experiment("ESRGAN_small_regio_t_seq_stepLR_forD_early_stopping_on_val_increased_rrdb")
base_dir = "/home/sdhinakaran/downScaleML/downscaleml/main/mlartifacts"



class Generator(nn.Module):
    def __init__(self, in_channels=10, out_channels=1):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        
        # Manually create residual blocks with unique names
        self.residual_blocks = nn.ModuleList([ResidualBlock(64) for _ in range(23)])
        
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        residual = x
        for block in self.residual_blocks:
            x = block(x) + residual  # Apply each block, maintaining the residual
        x = self.conv_out(x)
        x = x.mean(dim=0, keepdim=True)  # Reduce over the batch dimension
        return x


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


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=1, num_filters=64):
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)


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

def train_with_mlflow_train_test(dataloader_train, dataloader_test, generator, discriminator, optimizer_G, optimizer_D, scheduler_G, scheduler_D,
                      num_epochs=50, patience=5, min_delta=0.001, device='cuda'):
    generator.train()
    discriminator.train()

    # Metrics for MLflow
    g_loss_metric = MeanMetric().to(device)
    d_loss_metric = MeanMetric().to(device)

    best_g_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        g_loss_metric.reset()
        d_loss_metric.reset()

        for input_batch, target_batch in dataloader_train:
            input_batch, target_batch = input_batch.squeeze(0).to(device), target_batch.to(device)

            # Train Discriminator
            optimizer_D.zero_grad()
            real_output = discriminator(target_batch)
            real_loss = F.binary_cross_entropy_with_logits(real_output, torch.ones_like(real_output))

            fake_images = generator(input_batch)
            fake_output = discriminator(fake_images.detach())
            fake_loss = F.binary_cross_entropy_with_logits(fake_output, torch.zeros_like(fake_output))

            d_loss = real_loss + fake_loss
            d_loss.backward()
            nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_D.step()
            d_loss_metric.update(d_loss)

            # Train Generator
            optimizer_G.zero_grad()
            fake_output = discriminator(fake_images)
            g_loss_adv = F.binary_cross_entropy_with_logits(fake_output, torch.ones_like(fake_output))
            g_loss_content = F.mse_loss(fake_images, target_batch)
            g_loss = g_loss_adv + g_loss_content
            g_loss.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_G.step()
            g_loss_metric.update(g_loss)

        scheduler_D.step()

        avg_g_loss = g_loss_metric.compute().item()
        avg_d_loss = d_loss_metric.compute().item()

        # Log metrics to MLflow
        mlflow.log_metric("generator_loss", avg_g_loss, step=epoch)
        mlflow.log_metric("discriminator_loss", avg_d_loss, step=epoch)

        lr_G = optimizer_G.param_groups[0]['lr']
        lr_D = optimizer_D.param_groups[0]['lr']
        mlflow.log_metric("lr_generator", lr_G, step=epoch)
        mlflow.log_metric("lr_discriminator", lr_D, step=epoch)

        # Validation Loop
        generator.eval()
        with torch.no_grad():
            val_r2, val_mae, val_mse = 0.0, 0.0, 0.0
            val_samples = 0

            for val_input, val_target in dataloader_test:
                val_input, val_target = val_input.squeeze(0).to(device), val_target.to(device)
                val_pred = generator(val_input)

                # Compute validation metrics
                val_r2 += r2_score(val_target.cpu().numpy().flatten(), val_pred.cpu().numpy().flatten()) * len(val_target)
                val_mae += F.l1_loss(val_pred, val_target).item() * len(val_target)
                val_mse += F.mse_loss(val_pred, val_target).item() * len(val_target)
                val_samples += len(val_target)

            val_r2 /= val_samples
            val_mae /= val_samples
            val_mse /= val_samples

        scheduler_G.step(val_mse)

        mlflow.log_metric("val_r2", val_r2, step=epoch)
        mlflow.log_metric("val_mae", val_mae, step=epoch)
        mlflow.log_metric("val_mse", val_mse, step=epoch)
        
        generator.train()

        LogConfig.init_log(f"Epoch {epoch+1}/{num_epochs}, Avg G Loss: {avg_g_loss:.4f}, Avg D Loss: {avg_d_loss:.4f}")

        if val_mse < best_g_loss - min_delta:
            best_g_loss = val_mse
            epochs_without_improvement = 0

            torch.save(generator, "best_generator.pth")
            torch.save(discriminator, "best_discriminator.pth")
            
            mlflow.log_artifact("best_generator.pth")
            mlflow.log_artifact("best_discriminator.pth")
            
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")

                run_id = mlflow.active_run().info.run_id
                artifact_uri = mlflow.get_artifact_uri(run_id)
                # Remove the 'mlflow-artifacts:' prefix
                artifact_path = artifact_uri.replace("mlflow-artifacts:", "")
            
                    
                # Construct the full path to the artifact directory by removing the last part after the last '/'
                artifact_dir = os.path.dirname(artifact_path.lstrip("/"))
                
                # Join with the base directory
                full_artifact_dir = os.path.join(base_dir, artifact_dir)
                # Load the models

                # Example: Load the discriminator model from this path
                discriminator_model_path = os.path.join(full_artifact_dir, "best_discriminator.pth")
                generator_model_path = os.path.join(full_artifact_dir, "best_generator.pth")
                
                generator = torch.load(generator_model_path, map_location=device)
                discriminator = torch.load(discriminator_model_path, map_location=device)
                break

    # End of training
    if epochs_without_improvement < patience:  # If early stopping was NOT triggered
        print("Training completed all epochs.")
    else:
        print("Training ended early due to early stopping.")
        
    run_id = mlflow.active_run().info.run_id
    artifact_uri = mlflow.get_artifact_uri(run_id)
    # Remove the 'mlflow-artifacts:' prefix
    artifact_path = artifact_uri.replace("mlflow-artifacts:", "")

        
    # Construct the full path to the artifact directory by removing the last part after the last '/'
    artifact_dir = os.path.dirname(artifact_path.lstrip("/"))
    
    # Join with the base directory
    full_artifact_dir = os.path.join(base_dir, artifact_dir)
    # Load the models

    # Example: Load the discriminator model from this path
    discriminator_model_path = os.path.join(full_artifact_dir, "best_discriminator.pth")
    generator_model_path = os.path.join(full_artifact_dir, "best_generator.pth")

    print(f'Generator Path: {generator_model_path}')
    print(f'Discriminator Path: {discriminator_model_path}')

    
    generator = torch.load(generator_model_path, map_location=device)
    discriminator = torch.load(discriminator_model_path, map_location=device)
    
    return generator, discriminator



def stacker(xarray_dataset):
    # stack along the lat and lon dimensions
    stacked = xarray_dataset.stack()
    dask_arr = stacked.to_array().data
    xarray_dataset = dask_arr.T
    LogConfig.init_log('Shape is in (spatial, time, variables):{}'.format(xarray_dataset.shape))
    return xarray_dataset

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

def get_full_year_dates_from_dataset(dataset, time_var='time'):
    """
    Extract the full years from the time range in an xarray dataset.

    Parameters:
    dataset (xarray.Dataset): The dataset containing the time variable.
    time_var (str): The name of the time variable in the dataset.

    Returns:
    pd.DatetimeIndex: A date range covering the full years in the dataset's time range.
    """
    # Extract time values and convert to pandas datetime
    time_values = pd.to_datetime(dataset[time_var].values)
    
    # Determine the start and end of the time range
    start_date = time_values[0]
    end_date = time_values[-1]
    
    # Compute the first and last full years within the range
    first_full_year = datetime.date(start_date.year, 1, 1)
    last_full_year = datetime.date(end_date.year, 12, 31)

    # Adjust if the start or end date does not cover the entire year
    if start_date.month > 1 or (start_date.month == 1 and start_date.day > 1):
        first_full_year = datetime.date(start_date.year + 1, 1, 1)
    if end_date.month < 12 or (end_date.month == 12 and end_date.day < 31):
        last_full_year = datetime.date(end_date.year - 1, 12, 31)

    # Generate the full year date range
    full_year_dates = pd.date_range(first_full_year, last_full_year, freq='D')
    return full_year_dates

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

def extract_center_spatial_region(ds, x_dim='x', y_dim='y', region_size=32):
    """
    Extracts the center spatial region from an xarray dataset.
    
    Parameters:
    - ds (xarray.Dataset or xarray.DataArray): The input dataset or data array.
    - x_dim (str): The name of the x-dimension in the dataset.
    - y_dim (str): The name of the y-dimension in the dataset.
    - region_size (int): The size of the square region to extract.
    
    Returns:
    - xarray.Dataset or xarray.DataArray: A dataset or data array containing the center region.
    """
    # Get the spatial dimensions
    x_size = ds.sizes[x_dim]
    y_size = ds.sizes[y_dim]
    
    # Calculate the start and end indices for the center crop
    x_start = (x_size - region_size) // 2
    x_end = x_start + region_size
    y_start = (y_size - region_size) // 2
    y_end = y_start + region_size
    
    # Slice the dataset
    center_region = ds.isel({x_dim: slice(x_start, x_end), y_dim: slice(y_start, y_end)})
    
    return center_region


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

    Era5_ds = extract_center_spatial_region(Era5_ds, x_dim='x', y_dim='y', region_size=32)
    Obs_ds = extract_center_spatial_region(Obs_ds, x_dim='x', y_dim='y', region_size=32)    

    # training and validation dataset
    Era5_train, Obs_train = Era5_ds.sel(time=CALIB_PERIOD), Obs_ds.sel(time=CALIB_PERIOD)
    Era5_valid, Obs_valid = Era5_ds.sel(time=VALID_PERIOD), Obs_ds.sel(time=VALID_PERIOD)
    Era5_test, Obs_test = Era5_ds.sel(time=TEST_PERIOD), Obs_ds.sel(time=TEST_PERIOD)

    Era5_train = doy_encoding(Era5_train, Obs_train, doy=DOY)
    Era5_valid = doy_encoding(Era5_valid, Obs_valid, doy=DOY)
    Era5_test = doy_encoding(Era5_test, Obs_test, doy=DOY)
    
    transpose = False
    
    if transpose==True:
    
        predictors_train = stacker(Era5_train).transpose(3, 2, 0, 1).compute()
        predictors_valid = stacker(Era5_valid).transpose(3, 2, 0, 1).compute()
        predictand_train = stacker(Obs_train).transpose(3, 2, 0, 1)
        predictand_valid = stacker(Obs_valid).transpose(3, 2, 0, 1)
    
    else:
        predictors_train = stacker(Era5_train).compute()
        predictors_valid = stacker(Era5_valid).compute()
        predictand_train = stacker(Obs_train)
        predictand_valid = stacker(Obs_valid)
        predictors_test = stacker(Era5_test).compute()
        predictand_test = stacker(Obs_test)


    # Instantiate models and optimizers
    generator = Generator()
    discriminator = PatchGANDiscriminator()
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss for Patch-GAN

    lr_G = 1e-4
    lr_D = 1e-4
    optim_beta = (0.5, 0.9)
    optimizer_G = optim.Adam(generator.parameters(), lr=lr_G, betas=optim_beta)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_D, betas=optim_beta)

    scheduler_step_size = 10
    gamma = 0.8
    # Scheduler initialization

    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_G,
    mode='min',               # Minimize the loss
    factor=0.8,               # Multiply the learning rate by this factor (e.g., reduce by 10x)
    patience=5,               # Number of epochs with no improvement to wait before reducing
    threshold=1e-4,           # Minimum improvement to be considered as progress
    cooldown=2,               # Cooldown period before resuming normal operation
    min_lr=1e-8  # Prints a message when reducing the learning rate
    )

    scheduler_D = StepLR(optimizer_D, step_size=scheduler_step_size, gamma=gamma)
    
    # Set up parameters and data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)
    batch_size = 1  # Define batch size as per your requirement
    num_epochs = 300  # Define the number of training epochs
    window_size = 5

    patience = 35
    min_delta = 0.0001

    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(predictors_train)
    y_tensor = torch.from_numpy(predictand_train)

    X_tensor = normalize_tensor(X_tensor)
    
    # Convert to PyTorch tensors
    X_tensor_test = torch.from_numpy(predictors_test)
    y_tensor_test = torch.from_numpy(predictand_test)

    X_tensor_test = normalize_tensor(X_tensor_test)

    # Verify shapes
    LogConfig.init_log(f'Shape of X (Input): {X_tensor.shape}')  # Expected: (161, 96, 5844, 15)
    LogConfig.init_log(f'Shape of y (Target): {y_tensor.shape}')  # Expected: (161, 96, 5844, 1)
    
    LogConfig.init_log(f'Shape of X_test (Input): {X_tensor_test.shape}')  # Expected: (161, 96, 5844, 15)
    LogConfig.init_log(f'Shape of y_test (Target): {y_tensor_test.shape}')  # Expected: (161, 96, 5844, 1)
    
    dataloader_train = prepare_data(X_tensor, y_tensor, window_size=window_size, batch_size=batch_size)
    dataloader_test = prepare_data(X_tensor_test, y_tensor_test, window_size=window_size, batch_size=batch_size)
    
    with mlflow.start_run():
        mlflow.log_param("learning_rate_G", lr_G)
        mlflow.log_param("learning_rate_D", lr_D)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("patience", patience)
        mlflow.log_param("StepLR_scheduler_step_size", scheduler_step_size)
        mlflow.log_param("StepLR_gamma", gamma)
        mlflow.log_param("optim_beta", optim_beta)
    
        mlflow.pytorch.autolog()
    
        train_with_mlflow_train_test(
            dataloader_train=dataloader_train,
            dataloader_test=dataloader_test,
            generator=generator,
            discriminator=discriminator,
            optimizer_G=optimizer_G,
            optimizer_D=optimizer_D,
            scheduler_G=scheduler_G, 
            scheduler_D=scheduler_D,
            num_epochs=num_epochs,
            patience=patience,
            min_delta=min_delta,
            device=device
        )

    # Define batch size
    save_dir = "checkpoints"
    
    X_tensor_valid = torch.from_numpy(predictors_valid)
    
    X_tensor_valid = normalize_tensor(X_tensor_valid)
    
    dataloader_valid = prepare_data_prediction(X_tensor_valid, window_size=window_size)
    
    generator.eval()
    
    # Move to the correct device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate predictions
    predicted_outputs = []
    with torch.no_grad():  # No gradients needed during inference
        # Iterate through the DataLoader
        for i, input_window in enumerate(dataloader_valid):
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
    
    tiemd = get_full_year_dates_from_dataset(Era5_valid)
    # store predictions in xarray.Dataset
    predictions = xr.DataArray(data=prediction, dims=['time', 'lat', 'lon'], coords=dict(time=tiemd, lat=Obs_valid.y, lon=Obs_valid.x))
    predictions = predictions
    predictions = predictions.to_dataset(name=PREDICTAND)
    
    predictions = predictions.set_index(
        time='time',
        lat='lat',
        lon='lon'
    )
        
    predictions.to_netcdf("/home/sdhinakaran/eurac/downScaleML-refactoring/downscaleml/main/ESRGAN_tasmean_2016_downscaled_T_seq_lr_change_small_region_rdbINC_early_stopping.nc")
    
    LogConfig.init_log('Prediction Saved!!! SMILE PLEASE')


