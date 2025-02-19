# Standard library imports
import sys
import os
import time
import logging
import subprocess
import pathlib
from datetime import datetime, timedelta
from logging.config import dictConfig

# Data processing and ML libraries
import numpy as np
import pandas as pd
import xarray as xr
import dask
from dask.distributed import LocalCluster, Client
import optuna
from optuna.integration import LightGBMPruningCallback
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

# PyTorch-related imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.autograd import grad
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchmetrics import MeanMetric
from torch.utils.tensorboard import SummaryWriter
from pytorch_msssim import MS_SSIM

from torch.autograd import grad
from torchvision.models import vgg16  # For perceptual loss

# Image processing
from skimage.metrics import structural_similarity

# Visualization
import matplotlib.pyplot as plt

# MLflow
import mlflow
import mlflow.pytorch
from mlflow import MlflowClient

# Downscaleml imports
from downscaleml.core.hpo import objective
from downscaleml.core.dataset import (
    ERA5Dataset, NetCDFDataset, EoDataset, DatasetStacker, 
    SummaryStatsCalculator, doy_encoding
)
from downscaleml.core.clustering import ClusteringWorkflow
from downscaleml.main.config_reanalysis_dev_no_slide import (
    ERA5_PATH, OBS_PATH, DEM_PATH, MODEL_PATH, TARGET_PATH,
    NET, ERA5_PLEVELS, ERA5_PREDICTORS, PREDICTAND,
    CALIB_PERIOD, VALID_PERIOD, DOY, NORM,
    OVERWRITE, DEM, DEM_FEATURES, STRATIFY, WET_DAY_THRESHOLD,
    VALID_SIZE, start_year, end_year, CHUNKS, TEST_PERIOD, RESULT_PERIOD
)
from downscaleml.core.constants import (
    ERA5_P_VARIABLES, ERA5_P_VARIABLES_SHORTCUT, ERA5_P_VARIABLE_NAME,
    ERA5_S_VARIABLES, ERA5_S_VARIABLES_SHORTCUT, ERA5_S_VARIABLE_NAME,
    ERA5_VARIABLES, ERA5_VARIABLE_NAMES, ERA5_PRESSURE_LEVELS,
    PREDICTANDS
)
from downscaleml.core.utils import (
    NAMING_Model, normalize, search_files, LogConfig
)
from downscaleml.core.logging import log_conf
from einops import rearrange

from torchvision.models import vgg16  # For perceptual loss

from torch.optim.lr_scheduler import CosineAnnealingLR
from einops import rearrange
