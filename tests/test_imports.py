import pytest

def test_imports():
    try:
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
        import plotly
        import optuna
        from sklearn.metrics import mean_squared_error
        import math
        import random
        from functools import partial
        
        # externals
        import xarray as xr
        
        from xgboost import XGBRegressor
        from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
        from lightgbm import LGBMRegressor
        from sklearn.metrics import r2_score, mean_absolute_error
        from sklearn.model_selection import train_test_split
        
        # locals
        from downscaleml.core.dataset import ERA5Dataset, NetCDFDataset, EoDataset
        
        from downscaleml.main.config import (ERA5_PATH, OBS_PATH, DEM_PATH, MODEL_PATH, TARGET_PATH, NET, ERA5_PLEVELS, ERA5_PREDICTORS, PREDICTAND,
                                             CALIB_PERIOD, VALID_PERIOD, DOY, NORM,
                                             OVERWRITE, DEM, DEM_FEATURES, STRATIFY,
                                             WET_DAY_THRESHOLD, VALID_SIZE, 
                                             start_year, end_year, CHUNKS, combination, paramss)
        
        from downscaleml.core.constants import (ERA5_P_VARIABLES, ERA5_P_VARIABLES_SHORTCUT, ERA5_P_VARIABLE_NAME,
                                                ERA5_S_VARIABLES, ERA5_S_VARIABLES_SHORTCUT, ERA5_S_VARIABLE_NAME,
                                                ERA5_VARIABLES, ERA5_VARIABLE_NAMES, ERA5_PRESSURE_LEVELS,
                                                PREDICTANDS, ERA5_P_VARIABLES, ERA5_S_VARIABLES)
        
        from downscaleml.core.utils import NAMING_Model, normalize, search_files, LogConfig
        from downscaleml.core.logging import log_conf
        
        from matplotlib import cm
        from matplotlib.colors import ListedColormap
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        import scipy.stats as stats
        from IPython.display import Image
        from sklearn.metrics import r2_score
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")
