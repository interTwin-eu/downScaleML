import datetime
import numpy as np
import pathlib


from downscaleml.core.constants import (PREDICTANDS, ERA5_P_VARIABLES,
                                   ERA5_S_VARIABLES, MODELS)

PREDICTAND='tasmean'
assert PREDICTAND in PREDICTANDS

NET='LGBMRegressor'
assert NET in MODELS

ERA5="CHELSA_STAGE2_PREDICTORS"

ROOT = pathlib.Path('/mnt/CEPH_PROJECTS/InterTwin/Climate_Downscaling/two_stage/')

# path to this file
HERE = pathlib.Path(__file__).parent

ERA5_PATH = ROOT.joinpath(ERA5)

SEAS5_PATH = ROOT

OBS_PATH = ROOT.joinpath('CHELSA_STAGE2')

DEM_PATH = ROOT.joinpath('DEM_STAGE2')

RESULTS = pathlib.Path('/mnt/CEPH_PROJECTS/InterTwin/Climate_Downscaling/two_stage/')

MODEL_PATH = RESULTS.joinpath('model_RESULTS')

TARGET_PATH = RESULTS.joinpath('RESULTS_STAGE2')

# SEAS5 hindcast or forecast

#SEAS5_type = "forecast"
# SEAS5_type = "forecast"

# include day of year as predictor
DOY = True

# whether to normalize the training data to [0, 1]
NORM = True

# whether to overwrite existing models
OVERWRITE = False

# whether to use DEM slope and aspect as predictors
DEM_FEATURES = False

# stratify training/validation set for precipitation by number of wet days
STRATIFY = False

# size of the validation set w.r.t. the training set
# e.g., VALID_SIZE = 0.2 means: 80% of CALIB_PERIOD for training
#                               20% of CALIB_PERIOD for validation
VALID_SIZE = 0.2

WET_DAY_THRESHOLD=0

CHUNKS = {'time': 100, 'x': 300, 'y': 270}
#SEAS5_CHUNKS_forecast = {'time': 1429, 'x': 161, 'y': 96, 'number': 1}
#SEAS5_CHUNKS_forecast = {'number': 1, 'x': 161, 'y': 96, 'time': 365}
#SEAS5_CHUNKS = {'time': 365, 'x': 161, 'y': 96, 'number': 1}

# threshold  defining the minimum amount of precipitation (mm) for a wet day
#WET_DAY_THRESHOLD=0

# ERA5 pressure levels
ERA5_PLEVELS = []

if PREDICTAND is 'tasmean':
    ERA5_P_PREDICTORS = []
                          
    #ERA5_P_PREDICTORS = []
    assert all([var in ERA5_P_VARIABLES for var in ERA5_P_PREDICTORS])
    
    # ERA5 predictor variables on single levels
    ERA5_S_PREDICTORS=["2m_temperature"]
    
    #ERA5_S_PREDICTORS=["mean_sea_level_pressure", "total_precipitation"]
    assert all([var in ERA5_S_VARIABLES for var in ERA5_S_PREDICTORS])

elif PREDICTAND is 'pr':
    ERA5_P_PREDICTORS = []
                          
    #ERA5_P_PREDICTORS = []
    assert all([var in ERA5_P_VARIABLES for var in ERA5_P_PREDICTORS])
    
    # ERA5 predictor variables on single levels
    ERA5_S_PREDICTORS=["total_precipitation"]
    
    #ERA5_S_PREDICTORS=["mean_sea_level_pressure", "total_precipitation"]
    assert all([var in ERA5_S_VARIABLES for var in ERA5_S_PREDICTORS])

else:
    ERA5_P_PREDICTORS = []
                          
    #ERA5_P_PREDICTORS = []
    assert all([var in ERA5_P_VARIABLES for var in ERA5_P_PREDICTORS])
    
    # ERA5 predictor variables on single levels
    ERA5_S_PREDICTORS=["surface_solar_radiation_downward"]
    
    #ERA5_S_PREDICTORS=["mean_sea_level_pressure", "total_precipitation"]
    assert all([var in ERA5_S_VARIABLES for var in ERA5_S_PREDICTORS])
    
ERA5_PREDICTORS = ERA5_P_PREDICTORS + ERA5_S_PREDICTORS


DEM = True
if DEM:
    # remove model orography when using DEM
    if 'orography' in ERA5_S_PREDICTORS:
        ERA5_S_PREDICTORS.remove('orography')

#NET='AdaBoostRegressor'
#NET='AdaBoostRegressor'
#NET='AdaBoostRegressor'
#NET='AdaBoostRegressor'

CALIB_PERIOD = np.arange(
    datetime.datetime.strptime('2005-01-01', '%Y-%m-%d').date(),
    datetime.datetime.strptime('2016-01-01', '%Y-%m-%d').date())

TEST_PERIOD = np.arange(
    datetime.datetime.strptime('2016-01-01', '%Y-%m-%d').date(),
    datetime.datetime.strptime('2017-01-01', '%Y-%m-%d').date())

start_year = np.min(CALIB_PERIOD).astype(datetime.datetime).year
end_year = np.max(CALIB_PERIOD).astype(datetime.datetime).year

# validation period: testing
VALID_PERIOD = np.arange(
    datetime.datetime.strptime('2019-12-30', '%Y-%m-%d').date(),
    datetime.datetime.strptime('2020-08-06', '%Y-%m-%d').date())

RESULT_PERIOD = np.arange(
    datetime.datetime.strptime('2020-01-01', '%Y-%m-%d').date(),
    datetime.datetime.strptime('2020-08-03', '%Y-%m-%d').date())


