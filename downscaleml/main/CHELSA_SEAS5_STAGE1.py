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

# externals
import xarray as xr

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# locals
from downscaleml.core.dataset import ERA5Dataset, NetCDFDataset, EoDataset

from downscaleml.main.emo1_config_reanalysis_dev import (ERA5_PATH, OBS_PATH, DEM_PATH, MODEL_PATH, TARGET_PATH, SEAS5_PATH,
                                     NET, ERA5_PLEVELS, ERA5_PREDICTORS, PREDICTAND,
                                     CALIB_PERIOD, VALID_PERIOD, DOY, NORM,
                                     OVERWRITE, DEM, DEM_FEATURES, STRATIFY, WET_DAY_THRESHOLD,
                                     VALID_SIZE, start_year, end_year, CHUNKS, SEAS5_CHUNKS_forecast, params)

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
    
# module level logger
LOGGER = logging.getLogger(__name__)

def stacker(xarray_dataset):
    # stack along the lat and lon dimensions
    stacked = xarray_dataset.stack()
    dask_arr = stacked.to_array().data
    xarray_dataset = dask_arr.T
    LogConfig.init_log('Shape is in (spatial, time, variables):{}'.format(xarray_dataset.shape))
    return xarray_dataset

def doy_encoding(X, y=None, doy=False):

    # whether to include the day of the year as predictor variable
    if doy:
        # add doy to set of predictor variables
        LOGGER.info('Adding day of the year to predictor variables ...')
        X = X.assign(EoDataset.encode_doys(X, chunks=X.chunks))

    print(X)
    return X

def load_chunk(dataset, dim_name, chunk_index):
    # Get the chunk sizes for the specified dimension
    chunk_sizes = dataset.chunks[dataset.get_index(dim_name).name]

    # Calculate the start and end indices for the desired chunk
    start_idx = sum(chunk_sizes[:chunk_index])  # Start of the chunk
    end_idx = start_idx + chunk_sizes[chunk_index]  # End of the chunk

    # Select and load the chunk
    chunk = dataset.isel({dim_name: slice(start_idx, end_idx)}).load()

    return chunk

def calculate_statistics_to_dataset(first_chunk, dim='time'):
    # Dictionary to hold new DataArrays for each statistic
    new_data_vars = {}

    # Iterate over each data variable in the dataset
    for var_name, data_array in first_chunk.data_vars.items():
        # Calculate the desired statistics over the specified dimension
        percentiles = data_array.quantile([0.10, 0.90], dim=dim)
        #median = data_array.median(dim=dim)
        std_val = data_array.std(dim=dim)
        mean_val = data_array.mean(dim=dim)

        # Select the 10th and 90th percentiles and drop the 'quantile' dimension to avoid conflict
        percentile_10 = percentiles.sel(quantile=0.10).drop_vars('quantile')
        percentile_90 = percentiles.sel(quantile=0.90).drop_vars('quantile')

        # Create new variables with the statistics, appending the variable name with the statistic
        new_data_vars[f"{var_name}_10th_percentile"] = percentile_10
        new_data_vars[f"{var_name}_90th_percentile"] = percentile_90
        #new_data_vars[f"{var_name}_median"] = median
        new_data_vars[f"{var_name}_std"] = std_val
        new_data_vars[f"{var_name}_mean"] = mean_val

    # Create a new xarray dataset with the statistics
    stats_dataset = xr.Dataset(new_data_vars)

    return stats_dataset

def silhouette_analysiss(data, max_k=10):
    silhouette_scores = []

    for k in range(2, max_k+1):  # Start from 2 clusters
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        score = silhouette_score(data, cluster_labels)
        silhouette_scores.append(score)

    # Plot the Silhouette scores
    plt.plot(range(2, max_k+1), silhouette_scores, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis for Optimal k')
    plt.savefig("cluster.png")
    #plt.show()

def find_optimal_clusterss(data, max_k=10):
    wcss = []  # List to store the within-cluster sum of squares for each k

    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)  # Inertia is the sum of squared distances to cluster centers

    # Plot the Elbow graph
    plt.plot(range(1, max_k+1), wcss, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS (Inertia)')
    plt.title('Elbow Method for Optimal k')
    plt.savefig("elbow.png")
    #plt.show()

def silhouette_analysis(data, max_k=10):
    silhouette_scores = []
    
    for k in range(2, max_k + 1):  # Start from 2 clusters
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        score = silhouette_score(data, cluster_labels)
        silhouette_scores.append(score)

    # Plot the Silhouette scores
    plt.plot(range(2, max_k + 1), silhouette_scores, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis for Optimal k')
    plt.savefig("cluster.png")
    
    # Find the optimal k based on the highest Silhouette Score
    optimal_k_silhouette = silhouette_scores.index(max(silhouette_scores)) + 2  # Add 2 since index starts at 0 for k=2
    return silhouette_scores, optimal_k_silhouette

def find_optimal_clusters(data, max_k=10):
    wcss = []

    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    # Plot the Elbow graph
    plt.plot(range(1, max_k + 1), wcss, 'bx-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS (Inertia)')
    plt.title('Elbow Method for Optimal k')
    plt.savefig("elbow.png")
    
    # Automatically detect the elbow point (maximum rate of decrease)
    first_derivative = np.diff(wcss)
    second_derivative = np.diff(first_derivative)
    
    # Find the "elbow" where the second derivative is the largest negative value
    optimal_k_elbow = np.argmin(second_derivative) + 2  # Add 2 since np.diff reduces by 1 and we start from k=2
    return wcss, optimal_k_elbow

def apply_kmeans(data, n_clusters):
    """
    Applies KMeans clustering to the data and returns the cluster labels.
    
    Parameters:
    data (array-like): The input data to cluster.
    n_clusters (int): The number of clusters to form, determined dynamically.
    
    Returns:
    array: The cluster labels for each data point.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data)
    return cluster_labels


def select_optimal_k(silhouette_optimal_k, elbow_optimal_k):
    # Combine both methods by averaging the two results
    optimal_k = round((silhouette_optimal_k + elbow_optimal_k) / 2)
    return optimal_k

# Create clustered map
def create_clustered_map(cluster_labels, statistics_dataset):
    """
    Reshapes the cluster labels into the shape of the statistics dataset
    and returns a DataArray for plotting.
    
    Parameters:
    cluster_labels (array): Cluster labels from KMeans.
    statistics_dataset (xarray.Dataset): Dataset containing coordinates for reshaping.
    
    Returns:
    xarray.DataArray: The reshaped and plotted cluster map.
    """
    clustered_map = cluster_labels.reshape(statistics_dataset.sizes['y'], statistics_dataset.sizes['x'])
    cluster_map = xr.DataArray(clustered_map, coords=[statistics_dataset['y'], statistics_dataset['x']], dims=['y', 'x'])
    cluster_map.plot()
    return cluster_map

def full_clustering_workflow(statistics_dataset, data, max_k=10):
    """
    Runs the full workflow to perform clustering and return a clustered map.
    
    Parameters:
    statistics_dataset (xarray.Dataset): Dataset containing the statistics and coordinates.
    data (array-like): The input data to be clustered.
    max_k (int): Maximum number of clusters to consider.
    
    Returns:
    xarray.DataArray: The final clustered map for plotting.
    """
    # Step 1: Perform Silhouette and Elbow analysis
    silhouette_scores, optimal_k_silhouette = silhouette_analysis(data, max_k=max_k)
    wcss, optimal_k_elbow = find_optimal_clusters(data, max_k=max_k)

    # Step 2: Select optimal k by combining Silhouette and Elbow results
    optimal_k = select_optimal_k(optimal_k_silhouette, optimal_k_elbow)
    print(f"Optimal k based on Silhouette: {optimal_k_silhouette}, Elbow: {optimal_k_elbow}, Final Selected: {optimal_k}")

    # Step 3: Apply KMeans with the dynamically determined optimal k
    cluster_labels = apply_kmeans(data, n_clusters=optimal_k)

    # Step 4: Reshape cluster labels and plot the clustered map
    clustered_map = create_clustered_map(cluster_labels, statistics_dataset)

    return clustered_map




def compute_summary_stats(ds, temp_threshold=303.15):
    """
    Compute summary statistics for DEM, slope, aspect, and 2m_temperature, reducing over time.
    
    Parameters:
    ds (xarray.Dataset): Input dataset with variables 'dem', 'slope', 'aspect', and '2m_temperature'.
    temp_threshold (float): Threshold to compute extreme temperature frequency (default is 30°C).
    
    Returns:
    xarray.Dataset: A new dataset with computed summary statistics.
    """
    
    # Compute temperature-related metrics
    temperature = ds['t2m']
    
    temp_max = temperature.max(dim='time')
    temp_min = temperature.min(dim='time')
    temp_range = temp_max - temp_min
    # Quantile calculation with renaming and selecting the scalar value
    temp_95th = temperature.quantile(0.95, dim='time').squeeze(drop=True).rename('temp_95th')
    temp_5th = temperature.quantile(0.05, dim='time').squeeze(drop=True).rename('temp_5th')
    temp_extreme_freq = (temperature > temp_threshold).sum(dim='time')
    
    # DEM metrics
    dem = ds['elevation']
    dem_mean = dem.mean(dim='time')
    dem_max = dem.max(dim='time')
    dem_min = dem.min(dim='time')
    dem_range = dem_max - dem_min
    dem_variance = dem.var(dim='time')
    
    # Slope metrics
    slope = ds['slope']
    slope_mean = slope.mean(dim='time')
    slope_max = slope.max(dim='time')
    slope_min = slope.min(dim='time')
    slope_range = slope_max - slope_min
    slope_variance = slope.var(dim='time')
    
    # Aspect metrics
    aspect = ds['aspect']
    aspect_variability = aspect.var(dim='time')
    aspect_mean = aspect.mean(dim='time')  # You could also use circular statistics here if necessary
    
    # Combine all the computed metrics into a list of datasets
    stats_list = [
        xr.Dataset({'temp_max': temp_max}),
        xr.Dataset({'temp_min': temp_min}),
        xr.Dataset({'temp_range': temp_range}),
        xr.Dataset({'temp_95th': temp_95th}),
        xr.Dataset({'temp_5th': temp_5th}),
        xr.Dataset({'temp_extreme_freq': temp_extreme_freq}),
        xr.Dataset({'dem_mean': dem_mean}),
        xr.Dataset({'dem_max': dem_max}),
        xr.Dataset({'dem_min': dem_min}),
        xr.Dataset({'dem_range': dem_range}),
        xr.Dataset({'dem_variance': dem_variance}),
        xr.Dataset({'slope_mean': slope_mean}),
        xr.Dataset({'slope_max': slope_max}),
        xr.Dataset({'slope_min': slope_min}),
        xr.Dataset({'slope_range': slope_range}),
        xr.Dataset({'slope_variance': slope_variance}),
        xr.Dataset({'aspect_variability': aspect_variability}),
        xr.Dataset({'aspect_mean': aspect_mean})
    ]
    
    # Merge all datasets, ignoring conflicts with 'compat=override'
    summary_stats = xr.merge(stats_list, compat='override')
    
    return summary_stats

# Elbow Method for WCSS (Inertia)
def find_wcss(data, max_k=10):
    wcss = []

    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    return wcss

def silhouette(data, max_k=10):
    silhouette_scores = []

    for k in range(2, max_k + 1):  # Start from 2 clusters
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        score = silhouette_score(data, cluster_labels)
        silhouette_scores.append(score)

    return silhouette_scores

def plot_silhouette_vs_wcss(data, max_k=10):
    silhouette_scores = silhouette(data, max_k)
    wcss = find_wcss(data, max_k)

    # Create figure and axis
    fig, ax1 = plt.subplots()

    # Plot Silhouette Scores on the left y-axis
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Silhouette Score', color='tab:blue')
    ax1.plot(range(2, max_k + 1), silhouette_scores, 'bx-', label='Silhouette Score', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Create a twin axis for the WCSS plot
    ax2 = ax1.twinx()
    ax2.set_ylabel('WCSS (Inertia)', color='tab:red')
    ax2.plot(range(1, max_k + 1), wcss, 'rx-', label='WCSS (Inertia)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Add title and show the plot
    plt.title('Silhouette Score vs WCSS (Inertia) for Optimal k')
    fig.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig("silhoutte_vs_variance.png")

# Example of how to run the plot
# Assuming you have your data matrix 'data'
# data = ...
def plot_data_array(data_array, title='Data Array', cmap='viridis', vmin=None, vmax=None):
    plt.figure(figsize=(10, 6))
    
    # Plotting the data
    data_array.plot(cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Adding title and labels
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Show colorbar
    #plt.colorbar(label=data_array.name if data_array.name else 'Value')
    
    # Show plot
    plt.savefig("Clustered_MAp.png")

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
        ' - '.join([str(CALIB_PERIOD[0]), str(CALIB_PERIOD[-1])])))

    # initialize ERA5 predictor dataset
    LogConfig.init_log('Initializing ERA5 predictors.')
    Era5 = ERA5Dataset(ERA5_PATH.joinpath('ERA5'), ERA5_PREDICTORS,
                       plevels=ERA5_PLEVELS)
    Era5_ds = Era5.merge(chunks=CHUNKS)
    Era5_ds = Era5_ds.rename({'lon': 'x','lat': 'y'})
    LogConfig.init_log('{}'.format(Era5_ds))    

    LogConfig.init_log('Initializing SEAS5 predictors.')

    #print(SEAS5_year)
    
    #TEMPORARY PATHWORK HERE - SOLVE IT FOR LATER
    Seas5 = ERA5Dataset(SEAS5_PATH.joinpath("SEAS5_STAGE1"), ERA5_PREDICTORS,
                       plevels=ERA5_PLEVELS)
    print(Seas5)
    Seas5_ds = Seas5.merge(chunks=SEAS5_CHUNKS_forecast)
    Seas5_ds = xr.unify_chunks(Seas5_ds)
    Seas5_ds = Seas5_ds[0]
    Seas5_ds = Seas5_ds.rename({'lon': 'x','lat': 'y'})
    
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
        dem_path = search_files(DEM_PATH, '^interTwin_dem_chelsa_stage1.nc$').pop()

        # read elevation and compute slope and aspect
        dem = ERA5Dataset.dem_features(
            dem_path, {'y': Era5_ds.y, 'x': Era5_ds.x},
            add_coord={'time': Era5_ds.time})

        dem_seas5 = ERA5Dataset.dem_features(
            dem_path, {'y': Seas5_ds.y, 'x': Seas5_ds.x},
            add_coord={'time': Seas5_ds.time})
        
        LogConfig.init_log('{}'.format(Era5_ds.chunks))

        # check whether to use slope and aspect
        if not DEM_FEATURES:
            dem = dem.drop_vars(['slope', 'aspect']).chunk(Era5_ds.chunks)
            dem_seas5_n = dem_seas5.merge(dem_seas5.expand_dims(number=Seas5_ds['number'])).chunk(Seas5_ds.chunks)
            dem_seas5_n = dem_seas5_n.drop_vars(['slope', 'aspect']).chunk(Seas5_ds.chunks)

        # add dem to set of predictor variables
        dem = dem.chunk(Era5_ds.chunks)
        dem_seas5_n = dem_seas5_n.chunk(Seas5_ds.chunks)
        Era5_ds = xr.merge([Era5_ds, dem])
        Seas5_ds = xr.merge([Seas5_ds, dem_seas5_n])


    # initialize training data
    LogConfig.init_log('Initializing training data.')

    #Era5_ds = Era5_ds.sel(x=slice(10.35, 12.52), y=slice(46.16, 47.15))
    #Obs_ds = Obs_ds.sel(x=slice(10.35, 12.52), y=slice(46.16, 47.15))

    # training and validation dataset
    Era5_train, Obs_train = Era5_ds.sel(time=CALIB_PERIOD), Obs_ds.sel(time=CALIB_PERIOD)
    Seas5_ds = Seas5_ds.sel(time=VALID_PERIOD)
    dem_seas5 = dem_seas5.sel(time=VALID_PERIOD)

    Era5_train = doy_encoding(Era5_train, Obs_train, doy=DOY)
    dem_seas5 = doy_encoding(dem_seas5, doy=DOY)

    Seas5_ds = Seas5_ds.merge(dem_seas5.expand_dims(number=Seas5_ds['number'])).chunk(Seas5_ds.chunks)
    Seas5_ds = Seas5_ds.drop_vars(['slope', 'aspect'])
    Seas5_ds = Seas5_ds.transpose('time', 'y', 'x', 'number')
    
    LogConfig.init_log('Era5_Train')
    print(Era5_train)
    LogConfig.init_log('SEAS5_Training Data Here')
    print(Seas5_ds)

    predictors_train = Era5_train
    predictors_valid = Seas5_ds
    predictand_train = Obs_train

    predictors_train = stacker(predictors_train).compute()
    predictand_train = stacker(predictand_train)
    
    LogConfig.init_log('Dask computations done!')
    # iterate over the grid points
    LogConfig.init_log('Downscaling by Random Forest Starts: iterating each grid cell over time dimension')
    
    Models = {
        'RandomForestRegressor' : RandomForestRegressor,
        'XGBRegressor' : XGBRegressor,
        'AdaBoostRegressor': AdaBoostRegressor,
        'LGBMRegressor': LGBMRegressor,
    }
    Model_name = NET

    Era5_ds = None

    LogConfig.init_log('predictors_train shape[0] : {}'.format(predictors_train.shape[0]))
    LogConfig.init_log('predictors_train shape[1] : {}'.format(predictors_train.shape[1]))
    LogConfig.init_log('predictors_train shape[2] : {}'.format(predictors_train.shape[2]))
    LogConfig.init_log('predictors_valid shape[3] : {}'.format(predictors_train.shape[3]))

    for m in range(len(predictors_valid["number"])):
        
        point_valid = stacker(predictors_valid.isel(number=m)).compute()
        print(point_valid.shape)
        prediction = np.ones(shape=(point_valid.shape[2], point_valid.shape[1], point_valid.shape[0])) * np.nan
        
        for i in range(predictors_train.shape[0]):
            for j in range(predictors_train.shape[1]):

                point_predictors = predictors_train[i, j, :, :]
                point_predictors = normalize(point_predictors)
                point_predictand = predictand_train[i, j, :, :]

                # check if the grid point is valid
                if np.isnan(point_predictors).any() or np.isnan(point_predictand).any():
                    # move on to next grid point
                    continue

                # prepare predictors of validation period
                point_validation = point_valid[i, j, :, :]
                point_validation = normalize(point_validation)

                LogConfig.init_log('Current grid point: ({:d}), ({:d}), ({:d}) '.format(m, i, j))    
                # normalize each predictor variable to [0, 1]
                # point_predictors = normalize(point_predictors)

                # instanciate the model for the current grid point
                model = LGBMRegressor(**params)

                print(point_predictors.shape, point_predictand.flatten().shape)

                #point_predictors = np.array(point_predictors)
                #point_predictand = np.array(point_predictand).flatten()

                # train model on training data
                model.fit(point_predictors, point_predictand.ravel())
                # predict validation period
                pred = model.predict(point_validation)
                #LogConfig.init_log('Processing grid point: {:d}, {:d}, {:d} - score: {:.2f}'.format(m, i, j, r2_score(predictand_validation, pred)))

                # store predictions for current grid point
                prediction[:, j, i] = pred

        LogConfig.init_log('Model ensemble for ensemble member {} saved and Indexed'.format(str(m)))
    

        # store predictions in xarray.Dataset
        predictions = xr.DataArray(data=prediction, dims=['time', 'y', 'x'],
                                coords=dict(time=pd.date_range(Seas5_ds.time.values[0],Seas5_ds.time.values[-1], freq='D'),
                                            lat=Seas5_ds.y, lon=Seas5_ds.x))
        predictions = predictions.to_dataset(name=PREDICTAND)

        predictions = predictions.set_index(
            time='time',
            y='lat',
            x='lon'
        )
        
        # initialize network filename
        predict_file = NAMING_Model.state_file(
            NET, PREDICTAND, ERA5_PREDICTORS, ERA5_PLEVELS, WET_DAY_THRESHOLD, dem=DEM,
            dem_features=DEM_FEATURES, doy=DOY, stratify=STRATIFY)

        LogConfig.init_log('Target Location'.format(str(target.parent)))

        predictions.to_netcdf("{}/{}_{}_{}_egu_tasmean.nc".format(str(target.parent), str(predict_file), "2020_I", str(m)))

    LogConfig.init_log('Prediction Saved!!! SMILE PLEASE')
    
    
