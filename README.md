# downscaleml

# Overview

## Changes in Downscaling Approach

After building the prototype of the downscaling model, significant changes have been introduced to the approach for downscaling the SEAS5 dataset. Instead of performing downscaling in one shot, the new method uses a two-stage downscaling approach:

1. **Stage 1 Downscaling**: Increases the spatial resolution from 25 km to 5.5 km.
2. **Stage 2 Downscaling**: Utilizes a patch-based ESRGAN model to further downscale the resolution to 1 km.

This required intensive development, including:

- Building a modified ESRGAN.
- Training the ESRGAN from scratch.
- Iterative testing for network architecture and training schema optimization.
- Hyperparameter optimization (HPO) in Stage 2, which is currently in progress.

The package is under intensive development, and the current distributable file only supports Stage 1 Downscaling and Grid Search. The remaining features are actively being developed.


# downscaleml - v0.1.0

First information first! - 'Installing the Package'
The dist folder contains the **''downscaleml"** package, install it using `pip install downscaleml-0.1.0.tar.gz`. Make sure you already have GDAL dependencies installed in your conda/venv/any_environments. If suppose you face problems with the GDAL installation in your system as well as in your environment, don't worry, I am here for you.

This package requires GDAL==3.4.3. 

Follow this link to keep your GDAL installation clean and working:
https://mothergeo-py.readthedocs.io/en/latest/development/how-to/gdal-ubuntu-pkg.html

You could also probably face a problem similar to what i faced, after installation of GDAL, it's the **libstdc++.so.6** linkage problem. This is to do with either **path linking** or **file missing in the directory**. This file has to be found in your system and to be linked with the virtual environment you are working. This can be done by just following **stackoverflow**. I can drop a little clue, which can help you in linking, kindly change the paths relative to your system.
`ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /home/anavani/anaconda3/envs/dmcgb/bin/../lib/libstdc++.so.6` 

You could also follow compatibility issue beterrn GDAL and Numpy or GDAL array now, you could arrest this issue by following the process:-

`pip uninstall gdal`
`pip install numpy`
`pip install GDAL==$(gdal-config --version) --global-option=build_ext --global-option="-I/usr/include/gdal"`

# Recommended steps to get going!

1. Clone the git project in your local.
2. `pip install poetry` in your virtual environment
3. `poetry install` in the project local with the same local environment

## How to effectively use the package?

The data is not provided in this package, the paths for the input-output data is provided in the `downscaleml/main/config.py`. You can make necessary changes here to reflect elsewhere in the project.

Modify and use the script `/scripts/run_model.sh` to have control with the important parameters and the model output.

To extract the best hyperparameter for any given downscaling model, run the `downScaleML/downscaleml
/main/grid_search_downScale.py`, by tweaking the `combination` parameter in the `downscaleml/main/config.py`, one can control the running time to fetch the best hyperparameter. The combination parameter controls the randomised reduced grid size for optimised run time of hyperparameter search.

This script `downscaleml/main/preprocessing/combined_preprocess_CERRA.py` preprocesses CERRA (Copernicus European Regional Reanalysis) data by aggregating it to daily data. It supports reprojection and resampling to a target grid, with specific adjustments for temperature and precipitation data.

Example for running this scirpt: `python combined_preprocess_CERRA.py --source /path/to/source --target /path/to/target --reproject --variable 2m_temperature`

Similarly the script `downscaleml/main/preprocessing/combined_preprocess_ERA5.py` preprocesses ERA-5 (European Reanalysis) data by aggregating it to daily data. It supports reprojection and resampling to a target grid, with specific adjustments for temperature and precipitation data.

Example for running this scirpt: `python preprocess_era5.py --source /path/to/source --target /path/to/target --reproject --variable 2m_temperature`