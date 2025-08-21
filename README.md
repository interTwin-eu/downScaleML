# downScaleML: openEO-enabled Downscaling Pipeline for Climate Data

````markdown

This repository provides an openEO-based, Docker-compatible, reproducible testing framework for downscaling Earth Observation (EO) data. It supports a modular data processing and machine learning pipeline for climate data—powered by STAC, Dask, and LightGBM. The tests are designed to validate processing and modeling logic using both public and restricted datasets.

---

## 📦 Repository Features

- **openEO-compatible test pipelines** using [openeo-processes-dask](https://github.com/Open-EO/openeo-processes-dask)
- **STAC-based data loading** for ERA5, SEAS5, DEM, and EMO1 products
- **Hybrid Improved Precipitation downscaling framework** using LightGBM
- **Docker-based isolated runtime** with all dependencies
- **Flexible Makefile-based automation**
- **AWS-secured workflows for SEAS5 datasets**

---

## 📂 Dataset Summary

| Dataset       | Access       | Spatial Extent                      | Temporal Extent             | Notes                                 |
|---------------|--------------|-------------------------------------|-----------------------------|---------------------------------------|
| ERA5          | Public       | 2°E–20°E, 40°N–52°N (Alps region)   | 2000–2020 (daily)           | Used in open pytests                  |
| SEAS5         | Requires AWS | Same as ERA5                        | 12 Inits(Aug '21 - July '22)| Used in closed pytests only           |
| EMO1, DEM     | Public       | Same as ERA5                        | 2000–2022 used              | EMO1 contains downstream targets      |

---
````
## 🚀 Quick Start

### 🐳 Run with Docker (Recommended)

#### 1. Build the Docker Image and automatically run public pytest

```bash
make setup 
````

This will build the DockerFile and run the  pre-processing and downScaling pipeline using:

```
/app/tests/test_downscaleml_pipeline.py
```


#### 2. Run in an Interactive Shell

```bash
make shell
```

You’ll drop into a Docker shell with the micromamba environment pre-activated.

#### 3. Clean Docker Containers

```bash
make clean
```

---

## 🔐 Running Closed Tests (with SEAS5 data)

Some tests require access to SEAS5 data through AWS-authenticated STAC endpoints. These tests will only work **if valid AWS credentials are provided.**

### Precondition:

Set your AWS credentials as environment variables:

```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

Then, run:

```bash
make run TEST_FILE=/app/tests/pytest_A.py
```

---

## 🧪 Test Coverage

### ✅ `test_downscaleml_pipeline.py`

* Loads open ERA5, DEM, and EMO1 datasets via STAC
* Performs resampling, cube merging, and `sin_cos_doy` feature expansion
* Saves output as `.zarr` and registers with raster2stac
* Trains pixel-based LightGBM models for the target variable
* Validates predictions and saves them as Zarr

### 🔐 `pytest_A.py` (closed tests)

* Same workflow as above, but includes SEAS5 datasets
* Requires valid AWS credentials

---

## 🛠️ Components Used

* **[downScaleML](https://github.com/interTwin-eu/downScaleML)**: core downscaling package
* **[raster2stac](https://gitlab.inf.unibz.it/earth_observation_public/raster-to-stac)**: Zarr-to-STAC converter
* **[openeo-processes-dask](https://github.com/interTwin-eu/openeo-processes-dask)**: local execution of openEO processes
* **LightGBM + scikit-learn**: pixel-based regression models
* **Dask**: parallel computation backend
* **Micromamba**: lightweight conda environment manager

---

## 🧬 Environment Setup (outside Docker, optional)

```bash
micromamba env create -f environment.yml
micromamba activate openEO_downScaleML
pip install -r test_requirements.txt
pytest tests/test_downscaleml_pipeline.py
```

---

## 📁 Project Structure

```text
.
├── Dockerfile
├── Makefile
├── environment.yml
├── test_requirements.txt
├── tests/
│   ├── test_downscaleml_pipeline.py   # open test pipeline
│   ├── pytest_A.py                    # closed test pipeline (requires AWS)
│   └── ...
└── app/
    └── test_data/                     # Output and intermediate results
```

---

## 🔍 Notes

* The `sin_cos_doy` and `raster2stac` operations are registered openEO processes from `openeo-processes-dask`.
* The `.zarr` and STAC metadata output are stored in `/app/test_data/`.
* `pytest_A.py` and any reference to SEAS5 require valid AWS credentials.

---

## 📝 License

Distributed under an open-source license aligned with interTwin project guidelines.

---

## 🤝 Acknowledgements

This work is part of the **[interTwin](https://intertwin.eu/)** project, and integrates components from the broader openEO ecosystem.

```
