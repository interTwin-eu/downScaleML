# Use micromamba base image
FROM mambaorg/micromamba:latest

USER root

# Set environment variables for micromamba
ENV MAMBA_DOCKERFILE_ACTIVATE=1 \
    MAMBA_ROOT_PREFIX=/opt/conda \
    PATH=/opt/conda/bin:$PATH

# Install git and openssh-client (required for SSH)
RUN apt-get update && \
    apt-get install -y git openssh-client libstdc++6
    
# Create working directory and fix permissions
WORKDIR /app

# Create test_data as mambauser directly
RUN mkdir -p /app/test_data

# Copy environment file
COPY environment.yml .
COPY test_requirements.txt .

RUN micromamba env create -f environment.yml && \
    micromamba clean --all --yes

# Set environment PATH and library path
ENV PATH=/opt/conda/envs/openEO_downScaleML/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/conda/envs/openEO_downScaleML/lib:$LD_LIBRARY_PATH

# Install downScaleML from the specified branch
RUN git clone https://github.com/interTwin-eu/downScaleML.git && \
    cd downScaleML && \
    git checkout openEO_downScaleML && \
    pip install .

# Install raster2stac from the test_cube branch
RUN git clone https://gitlab.inf.unibz.it/earth_observation_public/raster-to-stac.git && \
    cd raster-to-stac && \
    git checkout chunk_auto_bug && \
    pip install .

RUN git clone https://github.com/interTwin-eu/openeo-processes-dask.git && \
    cd openeo-processes-dask && \
    git checkout process/r2s && \
    git submodule set-url openeo_processes_dask/specs/openeo-processes https://github.com/interTwin-eu/openeo-processes.git && \
    git submodule update --init && \
    cd openeo_processes_dask/specs/openeo-processes && \
    git checkout downScaleML_processes && \
    cd ../../.. && \
    pip install .[implementations]

RUN pip install -r test_requirements.txt

RUN pip install s3fs

# Copy test files
COPY tests/ /app/tests/

RUN sed -i -e '/if X_feature_names is None and fitted_feature_names is not None:/,/^[[:space:]]*return/ { /if X_feature_names is None and fitted_feature_names is not None:/b; /^[[:space:]]*return/b; s/^/#/; }' /opt/conda/envs/openEO_downScaleML/lib/python3.11/site-packages/sklearn/utils/validation.py

# Default command
CMD ["bash"]