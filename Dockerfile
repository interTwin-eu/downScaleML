# Use micromamba base image
FROM mambaorg/micromamba:latest

USER root

# Set environment variables for micromamba
ENV MAMBA_DOCKERFILE_ACTIVATE=1 \
    MAMBA_ROOT_PREFIX=/opt/conda \
    PATH=/opt/conda/bin:$PATH

# Install git
RUN apt-get update && apt-get install -y git

# Create working directory and fix permissions
WORKDIR /app
RUN chown -R mambauser:mambauser /app

USER mambauser

# Copy environment file
COPY environment.yml .

RUN micromamba env create -f environment.yml && \
    micromamba clean --all --yes

ENV PATH=/opt/conda/envs/openEO_downScaleML/bin:$PATH

# Install pip packages
RUN pip install openeo-processes-dask
RUN pip install openeo-processes-dask[implementations]

# Clone and install openeo-processes-dask from custom branch
RUN git clone https://github.com/interTwin-eu/openeo-processes-dask.git && \
    cd openeo-processes-dask && \
    git checkout feature/merge_cubes_issue && \
    pip install .

RUN pip install raster2stac

# Install downScaleML from the specified branch
RUN git clone https://github.com/interTwin-eu/downScaleML.git && \
    cd downScaleML && \
    git checkout openEO_downScaleML && \
    pip install .

# Default command
CMD ["bash"]
