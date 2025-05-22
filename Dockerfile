# Use micromamba base image
FROM mambaorg/micromamba:latest

# Set environment variables for micromamba
ENV MAMBA_DOCKERFILE_ACTIVATE=1 \
    MAMBA_ROOT_PREFIX=/opt/conda \
    PATH=/opt/conda/bin:$PATH

# Create environment from micromamba
RUN micromamba install -n zarr_downScaleML -y \
    python=3.10 \
    xarray \
    zarr \
    dask \
    numpy \
    pip \
    -c conda-forge && \
    micromamba clean --all --yes

# Activate environment and set it as default
SHELL ["micromamba", "run", "-n", "zarr_downScaleML", "/bin/bash", "-c"]

# Set working directory
WORKDIR /app

# Copy your code to the image
COPY . /app

# Install your Python package if needed
RUN pip install .

# Set default command
CMD ["python"]
