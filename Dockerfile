# Enable BuildKit syntax (must be the first line)
# syntax=docker/dockerfile:1.4

# Use micromamba base image
FROM mambaorg/micromamba:latest

USER root

# Set environment variables for micromamba
ENV MAMBA_DOCKERFILE_ACTIVATE=1 \
    MAMBA_ROOT_PREFIX=/opt/conda \
    PATH=/opt/conda/bin:$PATH

# Install git and openssh-client (required for SSH)
RUN apt-get update && apt-get install -y git openssh-client

# Create working directory and fix permissions
WORKDIR /app
RUN chown -R mambauser:mambauser /app

USER mambauser

# Copy environment file
COPY environment.yml .
COPY test_requirements.txt .

# Set up SSH for GitHub (critical before any git clone)
RUN mkdir -p ~/.ssh && \
    chmod 700 ~/.ssh && \
    ssh-keyscan github.com >> ~/.ssh/known_hosts && \
    chmod 600 ~/.ssh/known_hosts


RUN micromamba env create -f environment.yml && \
    micromamba clean --all --yes

ENV PATH=/opt/conda/envs/openEO_downScaleML/bin:$PATH

# Install downScaleML from the specified branch
RUN git clone https://github.com/interTwin-eu/downScaleML.git && \
    cd downScaleML && \
    git checkout openEO_downScaleML && \
    pip install .

#RUN pip install raster2stac
RUN pip install raster2stac

# Install pip packages
RUN pip install openeo-processes-dask
RUN pip install openeo-processes-dask[implementations]

# Clone openeo-processes-dask (private repo) with SSH
RUN git clone --recurse-submodules git@github.com:Open-EO/openeo-processes-dask.git && \
    cd openeo-processes-dask && \
    git checkout feature/merge_cubes_issue && \
    pip install .

RUN pip install -r test_requirements.txt

# Copy test files
COPY tests/ /app/tests/

# Default command
CMD ["bash"]
