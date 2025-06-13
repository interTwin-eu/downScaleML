# Use micromamba base image
FROM mambaorg/micromamba:latest

USER root

# Set environment variables for micromamba
ENV MAMBA_DOCKERFILE_ACTIVATE=1 \
    MAMBA_ROOT_PREFIX=/opt/conda \
    PATH=/opt/conda/bin:$PATH

# Install git and openssh-client (required for SSH)
RUN apt-get update && apt-get install -y git

# Create working directory and fix permissions
WORKDIR /app
RUN chown -R mambauser:mambauser /app

USER mambauser

# Copy environment file
COPY environment.yml .
COPY test_requirements.txt .

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

ARG GITHUB_TOKEN
RUN git clone https://${GITHUB_TOKEN}@github.com/interTwin-eu/openeo-processes-dask.git && \
    cd openeo-processes-dask && \
    git checkout feature/merge_cubes_issue && \
    # Force Git to use PAT for submodules (now based on the branch's .gitmodules)
    git config --global url."https://${GITHUB_TOKEN}@github.com".insteadOf "https://github.com" && \
    git submodule update --init --recursive && \
    pip install .[implementations]

RUN pip install -r test_requirements.txt

RUN pip install s3fs

# Copy test files
COPY tests/ /app/tests/

# Default command
CMD ["bash"]
