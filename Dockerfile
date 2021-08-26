FROM python:3.9-slim-buster

LABEL Tyler Sutterley "tsutterl@uw.edu"

ENV DEBIAN_FRONTEND="noninteractive" TZ="America/Los_Angeles"

RUN useradd --create-home --shell /bin/bash smb

RUN apt-get update -y && \
    apt-get install -y \
        gdal-bin \
        libgdal-dev \
        libproj-dev \
        proj-data \
        proj-bin \
        libgeos-dev \
        libhdf5-dev \
        libnetcdf-dev \
        git && \
    apt-get clean

WORKDIR /home/smb

ENV MPICC=mpicc
ENV CC=mpicc
ENV HDF5_MPI="ON"

RUN pip3 install --no-cache-dir --no-binary=h5py,cartopy \
        setuptools_scm \
        mpi4py \
        numpy \
        scipy \
        lxml \
        pyproj \
        python-dateutil \
        pyyaml \
        pandas \
        scikit-learn \
        matplotlib \
        gdal \
        netCDF4 \
        zarr \
        h5py \
        cartopy && \
    pip3 install --no-cache-dir --no-deps git+https://github.com/smithb/pointCollection.git

COPY . .

RUN --mount=source=.git,target=.git,type=bind \
    pip install --no-cache-dir --no-deps .

USER smb

CMD ["bash"]