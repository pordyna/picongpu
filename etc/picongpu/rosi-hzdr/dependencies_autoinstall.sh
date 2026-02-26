#!/usr/bin/env bash
# Copyright 2023-2026 Axel Huebl, Marco Garten, Klaus Steiniger, Pawel Ordyna
#
# This file is part of PIConGPU.
#
# PIConGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PIConGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.
#
# last updated: 2026-02-26


# get PIConGPU profile
if [ ! -f "$PIC_PROFILE" ]; then
    printf "Source a profile!\n"
    exit 1
else
    source "$PIC_PROFILE"
fi

set -euf -o pipefail
# create temporary directory for software source files
TMP_DIR="/bigdata/hplsim/production/picongpu-deps/tmp"
mkdir -p "${TMP_DIR}"
SOURCE_DIR=$(mktemp -d -p "${TMP_DIR}")


#   c-blosc
if [ ! -d "$BLOSC_ROOT" ]; then
    cd "$SOURCE_DIR"
    git clone -b v${BLOSC_VERSION} https://github.com/Blosc/c-blosc2.git \
        "$SOURCE_DIR"/c-blosc
    mkdir c-blosc-build
    cd c-blosc-build
    cmake -DCMAKE_INSTALL_PREFIX="$BLOSC_ROOT" \
        "$SOURCE_DIR"/c-blosc
    make -j 16 install
fi

module load cmake/3.26.1
#   PNGwriter
if [ ! -d "$PNGwriter_ROOT" ]; then
    cd "$SOURCE_DIR"
    git clone -b ${PNGWRITER_VERSION} https://github.com/pngwriter/pngwriter.git \
        "$SOURCE_DIR"/pngwriter
    mkdir pngwriter-build
    cd pngwriter-build
    cmake -DCMAKE_INSTALL_PREFIX="$PNGwriter_ROOT" \
        "$SOURCE_DIR"/pngwriter
    make -j 16 install
fi
module load cmake/4.0.3

#   HDF5
if [ ! -d "$HDF5_ROOT" ]; then
    cd $SOURCE_DIR
    curl -Lo hdf5-${HDF5_VERSION}.tar.gz \
        https://support.hdfgroup.org/releases/hdf5/v${HDF5_VERSION_MAJOR}_${HDF5_VERSION_MINOR}/v${HDF5_VERSION_MAJOR}_${HDF5_VERSION_MINOR}_${HDF5_VERSION_PATCH}/downloads/hdf5-${HDF5_VERSION_MAJOR}.${HDF5_VERSION_MINOR}.${HDF5_VERSION_PATCH}.tar.gz
    tar -xzf hdf5-${HDF5_VERSION}.tar.gz
    cd hdf5-${HDF5_VERSION}
    ./configure --enable-parallel --enable-shared --prefix $HDF5_ROOT CC=$(which mpicc) CXX=$(which mpiCC)
    make -j 16 && make install
fi

#   ADIOS2
# force usage of MPI and HDF5 and point directly to MPI headers and libraries
if [ ! -d "$ADIOS2_ROOT" ]; then
    cd "$SOURCE_DIR"
    git clone -b v${ADIOS2_VERSION} https://github.com/ornladios/ADIOS2.git \
        "$SOURCE_DIR"/adios2
        cd "$SOURCE_DIR"/adios2
        sed -i 's|if (ADIOS2_HAVE_MPI_CLIENT_SERVER)|if (TRUE)|' cmake/DetectOptions.cmake
    mkdir "$SOURCE_DIR"/adios2-build
    cd "$SOURCE_DIR"/adios2-build
    cmake "$SOURCE_DIR"/adios2 -DADIOS2_BUILD_EXAMPLES=OFF \
        -DCMAKE_INSTALL_PREFIX="$ADIOS2_ROOT" -DADIOS2_USE_Fortran=OFF \
        -DMPI_CXX_COMPILER=$(which mpiCC) -DMPI_C_COMPILER=$(which mpicc) \
        -DADIOS2_USE_MPI=ON -DADIOS2_USE_HDF5=ON
    make -j 16 && make install
fi

#   openPMD-api
if [ ! -d "OPENPMD_ROOT" ]; then
    cd $SOURCE_DIR
    git clone -b ${OPENPMD_VERSION} https://github.com/openPMD/openPMD-api.git \
        $SOURCE_DIR/openpmd-api
    mkdir $SOURCE_DIR/openpmd-api-build
    cd $SOURCE_DIR/openpmd-api-build
    cmake $SOURCE_DIR/openpmd-api \
                -DopenPMD_USE_HDF5=ON  -DopenPMD_USE_ADIOS2=ON \
        -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF \
        -DMPI_CXX_COMPILER=$(which mpiCC) -DMPI_C_COMPILER=$(which mpicc) \
        -DCMAKE_INSTALL_PREFIX="$OPENPMD_ROOT"
    make -j 16 install
fi


# message to user
echo ''
echo 'edit user & email within picongpu.profile, e.g. via:'
echo '    vim $PIC_PROFILE'
echo 'delete temporary folder for library compilation'
printf "    rm -rf %s\n" $SOURCE_DIR
