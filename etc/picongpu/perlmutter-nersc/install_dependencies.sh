#!/usr/bin/env bash
#
# Authors: Axel Huebl, Marco Garten, Klaus Steiniger, Pawel Ordyna
#
# last updated: 2023-05-04

PROJECT=$proj
echo $PROJECT

source "$PIC_PROFILE"

set -euf -o pipefail
# create temporary directory for software source files
export SOURCE_DIR="$CFS/$PROJECT/$USER/lib_run_tmp"
mkdir -p $SOURCE_DIR
# Boost
if [ ! -d "$BOOST_ROOT" ]; then
    cd $SOURCE_DIR
    curl -L -s -o boost_1_82_0.tar.gz \
        https://boostorg.jfrog.io/artifactory/main/release/1.82.0/source/boost_1_82_0.tar.gz
    tar -xzf boost_1_82_0.tar.gz
    cd boost_1_82_0/
    ./bootstrap.sh --with-libraries=atomic,chrono,context,date_time,fiber,filesystem,math,program_options,serialization,system,thread --prefix=$BOOST_ROOT \
        CC=$(which cc) CXX=$(which CC)
    ./b2 cxxflags="-std=c++17" -j 10 && ./b2 install
fi

#   c-blosc
if [ ! -d "$BLOSC_ROOT" ]; then
    cd $SOURCE_DIR
    git clone -b v2.8.0 https://github.com/Blosc/c-blosc2.git \
        $SOURCE_DIR/c-blosc
    mkdir c-blosc-build
    cd c-blosc-build
    cmake -DCMAKE_INSTALL_PREFIX=$BLOSC_ROOT \
        -DMPI_C_COMPILER=cc -DMPI_CXX_COMPILER=CC \
        $SOURCE_DIR/c-blosc
    make -j 10 install
fi

#   PNGwriter
if [ ! -d "$PNGwriter_ROOT" ]; then
    cd $SOURCE_DIR
    git clone -b 0.7.0 https://github.com/pngwriter/pngwriter.git \
        $SOURCE_DIR/pngwriter
    mkdir pngwriter-build
    cd pngwriter-build
    cmake -DCMAKE_INSTALL_PREFIX=$PNGwriter_ROOT \
        $SOURCE_DIR/pngwriter
    make -j 10 install
fi

#   HDF5
if [ ! -d "$HDF5_ROOT" ]; then
    cd $SOURCE_DIR
    curl -Lo hdf5-1.14.0.tar.gz \
        https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.14/hdf5-1.14.0/src/hdf5-1.14.0.tar.gz
    tar -xzf hdf5-1.14.0.tar.gz
    cd hdf5-1.14.0
    ./configure --enable-parallel --enable-shared --prefix $HDF5_ROOT CC=$(which cc) CXX=$(which CC)
    make -j 10 && make install
fi

#   ADIOS2
# force usage of MPI and HDF5 and point directly to MPI headers and libraries
if [ ! -d "$ADIOS2_ROOT" ]; then
    cd $SOURCE_DIR
    git clone -b v2.9.0 https://github.com/ornladios/ADIOS2.git \
        $SOURCE_DIR/adios2
    mkdir $SOURCE_DIR/adios2-build
    cd $SOURCE_DIR/adios2-build
    cmake $SOURCE_DIR/adios2 -DADIOS2_BUILD_EXAMPLES=OFF \
        -DCMAKE_INSTALL_PREFIX=$ADIOS2_ROOT -DADIOS2_USE_Fortran=OFF \
        -DADIOS2_USE_BZip2=OFF \
        -DADIOS2_USE_MPI=ON -DADIOS2_USE_HDF5=ON \
        -DMPI_CXX_COMPILER=$(which CC) -DMPI_C_COMPILER=$(which cc) \
        -DMPI_CXX_HEADER_DIR=${MPICH_DIR}/include \
        -DMPI_C_HEADER_DIR=${MPICH_DIR}/include \
        -DMPI_mpi_gnu_91_LIBRARY=${MPICH_DIR}/lib/libmpi_gnu_91.so
    make -j 10 && make install
fi

#   openPMD-api
if [ ! -d "OPENPMD_ROOT" ]; then
    cd $SOURCE_DIR
    git clone -b 0.15.1 https://github.com/openPMD/openPMD-api.git \
        $SOURCE_DIR/openpmd-api
    mkdir $SOURCE_DIR/openpmd-api-build
    cd $SOURCE_DIR/openpmd-api-build
    cmake $SOURCE_DIR/openpmd-api \
        -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF \
        -DMPI_CXX_COMPILER=$(which CC) -DMPI_C_COMPILER=$(which cc) \
        -DMPI_CXX_HEADER_DIR=${MPICH_DIR}/include \
        -DMPI_C_HEADER_DIR=${MPICH_DIR}/include \
        -DMPI_mpi_gnu_91_LIBRARY=${MPICH_DIR}/lib/libmpi_gnu_91.so \
        -DCMAKE_INSTALL_PREFIX="$OPENPMD_ROOT"
    make -j 10 install
fi

# message to user
echo ''
echo 'edit user & email within picongpu.profile, e.g. via:'
echo '    vim $PIC_PROFILE'
echo 'delete temporary folder for library compilation'
printf "    rm -rf %s\n" $SOURCE_DIR
