#!/usr/bin/env bash
# Copyright 2021-2024 Rene Widera
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

# This script is executing the expression given as parameters and forwards signals to the application.
# Signals will NOT only be forwarded, they will be mapped to use SLURM signals in a useful way.
#
# You need to source this script with your application as argument:
#   source handleSlurmSignals.sh foo.exe --foArg1="alice" --foArg2="bar"
#
# Signal mapping
#
# SIGTERM -> SIGUSR2
# SIGCONT -> SIGUSR1
# SIGUSR1 -> SIGUSR1
# SIGUSR2 -> SIGUSR2
# SIGALRM -> SIGUSR1 and SIGUSR2
#

HANDLE_SLURM_SIGNALS_PATH=$1
shift

export CUDA_VISIBLE_DEVICES=$((SLURM_GPUS_PER_NODE-1-SLURM_LOCALID))
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES on SLURM_LOCALID: $SLURM_LOCALID"

source "$HANDLE_SLURM_SIGNALS_PATH" "$@"
