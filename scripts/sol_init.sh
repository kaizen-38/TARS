#!/usr/bin/env bash
# Sourced by --wrap jobs to set up the conda environment.
module purge
module load mamba/latest
eval "$(conda shell.bash hook)"
conda activate tars
