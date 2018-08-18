#!/bin/bash
#
# Submission script for NIC4
#SBATCH --job-name=test_m
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=90
#SBATCH --mem-per-cpu=2000#megabytes

srun python ~/spectral/Cytomine-python-client-fork/test_model.py
