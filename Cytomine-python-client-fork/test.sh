#!/bin/bash
#
# Submission script for NIC4
#SBATCH --job-name=time_read
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000#megabytes

srun python ~/spectral/Cytomine-python-client-fork/test.py
