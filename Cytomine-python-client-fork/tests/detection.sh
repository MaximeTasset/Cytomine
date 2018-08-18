#!/bin/bash
#
# Submission script for NIC4
#SBATCH --job-name=test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96#24#48
#SBATCH --mem-per-cpu=1000#megabytes

srun python ~/spectral/Cytomine-python-client-fork/detection.py
