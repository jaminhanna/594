#!/bin/sh
#SBATCH --account=ACF-UTK0011
#SBATCH --partition=campus
#SBATCH --qos=campus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
#SBATCH --output=train.out
#SBATCH --error=train.err

module load matlab/r2025b-academic
module load gurobi/13.0.2

matlab -batch train
