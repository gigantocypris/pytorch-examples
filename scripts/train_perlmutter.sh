#!/bin/bash
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J pytorch       # job name
#SBATCH -L SCRATCH       # job requires SCRATCH files
#SBATCH -A m2859_g       # allocation
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH -t 00:36:00
#SBATCH -J train-pm
#SBATCH -o logs/%x-%j.out

# Setup software
module load pytorch/1.13.1
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

echo "jobstart $(date)";pwd

# Run the training
srun -l -u python train.py -d nccl --rank-gpu --ranks-per-node=${SLURM_NTASKS_PER_NODE} $@

echo "jobend $(date)";pwd