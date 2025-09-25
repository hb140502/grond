#!/bin/bash -e
#SBATCH --job-name grond
#SBATCH --partition=icis
#SBATCH --account=icis
#SBATCH --qos=icis-preempt
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=10:00:00
#SBATCH --output=./slurm_log/my-experiment-%j.out
#SBATCH --error=./slurm_log/my-experiment-%j.err
#SBATCH --mail-user=xiaoyun.xu@ru.nl
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodelist=cn115

source /scratch/xxu/pytorch/bin/activate
cd /home/xxu/back_stealthiness/backdoorbench

./train_cn.sh grond vit_small cifar10 0.05 200