#!/bin/bash
#SBATCH -A cseduproject
#SBATCH -p csedu-prio,csedu
#SBATCH --qos=csedu-normal
#SBATCH -c 2
#SBATCH --mem 5G
#SBATCH --gres=gpu:rtx_2080_ti:1 
#SBATCH --time=2:30:00

. ./input_validation.sh
input_validation $@

./train.sh $attack $model $dataset $pratio $n_epochs
