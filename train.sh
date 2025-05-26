#!/bin/bash

. ./input_validation.sh
input_validation $@

repo_dir="$HOME/master-thesis/code/backdoorbench"
my_dir="/vol/csedu-nobackup/project/hberendsen"
data_dir="$my_dir/data"
record_dir="$my_dir/record"
timestamp=$(date +"T%d-%m_%H-%M")

# gpu=$(python get_gpu.py)

# if [[ ! $gpu =~ "RTX 2080 Ti" ]]; then
#     echo "Unexpected GPU: ${gpu}"
#     exit 1
# fi

pratio_label=$(echo p$pratio | tr . -)
attack_id="${attack}_${model}_${dataset}_${pratio_label}"

python train_backdoor.py --pr 0.1 --clean_data_path $data_dir/$dataset

# Handle Grond failure
if [[ $? -ne 0 ]]; then
    mv "$record_dir/$attack_id" "$record_dir/FAIL_${attack_id}_${timestamp}"
    echo "!!! GROND FAILURE !!!"
    exit 1
fi

echo "!!! FINISHED TRAINING !!!"

# cd $record_dir    
# tar -cf "${attack_id}_${timestamp}.tar" $attack_id
