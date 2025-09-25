#!/bin/bash

. ./input_validation.sh
input_validation $@

my_dir="/home/xxu/back_stealthiness"

if [[ -z $my_dir ]]; then
    echo "Please set the 'my_dir' variable to the parent directory of the data and record folders"
    exit 1
fi

data_dir="/home/xxu/back_stealthiness/record/data"
record_dir="$my_dir/record"
timestamp=$(date +"T%d-%m_%H-%M")

# gpu=$(python get_gpu.py)

# if [[ ! $gpu =~ "RTX 2080 Ti" ]]; then
#     echo "Unexpected GPU: ${gpu}"
#     exit 1
# fi

pratio_label=$(echo p$pratio | tr . -)
model_lowercase=$(echo "$model" | awk '{print tolower($0)}')
attack_id="${attack}_${model_lowercase}_${dataset}_${pratio_label}"

function check_failure() {
    error_code=$1
    error_message=$2

    if [[ $error_code -ne 0 ]]; then
        mv "$record_dir/$attack_id" "$record_dir/FAIL_${attack_id}_${timestamp}"
        echo "!!! $error_message !!!"
        exit 1
    fi
}

# Smaller batch size for Imagenette, to decrease required RAM
if [[ $dataset == "imagenette" ]]; then
    bs=20
elif [[ $dataset == "tiny" ]]; then
    bs=256
else
    bs=100
fi
gpuid=0
target_cls=0
# Generate UPGD trigger
echo "!!! GENERATING TRIGGER !!!"
clean_model_path="$record_dir/prototype_${model_lowercase}_${dataset}_pNone/clean_model.pth"
python generate_upgd.py --arch $model --dataset $dataset \
                        --target_cls $target_cls \
                        --eps 16 \
                        --gpuid $gpuid \
                        --batch_size $bs --num_workers 6 \
                        --data_root $data_dir/$dataset --model_path $clean_model_path \
                        --upgd_path $record_dir/$attack_id

check_failure $? "FAILURE WHILE GENERATING TRIGGER"

# Train on data poisoned with above trigger
echo "!!! TRAINING BACKDOORED MODEL !!! "
python train_backdoor.py --arch $model --dataset $dataset --pr $pratio --epochs $n_epochs \
                         --target_cls $target_cls \
                         --gpuid $gpuid \
                         --batch_size $bs --num_workers 6 \
                         --clean_data_path $data_dir/$dataset --upgd_path $record_dir/$attack_id \
                         --out_dir $record_dir/$attack_id


check_failure $? "FAILURE WHILE TRAINING MODEL"

echo "!!! FINISHED TRAINING !!!"

cd $record_dir    
tar -cf "${attack_id}_${timestamp}.tar" $attack_id
