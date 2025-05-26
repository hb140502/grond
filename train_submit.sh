#!/bin/bash

. ./input_validation.sh
input_validation $@

if [ $attack == "bpp" ]; then
    qos="csedu-normal"
    time_limit="6:00:00" 
else
    qos="csedu-small"
    time_limit="4:00:00" 
fi

job_save="jobs/%j_${attack}_${model}_${dataset}"

sbatch --time $time_limit -q $qos --output "${job_save}.out" --error "${job_save}.err" train_batch.sh $@