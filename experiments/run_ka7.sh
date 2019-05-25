#!/usr/bin/env bash

today=`date +%Y%m%d`

mkdir -p ./results/${today} 2>/dev/null

for function_type in rbf matern52
do
    for objective_type in tv_and_type_dep
    do
        for sampler_type in normal dtv
        do
            nohup python syn.py --function-type ${function_type} --objective-type ${objective_type} --sampler-type ${sampler_type} --target-dir ./results/${today}/ &> ./results/${today}/output_log_${function_type}_${objective_type}_${sampler_type}.log &
        done
    done
done