#!/usr/bin/env bash

today=`date +%Y%m%d`
for function_type in matern52
do
    for objective_type in bogunovic
    do
        for sampler_type in ctv ctv_delta
        do
            nohup python syn.py --function-type ${function_type} --objective-type ${objective_type} --sampler-type ${sampler_type} --target-dir ./results/${today}/ &> ./results/${today}/output_log_${function_type}_${objective_type}_${sampler_type}.log &
        done
    done
done
