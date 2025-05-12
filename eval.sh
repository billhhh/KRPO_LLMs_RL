#!/usr/bin/env bash

time=$(date "+%Y%m%d-%H%M%S")
name=eval_krpo
CUDA_VISIBLE_DEVICES=$1 python -u eval.py --save_name ${name} > logs/${time}_train_${name}.log 2>&1 &
