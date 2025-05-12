#!/usr/bin/env bash

time=$(date "+%Y%m%d-%H%M%S")
name=train_krpo
CUDA_VISIBLE_DEVICES=$1 python -u train.py --save_name ${name} > logs/${time}_train_${name}.log 2>&1 &
