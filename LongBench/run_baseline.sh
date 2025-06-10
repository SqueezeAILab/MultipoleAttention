#!/bin/bash
# This script loops over a set of percentile values
# and calls the specified command for each percentile.

MODEL="DeepSeek-R1-Distill-Qwen-14B"
MODEL="Qwen3-8B"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 pred.py -m $MODEL --port 6064 -n 8 -t 1
echo "All commands executed."
