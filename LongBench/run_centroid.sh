#!/bin/bash

MODEL="DeepSeek-R1-Distill-Qwen-14B"
# MODEL="Qwen3-8B"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Define an array of percentile values
percentiles=(128)

# Loop over each percentile in the array
for PERCENTILE in "${percentiles[@]}"; do
  echo "Running command with percentile ${PERCENTILE}..."

  python3 pred.py -m ${MODEL} --port 6064 --use_centroids --percentiles_lst "${PERCENTILE}" --percent_clusters_lst 6.25  -n 8 -t 1 --cluster_interval 128
  python3 pred.py -m ${MODEL} --port 6064 --use_centroids --percentiles_lst "${PERCENTILE}" --percent_clusters_lst 6.25  -n 8 -t 1 --cluster_interval 128 --use_replacement

  # Optionally, you can check the exit status and break the loop if an error occurs
  if [ $? -ne 0 ]; then
    echo "Command failed with percentile ${PERCENTILE}. Exiting."
    exit 1
  fi
done
echo "All commands executed."
