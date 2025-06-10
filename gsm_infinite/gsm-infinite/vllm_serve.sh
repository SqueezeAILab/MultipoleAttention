export CUDA_VISIBLE_DEVICES=7

vllm serve Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 1 --gpu-memory-utilization 0.8 \
    --max_model_len 32768 --trust-remote-code --port 4020
