#!/bin/bash

# Configure your API Keys and URLs, Leave it blank if not necessary
# You can use environment variables here as well
# If you choose openai backend compatible backend, please fill SAMPLER_OPENAI_BASE_URL and SAMPLER_OPENAI_API_KEY with openai base url and api key of the backend
backend_type='openai' # can be 'openai', 'gemini' and 'anthropic'
SAMPLER_OPENAI_BASE_URL='http://127.0.0.1:4020/v1'
SAMPLER_OPENAI_API_KEY='EXAMPLE API KEY'
GEMINI_API_KEY='DUMMY'
ANTHROPIC_API_KEY='DUMMY'

# To evaluate symbolic subset, you should first launch an openai compatible backend.
# We use Qwen/Qwen2.5-7B-Instruct as our parser to extract the answer.
# Fill EVAL_OPENAI_BASE_URL and EVAL_OPENAI_API_KEY with openai base url and api key of the backend
EVAL_OPENAI_BASE_URL='http://127.0.0.1:4020/v1'
EVAL_OPENAI_API_KEY='DUMMY'

# Control sampling and evaluation (can be set from command line)
#run_sampling=false  # Set to "true" to run sampling, "false" to skip
#run_evaluation=true # Set to "true" to run evaluation, "false" to skip
run_sampling=true  # Set to "true" to run sampling, "false" to skip
run_evaluation=false # Set to "true" to run evaluation, "false" to skip
run_symbolic_evaluation=false # Set to "true" to ONLY run symbolic evaluation
run_realistic_evaluation=false # Set to "true" to ONLY run realistic evaluation

# Model and Dataset Configuration
model_name='Qwen3-8B'
dataset_base='gsm_infinite'
save_name='qwen-8b'

# Sampling Settings
num_samples=1
temperature_symbolic=1.0 # Temperature for symbolic
temperature_realistic=0.0 # Temperature for realistic
max_tokens=4096

# Batch size and example limit per op
batch_size=1
limit_symbolic=100 # Limit for symbolic
#limit_symbolic=25 # Limit for symbolic
limit_realistic=200 # Limit for realistic


# Lengths to process (can be numbers or strings like '8k')
lengths=(
#    "0"
    "8k"
    "16k"
#    "32k"
)

# Dataset suffixes
dataset_suffixes=(
    "symbolic"
#    "medium"
#    "hard"
)

# Operation Range Configuration (Per length and suffix). if empty, the subset will be skipped.
declare -A ops_config
# Example configurations:
# ops_config["0_symbolic"]='{"start": 1, "end": 1, "stride": 1}'
ops_config["8k_symbolic"]='{"start": 1, "end": 2, "stride": 1}'
ops_config["16k_symbolic"]='{"start": 1, "end": 2, "stride": 1}'
# ops_config["32k_symbolic"]='{"start": 1, "end": 3, "stride": 1}'

#ops_config["0_medium"]='{"start": 2, "end": 30, "stride": 1}'
#ops_config["8k_medium"]='{"start": 2, "end": 30, "stride": 1}'
#ops_config["16k_medium"]='{"start": 2, "end": 30, "stride": 1}'
#ops_config["32k_medium"]='{"start": 2, "end": 30, "stride": 1}'

#ops_config["0_hard"]='{"start": 2, "end": 30, "stride": 1}'
#ops_config["8k_hard"]='{"start": 2, "end": 30, "stride": 1}'
#ops_config["16k_hard"]='{"start": 2, "end": 30, "stride": 1}'
#ops_config["32k_hard"]='{"start": 2, "end": 30, "stride": 1}'


# Filter Configuration (JSON string, only used for realistic)
filter_config='[
    {"percentage": 0.4, "template": "crazy_zootopia", "mode": "normalforward"},
    {"percentage": 0.05, "template": "movie_festival_awards", "mode": "normalforward"},
    {"percentage": 0.05, "template": "teachers_in_school", "mode": "normalforward"},
    {"percentage": 0.4, "template": "crazy_zootopia", "mode": "forwardreverse"},
    {"percentage": 0.05, "template": "movie_festival_awards", "mode": "forwardreverse"},
    {"percentage": 0.05, "template": "teachers_in_school", "mode": "forwardreverse"}
]'
