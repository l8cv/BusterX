#!/usr/bin/env bash
set -euo pipefail

ROLLOUT_GPU_NUMS=${ROLLOUT_GPU_NUMS:-2}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

BASE_MODEL=${BASE_MODEL:-${1:-Qwen/Qwen3.5-4B}}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
IMAGE_MIN_TOKEN_NUM=64 \
IMAGE_MAX_TOKEN_NUM=64 \
VIDEO_MIN_TOKEN_NUM=64 \
VIDEO_MAX_TOKEN_NUM=64 \
FPS=2.0 \
uv run --no-sync swift rollout \
	--model ${BASE_MODEL} \
	--vllm_max_model_len ${MAX_MODEL_LEN} \
	--vllm_gpu_memory_utilization 0.80 \
	--vllm_max_num_seqs 32 \
	--vllm_tensor_parallel_size 1 \
	--vllm_data_parallel_size ${ROLLOUT_GPU_NUMS} \
	--vllm_enable_prefix_caching False \
	--use_hf true
