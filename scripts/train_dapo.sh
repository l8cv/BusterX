#!/usr/bin/env bash
set -euo pipefail

TRAIN_GPU_NUMS=${TRAIN_GPU_NUMS:-6}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3,4,5,6,7}

BASE_MODEL=${BASE_MODEL:-${1:-Qwen/Qwen3.5-4B}}
VLLM_MAX_COMPLETION_LEN=${VLLM_MAX_COMPLETION_LEN:-4096}
SOFT_CACHE_LENGTH=${SOFT_CACHE_LENGTH:-256}
LEARNING_RATE=${LEARNING_RATE:-1e-6}

PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-16}
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-768}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}

# qwen-vl-utils, for Qwen3.5, patch_size = 16, spatial_merge_size = 2, 256 * 256 pixels = (8 * (2*16)) * (8 * (2*16)) pixels = 8 * 8 tokens

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
NPROC_PER_NODE=${TRAIN_GPU_NUMS} \
IMAGE_MIN_TOKEN_NUM=64 \
IMAGE_MAX_TOKEN_NUM=64 \
VIDEO_MIN_TOKEN_NUM=64 \
VIDEO_MAX_TOKEN_NUM=64 \
FPS=2.0 \
WANDB_PROJECT=BusterXpp \
uv run --no-sync swift rlhf \
	--rlhf_type grpo \
	--model ${BASE_MODEL} \
	--tuner_type full \
	--torch_dtype bfloat16 \
	--dataset ./genbuster_unified_image.jsonl ./genbuster_unified_video.jsonl \
	--reward_funcs external_format external_accuracy external_format_based_accuracy soft_overlong \
	--reward_weights 1 1 0 1 \
	--external_plugins plugins/reward_func.py \
	--use_vllm true \
	--vllm_mode server \
	--vllm_server_host 127.0.0.1 \
	--vllm_server_port 8000 \
	--max_completion_length ${VLLM_MAX_COMPLETION_LEN} \
	--loss_type dapo \
	--beta 0.001 \
	--epsilon 0.2 \
	--epsilon_high 0.28 \
	--soft_cache_length ${SOFT_CACHE_LENGTH} \
	--dynamic_sample False \
	--max_resample_times 1 \
	--num_iterations 1 \
	--num_train_epochs 1 \
	--per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
	--gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
	--overlong_filter True \
	--generation_batch_size ${GENERATION_BATCH_SIZE} \
	--num_generations ${NUM_GENERATIONS} \
	--temperature 1 \
	--learning_rate ${LEARNING_RATE} \
	--warmup_ratio 0.05 \
	--lr_scheduler_type constant_with_warmup \
	--save_steps 32 \
	--save_total_limit 50 \
	--logging_steps 1 \
	--output_dir ${BASE_MODEL}_external_dapo \
	--dataloader_num_workers 8 \
	--dataset_num_proc 8 \
	--log_completions true \
	--log_entropy true \
	--rollout_importance_sampling_mode token_mask \
	--deepspeed zero2 \
	--report_to wandb \
	--use_hf true
