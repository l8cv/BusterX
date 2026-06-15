#!/usr/bin/env sh
set -eu

# qwen-vl-utils, for Qwen3.5, patch_size = 16, spatial_merge_size = 2, 256 * 256 pixels = (8 * (2*16)) * (8 * (2*16)) pixels = 8 * 8 tokens

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <model>" >&2
    exit 2
fi

GPU_NUMS=${GPU_NUMS:-8}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
MODEL=$1

TIME_STR=${TIME_STR:-$(date +%Y%m%d_%H%M%S)}
VIDEO_RESULT=${VIDEO_RESULT:-./result/infer_genbuster_bench/${TIME_STR}.jsonl}

echo "=== Inference GenBuster Bench... Model: ${MODEL} Time: ${TIME_STR} ==="

IMAGE_MIN_TOKEN_NUM=64 \
IMAGE_MAX_TOKEN_NUM=64 \
NPROC_PER_NODE=${GPU_NUMS} \
uv run --no-sync swift infer \
	--model ${MODEL} \
	--use_hf True \
	--remove_unused_columns False \
	--infer_backend vllm \
	--val_dataset ./genbuster_bench.jsonl \
	--temperature 0 \
	--vllm_gpu_memory_utilization 0.80 \
	--vllm_tensor_parallel_size 1 \
	--vllm_max_model_len ${MAX_MODEL_LEN} \
	--vllm_enable_prefix_caching False \
	--max_new_tokens ${MAX_NEW_TOKENS} \
	--write_batch_size 3200 \
	--result_path ${VIDEO_RESULT}

echo "=== Eval GenBuster Bench... Model: ${MODEL} Time: ${TIME_STR} ==="
uv run --no-sync busterx eval \
	--dataset_pp genbuster_bench \
	--result_file ${VIDEO_RESULT}
