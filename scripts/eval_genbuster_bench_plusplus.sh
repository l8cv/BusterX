#!/usr/bin/env sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <model>" >&2
    exit 2
fi

GPU_NUMS=${GPU_NUMS:-8}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
MODEL=$1

TIME_STR=${TIME_STR:-$(date +%Y%m%d_%H%M%S)}
VIDEO_RESULT=${VIDEO_RESULT:-./result/infer_genbuster_bench_plusplus/${TIME_STR}_video.jsonl}
IMAGE_RESULT=${IMAGE_RESULT:-./result/infer_genbuster_bench_plusplus/${TIME_STR}_image.jsonl}

echo "=== Inference GenBuster Bench Plus Plus Video Part... Model: ${MODEL} Time: ${TIME_STR} ==="
IMAGE_MIN_TOKEN_NUM=64 \
IMAGE_MAX_TOKEN_NUM=64 \
NPROC_PER_NODE=${GPU_NUMS} \
uv run --no-sync swift infer \
	--model ${MODEL} \
	--use_hf True \
	--remove_unused_columns False \
	--infer_backend vllm \
	--val_dataset ./genbuster_bench_plusplus_video.jsonl \
	--temperature 0 \
	--vllm_gpu_memory_utilization 0.80 \
	--vllm_tensor_parallel_size 1 \
	--vllm_max_model_len ${MAX_MODEL_LEN} \
	--vllm_enable_prefix_caching False \
	--max_new_tokens ${MAX_NEW_TOKENS} \
	--write_batch_size 2000 \
	--result_path ${VIDEO_RESULT}

# sleep 2s
sleep 2s

echo "=== Inference GenBuster Bench Plus Plus Image Part... Model: ${MODEL} Time: ${TIME_STR} ==="
IMAGE_MIN_TOKEN_NUM=64 \
IMAGE_MAX_TOKEN_NUM=64 \
NPROC_PER_NODE=${GPU_NUMS} \
uv run --no-sync swift infer \
	--model ${MODEL} \
	--use_hf True \
	--remove_unused_columns False \
	--infer_backend vllm \
	--val_dataset ./genbuster_bench_plusplus_image.jsonl \
	--temperature 0 \
	--vllm_gpu_memory_utilization 0.80 \
	--vllm_tensor_parallel_size 1 \
	--vllm_max_model_len ${MAX_MODEL_LEN} \
	--vllm_enable_prefix_caching False \
	--max_new_tokens ${MAX_NEW_TOKENS} \
	--write_batch_size 2000 \
	--result_path ${IMAGE_RESULT}

echo "=== Eval GenBuster Bench Plus Plus Video Part... Model: ${MODEL} Time: ${TIME_STR} ==="
uv run --no-sync busterx eval \
	--dataset_pp genbuster_bench_plusplus_video \
	--result_file ${VIDEO_RESULT}

echo "=== Eval GenBuster Bench Plus Plus Image Part... Model: ${MODEL} Time: ${TIME_STR} ==="
uv run --no-sync busterx eval \
	--dataset_pp genbuster_bench_plusplus_image \
	--result_file ${IMAGE_RESULT}
