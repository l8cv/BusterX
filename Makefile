test:
	uv run --no-sync pytest --cov=busterx --cov-report=xml --cov-report=html

lint:
	uv run --no-sync pre-commit install --install-hooks 2>/dev/null || true
	uv run --no-sync pre-commit run --all-files

build_wheel:
	uv build --wheel

login:
	uv run --no-sync hf auth login
	uv run --no-sync wandb login

fetch_data:
	uv run --no-sync hf download --type dataset l8cv/GenBuster-200K-mini --local-dir ./data/genbuster_200k
	uv run --no-sync hf download --type dataset l8cv/GenBuster-Unified --local-dir ./data/genbuster_unified
	uv run --no-sync hf download --type dataset l8cv/GenBuster-Bench --local-dir ./data/genbuster_bench
	uv run --no-sync hf download --type dataset l8cv/GenBuster-Bench-plusplus --local-dir ./data/genbuster_bench_plusplus
	echo "auto unzip downloaded datasets..."; \
	for file in ./data/genbuster_200k/*.zip; do unzip -o "$$file" -d ./data/genbuster_200k/; done; \
	for file in ./data/genbuster_unified/*.zip; do unzip -o "$$file" -d ./data/genbuster_unified/; done; \
	for file in ./data/genbuster_bench/*.zip; do unzip -o "$$file" -d ./data/genbuster_bench/; done; \
	for file in ./data/genbuster_bench_plusplus/*.zip; do unzip -o "$$file" -d ./data/genbuster_bench_plusplus/; done; \

build_genbuster_200k:
	uv run --no-sync busterx build_dataset \
	--dataset_pp genbuster_200k \
	--dataset_dir ./data/genbuster_200k/train \
	--output ./genbuster_200k.jsonl \
	--prompt ./prompts/default_user_prompt.txt \
	--input_mode image \
	--sample_fps 2.0 \
	--max_workers 44 \
	--decord_timeout 120 \
	--jpeg_quality 95

build_genbuster_200k_video_mode:
	uv run --no-sync busterx build_dataset \
	--dataset_pp genbuster_200k \
	--dataset_dir ./data/genbuster_200k/train \
	--output ./genbuster_200k_v.jsonl \
	--prompt ./prompts/default_user_prompt.txt \
	--input_mode video

build_genbuster_bench:
	uv run --no-sync busterx build_dataset \
	--dataset_pp genbuster_bench \
	--dataset_dir ./data/genbuster_bench/GenBuster-Bench \
	--output ./genbuster_bench.jsonl \
	--prompt prompts/default_user_prompt.txt \
	--input_mode image \
	--sample_fps 2.0 \
	--max_workers 44 \
	--decord_timeout 120 \
	--jpeg_quality 95

build_genbuster_unified_video:
	uv run --no-sync busterx build_dataset \
	--dataset_pp genbuster_unified_video \
	--dataset_dir ./data/genbuster_unified/video \
	--output ./genbuster_unified_video.jsonl \
	--prompt ./prompts/default_user_prompt.txt \
	--input_mode image \
	--sample_fps 2.0 \
	--max_workers 44 \
	--decord_timeout 120 \
	--jpeg_quality 95

build_genbuster_unified_image:
	uv run --no-sync busterx build_dataset \
	--dataset_pp genbuster_unified_image \
	--dataset_dir ./data/genbuster_unified/image \
	--output ./genbuster_unified_image.jsonl \
	--prompt ./prompts/default_user_prompt_image.txt

build_genbuster_bench_plusplus_video:
	uv run --no-sync busterx build_dataset \
	--dataset_pp genbuster_bench_plusplus_video \
	--dataset_dir ./data/genbuster_bench_plusplus/video \
	--output ./genbuster_bench_plusplus_video.jsonl \
	--prompt ./prompts/default_user_prompt.txt \
	--input_mode image \
	--sample_fps 2.0 \
	--max_workers 44 \
	--decord_timeout 120 \
	--jpeg_quality 95

build_genbuster_bench_plusplus_image:
	uv run --no-sync busterx build_dataset \
	--dataset_pp genbuster_bench_plusplus_image \
	--dataset_dir ./data/genbuster_bench_plusplus/image \
	--output ./genbuster_bench_plusplus_image.jsonl \
	--prompt ./prompts/default_user_prompt_image.txt

build_all_datasets:
	echo "=== Building all datasets... ==="
# 	$(MAKE) build_genbuster_200k
# 	$(MAKE) build_genbuster_200k_video_mode
	$(MAKE) build_genbuster_bench
	$(MAKE) build_genbuster_unified_video
	$(MAKE) build_genbuster_unified_image
	$(MAKE) build_genbuster_bench_plusplus_video
	$(MAKE) build_genbuster_bench_plusplus_image

build_custom_benchmark:
	uv run --no-sync busterx build_dataset \
	--dataset_pp custom_benchmark \
	--dataset_dir ./custom_benchmark \
	--output ./custom_benchmark.jsonl \
	--prompt ./prompts/default_user_prompt.txt \
	--input_mode image \
	--sample_fps 2.0 \
	--max_workers 44 \
	--decord_timeout 120 \
	--jpeg_quality 95
