<div align="center">

# BusterX

</div>

[[🚀🤗BusterX++]](https://huggingface.co/l8cv/BusterX-plusplus)
[[📦️🤗GenBuster-Bench++]](https://huggingface.co/datasets/l8cv/GenBuster-Bench-plusplus)
[[📦️🤗GenBuster-Bench]](https://huggingface.co/datasets/l8cv/GenBuster-Bench)
[[📦️🤗GenBuster-Unified]](https://huggingface.co/datasets/l8cv/GenBuster-Unified)
[[📦️🤗GenBuster-200K]](https://huggingface.co/datasets/l8cv/GenBuster-200K)
[[📦️🤗GenBuster-200K-mini]](https://huggingface.co/datasets/l8cv/GenBuster-200K-mini)
[[🔥📜BusterX++ paper]](https://www.alphaxiv.org/abs/2507.14632)
[[🔥📜BusterX paper]](https://www.alphaxiv.org/abs/2505.12620)

______________________________________________________________________

## Overview

BusterX is a family of MLLM-based methods for detecting and explaining AI-generated content. The project includes:

- **BusterX++**: Towards Unified Cross-Modal AI-Generated Content Detection and Explanation with MLLM
- **BusterX**: MLLM-Powered AI-Generated Video Forgery Detection and Explanation

> [!IMPORTANT]
> We release the **2026/06 Revised Edition** of BusterX and BusterX++. Base models have been updated to the latest Qwen3.5, and the datasets and benchmarks have been updated.

## Installation

We use [uv](https://docs.astral.sh/uv/) for environment management. The default CUDA version is 12.x; check [`pyproject.toml`](pyproject.toml) if you need a different CUDA version.

```bash
# pip install uv
uv sync
```

## Dataset & Benchmark Preparation

Log in to Hugging Face (make sure you have access to the gated datasets and models), then download and build all datasets:

```bash
make login
make fetch_data
make build_all_datasets
```

Available datasets and benchmarks:

| Name | Description |
|---|---|
| GenBuster-200K / 200K-mini | Large-scale video training data |
| GenBuster-Unified | Image + video training data |
| GenBuster-Bench | Video-only evaluation benchmark |
| GenBuster-Bench++ | Cross-modal (image + video) evaluation benchmark |

## Quick Start

> Default settings target 8×H100 80GB. Adjust hyperparameters as needed.

### Evaluate BusterX++ on GenBuster-Bench

```bash
bash scripts/eval_genbuster_bench.sh l8cv/BusterX-plusplus
```

For cross-modal evaluation on GenBuster-Bench++:

```bash
bash scripts/eval_genbuster_bench_plusplus.sh l8cv/BusterX-plusplus
```

### Train with DAPO

We recommend server rollout mode for stability and efficiency. Start the rollout server first:

```bash
bash scripts/rollout.sh
```

Then launch training:

```bash
bash scripts/train_dapo.sh
```

GSPO training is also supported:

```bash
bash scripts/train_gspo.sh
```

### Custom Benchmark Evaluation

```bash
make build_custom_benchmark
bash scripts/eval_custom_benchmark.sh <model>
```

## Citation

```bibtex
@article{wen2025busterxunifiedcrossmodalaigenerated,
    title={BusterX++: Towards Unified Cross-Modal AI-Generated Content Detection and Explanation with MLLM},
    author={Haiquan Wen and Tianxiao Li and Zhenglin Huang and Yiwei He and Guangliang Cheng},
    journal={Arxiv},
    year={2025},
}

@article{wen2025busterxmllmpoweredaigeneratedvideo,
    title={BusterX: MLLM-Powered AI-Generated Video Forgery Detection and Explanation},
    author={Haiquan Wen and Yiwei He and Zhenglin Huang and Tianxiao Li and Zihan Yu and Xingru Huang and Lu Qi and Baoyuan Wu and Xiangtai Li and Guangliang Cheng},
    journal={Arxiv},
    year={2025},
}
```

## Acknowledgement

This work is built upon [ms-swift](https://github.com/modelscope/ms-swift). Thanks for their excellent work!

## License

This project is licensed under the BSD 3-Clause License — see [LICENSE](LICENSE) for details.
