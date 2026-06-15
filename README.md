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

We use uv to manage the environment, the default CUDA version is 12.x, you may need to specify another version -- check [pyproject.toml](pyproject.toml) for details.

```bash
# pip install uv
uv sync
```

## Prepare Datasets and Benchmarks

```bash
# login hf, please make sure you have the proper permissions to access the datasets and models
make login
make fetch_data
make build_all_datasets
```

## Quick Start

The default settings is for 8xH100 80GB, you may need to adjust some hyperparameters.

### Evaluate BusterX++ on GenBuster-Bench:

```bash
bash scripts/eval_genbuster_bench.sh l8cv/BusterX-plusplus
```

### Train with DAPO:

We suggest use server rollout mode for stability and efficiency.

```bash
bash scripts/rollout.sh
```

```bash
bash scripts/train_dapo
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
