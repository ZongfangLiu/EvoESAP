<div align="center">

# EvoESAP: Non-Uniform Expert Pruning for Sparse MoE


[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](./.python-version)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-ee4c2c.svg)](./pyproject.toml)
[![Transformers](https://img.shields.io/badge/Transformers-5.0.0dev-fcc624.svg)](./pyproject.toml)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](./LICENSE)

</div>

---

## Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [vLLM Integration](#vllm-integration)
- [Usage Example](#usage-example)
- [Parameter Guide](#parameter-guide)
- [Outputs](#outputs)
- [Acknowledgement](#acknowledgement)

## Overview

At a high level, EvoESAP runs:

1. Observer data collection on calibration prompts.
2. Evolutionary search over layer-wise pruning plans.
3. Best-plan materialization (structural non-uniform or mask-based).
4. Optional final evaluation on selected benchmarks.

```mermaid
flowchart LR
  A[Calibration Dataset] --> B[Observer Statistics]
  B --> C[Evolutionary Search]
  C --> D[Best Layer-wise Pruning Plan]
  D --> E[Pruned MoE Checkpoint]
  E --> F[Evaluation and Reports]
```

## Environment Setup

### Minimal setup

```bash
# 1) Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) Build environment from repo root
bash scripts/build.sh

# 3) Activate
source .venv/bin/activate
```

For Docker installation, refer to the original REAP repository:  
[CerebrasResearch/reap](https://github.com/CerebrasResearch/reap)

For using free gpt-oss-120b api to evaluate on wildbench, acquire from [here](https://build.nvidia.com/openai/gpt-oss-120b).

## vLLM Integration

Upstream vLLM does not natively support checkpoints with a non-uniform number of experts per layer.  
You can either:

1. Evaluate with **mask-based pruning** (no structural expert removal), or
2. Patch vLLM to support **structurally pruned** checkpoints.

To enable structural expert removal in vLLM:

1. Place the `vllm` source under `third-party/`.
2. Install `vllm` in editable mode.
3. Apply the model implementations from `src/reap/models/non_uniform/` for your target model family.

## Usage Example

```bash
bash experiments/search-pruning-cli.sh 0 allenai/OLMoE-1B-7B-0125-Instruct reap 42 \
16 \
theblackcat102/evol-codealpaca-v1 \
allenai/tulu-3-sft-personas-math \
32 50 4 4 3 32 64 \
esp-dataset \
true true true true true \
false false \
false 0 false \
true true \
true true "" 64 0
```

## Parameter Guide

The script `experiments/search-pruning-cli.sh` uses positional arguments.

### Quick grouping

| Group | Positions | Purpose |
| --- | --- | --- |
| Runtime | `1-4` | GPU, model, pruning metric, random seed |
| Search budget/data | `5-15` | sparsity budget, calibration/search datasets, EA hyperparameters, fitness |
| Evaluation toggles | `16-20` | turn benchmark suites on/off |
| Routing/eval behavior | `21-27` | singleton control, lm-eval formatting, server mode, structural mode |
| Control flow | `28-33` | run search/eval, override dir, expert count for naming, resume/checkpoint |

<details>
<summary><strong>Full positional argument table</strong></summary>

| Pos | Value in example | Name in script | Meaning |
| --- | --- | --- | --- |
| 1 | `0` | `CUDA_VISIBLE_DEVICES` | GPU device(s) to use. `0` means GPU 0. |
| 2 | `allenai/OLMoE-1B-7B-0125-Instruct` | `model_name` | Base model checkpoint name. |
| 3 | `reap` | `pruning_method` | Expert saliency/pruning metric used by search. |
| 4 | `42` | `seed` | Random seed for reproducibility. |
| 5 | `16` | `int_sparsity` | Integer sparsity budget (experts pruned per layer baseline). |
| 6 | `theblackcat102/evol-codealpaca-v1` | `dataset_name` | Calibration/observer dataset used to collect pruning statistics. |
| 7 | `allenai/tulu-3-sft-personas-math` | `search_dataset_name` | Dataset used to score candidate plans in search. |
| 8 | `32` | `population_size` | Number of candidate plans per generation. |
| 9 | `50` | `generations` | Number of evolutionary generations. |
| 10 | `4` | `topk` | Number of top candidates kept each generation. |
| 11 | `4` | `mutation_max_delta` | Max experts shifted in one mutation step. |
| 12 | `3` | `mutation_times` | Mutation attempts per offspring plan. |
| 13 | `32` | `mutation_max_attempts` | Max retries to create a valid mutation. |
| 14 | `64` | `search_samples_per_category` | Search scoring sample count. |
| 15 | `esp-dataset` | `fitness` | Fitness objective used by search. |
| 16 | `true` | `run_lm_eval` | Run lm-eval tasks after search. |
| 17 | `true` | `run_evalplus` | Run EvalPlus coding benchmarks. |
| 18 | `true` | `run_livecodebench` | Run LiveCodeBench benchmarks. |
| 19 | `true` | `run_math` | Run math evaluation tasks. |
| 20 | `true` | `run_wildbench` | Run WildBench evaluation. |
| 21 | `false` | `singleton_super_experts` | Keep detected super-experts as singleton. |
| 22 | `false` | `singleton_outlier_experts` | Keep detected outlier experts as singleton. |
| 23 | `false` | `lm_eval_apply_chat_template` | Apply tokenizer chat template in lm-eval. |
| 24 | `0` | `lm_eval_num_fewshot` | Few-shot count for lm-eval. |
| 25 | `false` | `lm_eval_fewshot_as_multiturn` | Format few-shot examples as multi-turn chat. |
| 26 | `true` | `use_server` | Use vLLM server backend for evaluation. |
| 27 | `true` | `structural_nonuniform` | Use structural non-uniform checkpoint output when supported. |
| 28 | `true` | `run_search` | Execute search stage. |
| 29 | `true` | `run_final_eval` | Execute final evaluation stage. |
| 30 | `""` | `pruned_dir_override` | Optional explicit pruned model directory to evaluate. |
| 31 | `64` | `num_experts` | Experts per layer used for ratio naming (`int_sparsity / num_experts`). |
| 32 | `0` | `search_resume_from` | Resume generation target (`0` means from scratch). |
| 33 | *(not provided)* | `search_checkpoint_every` | Checkpoint interval during search (defaults to `10`). |

</details>

## Outputs

Typical run artifacts are written under:

```text
artifacts/<model_name>/<dataset_name>/pruned_models_searched/<run_name>/
```

Important files:

- `pruned_experts.json`: pruned expert indices by layer
- `search_history.json`: generation-by-generation search trace
- `search_metadata.json`: run config and bookkeeping
- `nonuniform_plan.json`: final selected layer-wise plan
- `eval_*`: evaluation outputs

## Acknowledgement

Built on top of the excellent REAP codebase:  
**REAP (Router-weighted Expert Activation Pruning)** by Cerebras Research.
