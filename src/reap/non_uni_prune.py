from __future__ import annotations

import dataclasses
import hashlib
import gc
import json
import logging
import pathlib
import random
import re
import time
from typing import Any

import torch
from accelerate.hooks import remove_hook_from_module
from accelerate.utils import set_seed
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from reap.models.non_uniform.olmoe.configuration_olmoe_nonuniform import (
    NonUniformOlmoeConfig,
)
from reap.models.non_uniform.olmoe.modeling_olmoe_nonuniform import (
    NonUniformOlmoeForCausalLM,
)
from reap.models.non_uniform.ernie4_5_moe.configuration_ernie4_5_moe_nonuniform import (
    NonUniformErnie4_5_MoeConfig,
)
from reap.models.non_uniform.ernie4_5_moe.modeling_ernie4_5_moe_nonuniform import (
    NonUniformErnie4_5_MoeForCausalLM,
    Ernie4_5_MoeSparseMoeBlock,
)
from reap.models.non_uniform.qwen3_moe.configuration_qwen3_moe_nonuniform import (
    NonUniformQwen3MoeConfig,
)
from reap.models.non_uniform.qwen3_moe.modeling_qwen3_moe_nonuniform import (
    NonUniformQwen3MoeForCausalLM,
    Qwen3MoeSparseMoeBlock,
)
from reap.args import (
    ClusterArgs,
    DatasetArgs,
    EvalArgs,
    ModelArgs,
    ObserverArgs,
    PruneArgs,
    ReapArgs,
    SearchArgs,
)
from reap.data import DATASET_REGISTRY
from reap.eval import run_evaluate
from reap.main import (
    apply_layerwise_pruning,
    record_activations,
    create_results_directory,
    str_to_directory_name,
)
from reap.model_util import MODEL_ATTRS, get_super_expert_indices, patched_model_map
from reap.prune import dump_args_to_yaml
from reap.search_utils import evolutionary_search, normalize_run_name_base

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _batch_to_tensors(
    batch: Any,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if isinstance(batch, torch.Tensor):
        input_ids = batch
        attention_mask = None
        labels_mask = None
    elif isinstance(batch, dict):
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels_mask = batch.get("labels_mask")
    else:
        input_ids = batch.input_ids
        attention_mask = getattr(batch, "attention_mask", None)
        labels_mask = getattr(batch, "labels_mask", None)

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if attention_mask is not None and attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)
    if labels_mask is not None and labels_mask.dim() == 1:
        labels_mask = labels_mask.unsqueeze(0)
    return input_ids, attention_mask, labels_mask


def _collate_microbatch(items: list[Any], pad_token_id: int) -> Any:
    if not items:
        raise ValueError("Cannot collate empty microbatch.")

    input_ids_list: list[torch.Tensor] = []
    attention_mask_list: list[torch.Tensor | None] = []
    labels_mask_list: list[torch.Tensor | None] = []
    max_len = 0
    need_attention_mask = False
    need_labels_mask = False

    for item in items:
        input_ids, attention_mask, labels_mask = _batch_to_tensors(item)
        input_ids = input_ids.detach().cpu()
        input_ids_list.append(input_ids)
        max_len = max(max_len, int(input_ids.shape[1]))

        if attention_mask is not None:
            need_attention_mask = True
            attention_mask = attention_mask.detach().cpu()
        attention_mask_list.append(attention_mask)

        if labels_mask is not None:
            need_labels_mask = True
            labels_mask = labels_mask.detach().cpu()
        labels_mask_list.append(labels_mask)

    all_same_len = all(int(t.shape[1]) == max_len for t in input_ids_list)
    all_tensor_items = all(isinstance(x, torch.Tensor) for x in items)
    if all_same_len and all_tensor_items and (not need_attention_mask) and (not need_labels_mask):
        return torch.cat(input_ids_list, dim=0)

    def _pad_2d(t: torch.Tensor, value: int) -> torch.Tensor:
        pad = max_len - int(t.shape[1])
        if pad <= 0:
            return t
        return torch.nn.functional.pad(t, (0, pad), value=value)

    batch_input_ids = torch.cat(
        [_pad_2d(t, pad_token_id) for t in input_ids_list],
        dim=0,
    )

    batch_attention_mask: torch.Tensor | None = None
    if need_attention_mask or not all_same_len:
        masks = []
        for idx, ids in enumerate(input_ids_list):
            attn = attention_mask_list[idx]
            if attn is None:
                attn = torch.ones_like(ids, dtype=torch.long)
            masks.append(_pad_2d(attn.to(dtype=torch.long), 0))
        batch_attention_mask = torch.cat(masks, dim=0)

    batch_labels_mask: torch.Tensor | None = None
    if need_labels_mask:
        masks = []
        for idx, ids in enumerate(input_ids_list):
            m = labels_mask_list[idx]
            if m is None:
                m = torch.ones_like(ids, dtype=torch.long)
            masks.append(_pad_2d(m.to(dtype=torch.long), 0))
        batch_labels_mask = torch.cat(masks, dim=0)

    out: dict[str, Any] = {"input_ids": batch_input_ids}
    if batch_attention_mask is not None:
        out["attention_mask"] = batch_attention_mask
    if batch_labels_mask is not None:
        out["labels_mask"] = batch_labels_mask
    return out


def _rebatch_search_batches(
    search_batches: list[Any], microbatch_size: int, pad_token_id: int
) -> list[Any]:
    if microbatch_size <= 1:
        return search_batches
    out: list[Any] = []
    for i in range(0, len(search_batches), microbatch_size):
        out.append(_collate_microbatch(search_batches[i : i + microbatch_size], pad_token_id=pad_token_id))
    return out


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_dataset_list(value: str | None, fallback: str | None = None) -> list[str]:
    raw = value or fallback
    if not raw:
        return []
    items = _split_csv(raw)
    return items if items else [raw.strip()]


def _parse_weight_list(weights: str | None, count: int) -> list[float]:
    if not weights:
        return [1.0] * count
    items = _split_csv(weights)
    if len(items) != count:
        raise ValueError(
            "search_dataset_weights must match search_dataset_name count "
            f"(expected {count}, got {len(items)})"
        )
    parsed: list[float] = []
    for item in items:
        try:
            weight = float(item)
        except ValueError as exc:
            raise ValueError(f"Invalid weight '{item}' in search_dataset_weights.") from exc
        if weight <= 0:
            raise ValueError(
                f"search_dataset_weights values must be > 0 (got {weight})."
            )
        parsed.append(weight)
    return parsed


def _format_weight_token(weight: float) -> str:
    if abs(weight - round(weight)) < 1e-6:
        return str(int(round(weight)))
    return f"{weight:g}"


def _mix_short_dataset_name(dataset_name: str) -> str:
    base = dataset_name.split("/")[-1]
    base_lower = base.lower()
    if base_lower == "c4":
        return "c4"
    if base_lower.startswith("tulu-3"):
        return "tulu"
    if "evol-codealpaca" in base_lower:
        return "evol"
    return str_to_directory_name(base)


def _build_mixed_calib_tag(dataset_names: list[str], weights: list[float]) -> str:
    if len(dataset_names) != len(weights):
        raise ValueError("Dataset/weight length mismatch when building calib tag.")
    parts: list[str] = []
    for dataset_name, weight in zip(dataset_names, weights):
        parts.append(_mix_short_dataset_name(dataset_name))
        parts.append(_format_weight_token(weight))
    return "_".join(parts)


def _allocate_dataset_samples(total: int, weights: list[float]) -> list[int]:
    if total <= 0:
        raise ValueError("search_samples_per_category must be > 0.")
    if not weights:
        raise ValueError("Weights list is empty.")
    if total < len(weights):
        raise ValueError(
            "search_samples_per_category must be >= number of datasets when mixing."
        )
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("search_dataset_weights must sum to > 0.")

    counts = [1] * len(weights)
    remaining = total - len(weights)
    if remaining <= 0:
        return counts

    raw = [remaining * (w / total_weight) for w in weights]
    floors = [int(x) for x in raw]
    counts = [base + inc for base, inc in zip(counts, floors)]
    leftover = remaining - sum(floors)
    if leftover > 0:
        fractional = sorted(
            enumerate(raw),
            key=lambda item: item[1] - floors[item[0]],
            reverse=True,
        )
        for i in range(leftover):
            idx = fractional[i % len(fractional)][0]
            counts[idx] += 1
    return counts


def _prepare_search_batches(
    tokenizer,
    search_args: SearchArgs,
    ds_args: DatasetArgs,
    obs_args: ObserverArgs,
):
    """Load and tokenize the search dataset for scoring candidates."""
    dataset_config = search_args.search_dataset_config_name or ds_args.dataset_config_name
    split = search_args.search_dataset_split or ds_args.split

    pack_samples = search_args.search_pack_samples
    if search_args.fitness in {"spec-dec", "esp-p", "kl-p"} and pack_samples:
        logger.warning(
            "Disabling search_pack_samples for %s fitness to preserve generation headroom.",
            search_args.fitness,
        )
        pack_samples = False

    dataset_names = _parse_dataset_list(
        search_args.search_dataset_name, ds_args.dataset_name
    )
    if not dataset_names:
        raise RuntimeError("No search dataset specified.")
    mixed = len(dataset_names) > 1
    if mixed:
        weights = _parse_weight_list(
            search_args.search_dataset_weights, len(dataset_names)
        )
        dataset_targets = _allocate_dataset_samples(
            search_args.search_samples_per_category, weights
        )
    else:
        if search_args.search_dataset_weights:
            logger.warning(
                "search_dataset_weights provided for single dataset; ignoring."
            )
        weights = [1.0] * len(dataset_names)
        dataset_targets = [search_args.search_samples_per_category]

    logger.info(
        "Loading search dataset(s)=%s config=%s split=%s",
        ",".join(dataset_names),
        dataset_config,
        split,
    )

    batches: list[Any] = []
    example_preview = None
    for dataset_name, weight, target_total in zip(dataset_names, weights, dataset_targets):
        samples_per_category = target_total

        try:
            if dataset_name == "allenai/c4":
                file_urls = [
                    "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz",
                    "https://hf-mirror.com/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz",
                ]
                raw_ds = None
                last_exc = None
                for file_url in file_urls:
                    try:
                        raw_ds = load_dataset(
                            "json",
                            data_files={"train": file_url},
                            split="train",
                            streaming=False,
                        )
                        last_exc = None
                        break
                    except Exception as e:
                        last_exc = e
                        logger.warning("Failed to load C4 shard from %s: %s", file_url, e)
                if last_exc is not None:
                    raise last_exc
            elif dataset_config:
                try:
                    raw_ds = load_dataset(dataset_name, name=dataset_config, split=split)
                except ValueError as e:
                    logger.warning(
                        "Search dataset config '%s' not found for '%s': %s. "
                        "Retrying with default config.",
                        dataset_config,
                        dataset_name,
                        e,
                    )
                    raw_ds = load_dataset(dataset_name, split=split)
            else:
                raw_ds = load_dataset(dataset_name, split=split)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load search dataset '{dataset_name}': {e}"
            )

        proc_cls = DATASET_REGISTRY.get(dataset_name)
        if proc_cls is None:
            raise ValueError(
                f"No DatasetProcessor registered for search dataset '{dataset_name}'. "
                f"Supported: {list(DATASET_REGISTRY.keys())}"
            )

        processor = proc_cls(
            dataset=raw_ds,
            tokenizer=tokenizer,
            pack_samples=pack_samples,
            max_input_len=obs_args.model_max_length,
            split=split,
            split_by_category=obs_args.split_by_category,
            return_vllm_tokens_prompt=False,
            truncate=obs_args.truncate,
            select_only_categories=obs_args.select_only_categories,
        )
        category_count = len(
            processor.select_only_categories or processor.categories
        )
        if obs_args.split_by_category and category_count > 0:
            samples_per_category = max(
                1, int((target_total + category_count - 1) / category_count)
            )
        category_data = processor.get_processed_dataset(
            samples_per_category=samples_per_category,
            return_loss_mask=search_args.fitness
            in {"nll-assistant", "kl-baseline", "kl-pq-b", "esp-dataset", "sp-dataset"},
        )
        dataset_batch_count = 0
        for _, items in category_data.items():
            batches.extend(items)
            dataset_batch_count += len(items)
        logger.info(
            "Prepared %d search batches from %d categories for dataset %s "
            "(samples_per_category=%d, weight=%s)",
            dataset_batch_count,
            len(category_data),
            dataset_name,
            samples_per_category,
            _format_weight_token(weight),
        )
        if obs_args.split_by_category and category_count > 1:
            logger.info(
                "Dataset %s target_total=%d, split_by_category=%s, categories=%d, "
                "requested_per_category=%d (approx_total=%d)",
                dataset_name,
                target_total,
                obs_args.split_by_category,
                category_count,
                samples_per_category,
                samples_per_category * category_count,
            )

        if example_preview is None:
            try:
                mapped_ds = processor._mapped_dataset or processor.dataset
                if len(mapped_ds) > 0:
                    raw_sample = mapped_ds[0]
                    prompt_text = None
                    target_text = None
                    if isinstance(raw_sample, dict) and "messages" in raw_sample:
                        messages = raw_sample["messages"]
                        if messages and messages[-1].get("role") == "assistant":
                            target_text = messages[-1].get("content", "")
                            prompt_messages = messages[:-1]
                        else:
                            prompt_messages = messages
                        prompt_text = tokenizer.apply_chat_template(
                            prompt_messages,
                            add_generation_prompt=True,
                            tokenize=False,
                        )
                    elif isinstance(raw_sample, dict) and "text" in raw_sample:
                        prompt_text = raw_sample["text"]
                    if prompt_text:
                        prompt_ids = tokenizer(
                            prompt_text,
                            truncation=True,
                            max_length=obs_args.model_max_length,
                            return_tensors="pt",
                        )["input_ids"]
                        example_preview = {
                            "prompt_text": prompt_text,
                            "target_text": target_text,
                            "prompt_ids": prompt_ids,
                        }
            except Exception as e:
                logger.warning("Failed to build search example preview: %s", e)

    logger.info(
        "Prepared %d total search batches from %d dataset(s)",
        len(batches),
        len(dataset_names),
    )
    if example_preview is None:
        logger.warning("No search example preview was generated.")

    return batches, example_preview


def _get_non_uniform_dir(
    results_dir: pathlib.Path,
    run_name: str,
) -> pathlib.Path:
    safe_name = str_to_directory_name(run_name)
    return results_dir / "pruned_models_searched" / safe_name


def _resolve_search_checkpoint_dir(
    pruned_model_dir: pathlib.Path,
) -> pathlib.Path:
    return pruned_model_dir / "search_checkpoints"


def _fingerprint_search_batches(search_batches: list[Any]) -> str:
    hasher = hashlib.sha256()
    for batch in search_batches:
        input_ids, attention_mask, labels_mask = _batch_to_tensors(batch)
        for tensor in (input_ids, attention_mask, labels_mask):
            if tensor is None:
                continue
            buf = tensor.detach().cpu().contiguous().numpy().tobytes()
            hasher.update(buf)
    return hasher.hexdigest()


def _run_name_without_generation(run_name: str) -> str:
    return re.sub(r"-gen\d+", "", run_name)


def _format_run_name(
    prune_args: PruneArgs,
    search_args: SearchArgs,
    seed: int,
    prune_ratio: float,
    calib_dataset: str,
) -> str:
    dataset_names = _parse_dataset_list(calib_dataset)
    if len(dataset_names) <= 1:
        if search_args.search_dataset_weights:
            logger.warning(
                "search_dataset_weights provided for single dataset; ignoring for naming."
            )
        dataset_name = (dataset_names[0] if dataset_names else calib_dataset).split("/")[-1]
        if dataset_name == "tulu-3-sft-personas-math":
            dataset_name = "tulu-3"
        calib_tag = str_to_directory_name(dataset_name)
    else:
        weights = _parse_weight_list(
            search_args.search_dataset_weights, len(dataset_names)
        )
        calib_tag = str_to_directory_name(
            _build_mixed_calib_tag(dataset_names, weights)
        )
    search_bits = [
        f"int{search_args.int_sparsity}",
        f"pop{search_args.population_size}",
        f"gen{search_args.generations}",
        f"top{search_args.topk}",
        f"mut{search_args.mutation_max_delta}x{search_args.mutation_times}",
        f"att{search_args.mutation_max_attempts}",
        f"samples{search_args.search_samples_per_category}",
    ]
    if getattr(search_args, "mutation_even_experts", False):
        search_bits.append("even")
    name = (
        f"fit_{search_args.fitness}-{prune_args.prune_method}"
        f"-seed_{seed}"
        f"-ratio_{prune_ratio:.3f}"
        f"-{'-'.join(search_bits)}"
        f"-calib_{calib_tag}"
    )
    if prune_args.perserve_super_experts:
        name += "-perserve_super"
    elif prune_args.perserve_outliers:
        name += "-perserve_outlier"
    uniform_only = bool(search_args.init_patterns) and all(
        pattern == "uniform" for pattern in search_args.init_patterns
    )
    if uniform_only:
        name += "-uni"
    return name


def _copy_nonuniform_model_code(target_dir: pathlib.Path):
    """Copy the custom non-uniform OLMoE modeling files into the saved model directory."""
    src_dir = pathlib.Path(__file__).resolve().parent / "models" / "non_uniform" / "olmoe"
    for fname in ["__init__.py", "configuration_olmoe_nonuniform.py", "modeling_olmoe_nonuniform.py"]:
        src = src_dir / fname
        dst = target_dir / fname
        dst.write_text(src.read_text())

def _copy_nonuniform_ernie_model_code(target_dir: pathlib.Path):
    """Copy the custom non-uniform ERNIE4.5 MoE modeling files into the saved model directory."""
    src_dir = pathlib.Path(__file__).resolve().parent / "models" / "non_uniform" / "ernie4_5_moe"
    for fname in [
        "__init__.py",
        "configuration_ernie4_5_moe_nonuniform.py",
        "modeling_ernie4_5_moe_nonuniform.py",
    ]:
        src = src_dir / fname
        dst = target_dir / fname
        dst.write_text(src.read_text())

def _copy_nonuniform_qwen3_model_code(target_dir: pathlib.Path):
    """Copy the custom non-uniform Qwen3 MoE modeling files into the saved model directory."""
    src_dir = pathlib.Path(__file__).resolve().parent / "models" / "non_uniform" / "qwen3_moe"
    for fname in [
        "__init__.py",
        "configuration_qwen3_moe_nonuniform.py",
        "modeling_qwen3_moe_nonuniform.py",
    ]:
        src = src_dir / fname
        dst = target_dir / fname
        dst.write_text(src.read_text())

def _build_nonuniform_config(base_config, per_layer_counts: list[int]) -> NonUniformOlmoeConfig:
    cfg_dict = base_config.to_dict()
    cfg_dict["num_experts_per_layer"] = per_layer_counts
    cfg = NonUniformOlmoeConfig(**cfg_dict)
    cfg.auto_map = {
        "AutoConfig": "configuration_olmoe_nonuniform.NonUniformOlmoeConfig",
        "AutoModelForCausalLM": "modeling_olmoe_nonuniform.NonUniformOlmoeForCausalLM",
        "AutoModel": "modeling_olmoe_nonuniform.NonUniformOlmoeModel",
    }
    cfg.architectures = ["NonUniformOlmoeForCausalLM"]
    return cfg


def _build_nonuniform_ernie_config(base_config, per_layer_counts: list[int]) -> NonUniformErnie4_5_MoeConfig:
    cfg_dict = base_config.to_dict()
    cfg_dict["num_experts_per_layer"] = per_layer_counts
    cfg = NonUniformErnie4_5_MoeConfig(**cfg_dict)
    cfg.auto_map = {
        "AutoConfig": "configuration_ernie4_5_moe_nonuniform.NonUniformErnie4_5_MoeConfig",
        "AutoModelForCausalLM": "modeling_ernie4_5_moe_nonuniform.NonUniformErnie4_5_MoeForCausalLM",
        "AutoModel": "modeling_ernie4_5_moe_nonuniform.NonUniformErnie4_5_MoeModel",
    }
    cfg.architectures = ["NonUniformErnie4_5_MoeForCausalLM"]
    return cfg


def _build_nonuniform_qwen3_config(base_config, per_layer_counts: list[int]) -> NonUniformQwen3MoeConfig:
    cfg_dict = base_config.to_dict()
    cfg_dict["num_experts_per_layer"] = per_layer_counts
    cfg = NonUniformQwen3MoeConfig(**cfg_dict)
    cfg.auto_map = {
        "AutoConfig": "configuration_qwen3_moe_nonuniform.NonUniformQwen3MoeConfig",
        "AutoModelForCausalLM": "modeling_qwen3_moe_nonuniform.NonUniformQwen3MoeForCausalLM",
        "AutoModel": "modeling_qwen3_moe_nonuniform.NonUniformQwen3MoeModel",
    }
    cfg.architectures = ["NonUniformQwen3MoeForCausalLM"]
    return cfg


def _structural_prune_olmoe(
    model: NonUniformOlmoeForCausalLM,
    pruned_experts_info: dict[str, list[int]],
    per_layer_counts: list[int],
):
    """Physically drop experts per layer for OLMoE."""
    for layer_idx, retained in enumerate(per_layer_counts):
        layer = model.model.layers[layer_idx]
        pruned = pruned_experts_info.get(str(layer_idx), [])
        if not pruned:
            continue
        moe = layer.mlp
        experts = getattr(moe, "experts", None)
        if isinstance(experts, torch.nn.ModuleList):
            num_experts = len(experts)
            retain_indices = [i for i in range(num_experts) if i not in pruned]
            retain_indices = sorted(retain_indices)
            moe.experts = torch.nn.ModuleList([experts[i] for i in retain_indices])
            moe.num_experts = len(retain_indices)
            if hasattr(moe, "top_k"):
                moe.top_k = min(int(moe.top_k), moe.num_experts)
            # gate weights: (num_experts, hidden_size)
            moe.gate.weight.data = moe.gate.weight.data[retain_indices]
            if moe.gate.bias is not None:
                moe.gate.bias.data = moe.gate.bias.data[retain_indices]
            continue

        if experts is None:
            raise ValueError(f"Layer {layer_idx}: missing experts module for structural prune.")
        num_experts = int(
            getattr(moe, "num_experts", None)
            or getattr(experts, "num_experts", None)
            or experts.gate_up_proj.shape[0]
        )
        retain_indices = [i for i in range(num_experts) if i not in pruned]
        retain_indices = sorted(retain_indices)
        kept = len(retain_indices)
        if kept <= 0:
            raise ValueError(f"Layer {layer_idx}: cannot prune all experts.")

        with torch.no_grad():
            if hasattr(experts, "gate_up_proj"):
                experts.gate_up_proj = torch.nn.Parameter(experts.gate_up_proj.data[retain_indices].clone())
            if hasattr(experts, "down_proj"):
                experts.down_proj = torch.nn.Parameter(experts.down_proj.data[retain_indices].clone())

        if hasattr(experts, "num_experts"):
            experts.num_experts = kept
        if hasattr(moe, "num_experts"):
            moe.num_experts = kept
        if hasattr(moe, "top_k"):
            moe.top_k = min(int(moe.top_k), kept)

        gate = getattr(moe, "gate", None)
        if gate is not None and hasattr(gate, "weight") and gate.weight is not None:
            gate.weight = torch.nn.Parameter(gate.weight.data[retain_indices].clone())
            if getattr(gate, "bias", None) is not None:
                gate.bias = torch.nn.Parameter(gate.bias.data[retain_indices].clone())
            if hasattr(gate, "num_experts"):
                gate.num_experts = kept
            if hasattr(gate, "top_k"):
                gate.top_k = min(int(gate.top_k), kept)

def _is_ernie_moe_layer(config, layer_idx: int) -> bool:
    try:
        end_idx = int(config.moe_layer_end_index)
        if end_idx == -1:
            end_idx = int(config.num_hidden_layers) - 1
        return (
            ((layer_idx + 1) % int(config.moe_layer_interval) == 0)
            and layer_idx >= int(config.moe_layer_start_index)
            and layer_idx <= end_idx
        )
    except Exception:
        return False


def _structural_prune_ernie4_5_moe(
    model: NonUniformErnie4_5_MoeForCausalLM,
    pruned_experts_info: dict[str, list[int]],
    per_layer_counts: list[int],
):
    """Physically drop experts per layer for ERNIE4.5 MoE (MoE layers only)."""
    for layer_idx, retained in enumerate(per_layer_counts):
        if not _is_ernie_moe_layer(model.config, layer_idx):
            continue
        pruned = pruned_experts_info.get(str(layer_idx), [])
        if not pruned:
            continue
        moe = model.model.layers[layer_idx].mlp
        if not isinstance(moe, Ernie4_5_MoeSparseMoeBlock):
            continue
        num_experts = int(getattr(moe, "num_experts", moe.gate.weight.shape[0]))
        retain_indices = [i for i in range(num_experts) if i not in pruned]
        retain_indices = sorted(retain_indices)

        kept = len(retain_indices)
        with torch.no_grad():
            # router weights: (num_experts, hidden_size)
            moe.gate.weight = torch.nn.Parameter(moe.gate.weight.data[retain_indices].clone())

            # correction bias: (groups=1, num_experts)
            moe.gate.moe_statics.e_score_correction_bias = torch.nn.Parameter(
                moe.gate.moe_statics.e_score_correction_bias.data[:, retain_indices].clone(),
                requires_grad=False,
            )

            # fused experts
            moe.experts.gate_up_proj = torch.nn.Parameter(moe.experts.gate_up_proj.data[retain_indices].clone())
            moe.experts.down_proj = torch.nn.Parameter(moe.experts.down_proj.data[retain_indices].clone())

        moe.num_experts = kept
        moe.top_k = min(int(getattr(moe, "top_k", model.config.moe_k)), kept)
        moe.gate.top_k = min(int(getattr(moe.gate, "top_k", model.config.moe_k)), kept)
        if hasattr(moe.gate, "num_experts"):
            moe.gate.num_experts = kept
        if hasattr(moe.experts, "num_experts"):
            moe.experts.num_experts = kept

        # keep config in sync for saved checkpoints
        model.config.num_experts_per_layer[layer_idx] = kept


def _is_qwen3_moe_layer(config, layer_idx: int) -> bool:
    try:
        mlp_only_layers = getattr(config, "mlp_only_layers", None) or []
        if layer_idx in mlp_only_layers:
            return False
        decoder_sparse_step = int(getattr(config, "decoder_sparse_step", 1))
        num_experts = getattr(config, "num_experts", 0)
        get_num_experts = getattr(config, "get_num_experts", None)
        if callable(get_num_experts):
            num_experts = get_num_experts(layer_idx)
        return int(num_experts) > 0 and ((layer_idx + 1) % decoder_sparse_step == 0)
    except Exception:
        return False


def _structural_prune_qwen3_moe(
    model: NonUniformQwen3MoeForCausalLM,
    pruned_experts_info: dict[str, list[int]],
    per_layer_counts: list[int],
):
    """Physically drop experts per layer for Qwen3 MoE."""
    for layer_idx, retained in enumerate(per_layer_counts):
        if not _is_qwen3_moe_layer(model.config, layer_idx):
            continue
        pruned = pruned_experts_info.get(str(layer_idx), [])
        if not pruned:
            continue
        moe = model.model.layers[layer_idx].mlp
        if not isinstance(moe, Qwen3MoeSparseMoeBlock):
            continue
        num_experts = int(getattr(moe, "num_experts", moe.gate.weight.shape[0]))
        retain_indices = [i for i in range(num_experts) if i not in pruned]
        retain_indices = sorted(retain_indices)
        kept = len(retain_indices)
        if kept <= 0:
            raise ValueError(
                f"Layer {layer_idx}: cannot prune all experts (num_experts={num_experts})."
            )

        with torch.no_grad():
            # router weights: (num_experts, hidden_size)
            moe.gate.weight = torch.nn.Parameter(
                moe.gate.weight.data[retain_indices].clone()
            )

            # fused experts
            moe.experts.gate_up_proj = torch.nn.Parameter(
                moe.experts.gate_up_proj.data[retain_indices].clone()
            )
            moe.experts.down_proj = torch.nn.Parameter(
                moe.experts.down_proj.data[retain_indices].clone()
            )

        if hasattr(moe, "num_experts"):
            moe.num_experts = kept
        if hasattr(moe.gate, "num_experts"):
            moe.gate.num_experts = kept
        if hasattr(moe.gate, "top_k"):
            moe.gate.top_k = min(int(moe.gate.top_k), kept)
        if hasattr(moe.experts, "num_experts"):
            moe.experts.num_experts = kept

        # keep config in sync for saved checkpoints
        model.config.num_experts_per_layer[layer_idx] = kept

def main():
    parser = HfArgumentParser(
        (
            ReapArgs,
            DatasetArgs,
            ObserverArgs,
            ModelArgs,
            EvalArgs,
            PruneArgs,
            ClusterArgs,
            SearchArgs,
        )
    )
    (
        reap_args,
        ds_args,
        obs_args,
        model_args,
        eval_args,
        prune_args,
        cluster_args,
        search_args,
    ) = parser.parse_args_into_dataclasses()

    if prune_args.perserve_super_experts and prune_args.perserve_outliers:
        raise ValueError("Only one of perserve_super_experts or perserve_outliers can be set to True.")

    set_seed(reap_args.seed)
    results_dir = create_results_directory(model_args.model_name, ds_args.dataset_name)
    model_name = patched_model_map(model_args.model_name)
    assistant_device = search_args.spec_dec_assistant_device
    baseline_device = search_args.spec_dec_baseline_device
    if search_args.fitness == "spec-dec" and not assistant_device and not baseline_device:
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            assistant_device = "cuda:0"
            baseline_device = "cuda:1"
            logger.info(
                "Spec-dec auto device assignment: assistant=%s baseline=%s",
                assistant_device,
                baseline_device,
            )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model_device_map = "auto"
    if (
        search_args.fitness == "spec-dec"
        and assistant_device
    ):
        model_device_map = {"": assistant_device}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=model_device_map,
        torch_dtype="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    logger.info(
        "Collecting observer data using dataset %s (split=%s) for pruning metrics",
        ds_args.dataset_name,
        ds_args.split,
    )
    observer_data = record_activations(
        model,
        tokenizer,
        reap_args,
        model_args,
        ds_args,
        obs_args,
        results_dir,
    )
    if reap_args.run_observer_only:
        logger.info(
            "Observer run completed. Exiting after collecting activation data since "
            "`run_observer_only` is set to True."
        )
        return

    if prune_args.perserve_super_experts or prune_args.perserve_outliers:
        super_expert_idx = get_super_expert_indices(
            observer_data, include_last_layers=prune_args.perserve_outliers
        )
        metrics = [
            "expert_proba",
            "ean_sum",
            "ean_mean",
            "weighted_expert_frequency_sum",
            "weighted_ean_sum",
            "reap",
            "reap_l2",
            "weighted_ean_sum_l2",
        ]
        for layer in observer_data:
            super_experts_in_layer = super_expert_idx[
                super_expert_idx[:, 0] == layer
            ][:, 1]
            if len(super_experts_in_layer) > 0:
                for metric in metrics:
                    if metric in observer_data[layer]:
                        observer_data[layer][metric][super_experts_in_layer] = float(
                            "inf"
                        )

    layers = sorted(observer_data.keys())
    int_sparsity = search_args.int_sparsity
    budget = int_sparsity * len(layers)
    logger.info(
        "Integer sparsity (experts per layer): %d across %d layers (global budget=%d)",
        int_sparsity,
        len(layers),
        budget,
    )
    total_experts = sum(len(observer_data[layer]["expert_frequency"]) for layer in layers)
    prune_ratio = (budget / total_experts) if total_experts > 0 else 0.0
    calib_dataset = search_args.search_dataset_name or ds_args.dataset_name
    if search_args.output_dir:
        run_name = pathlib.Path(search_args.output_dir).name
    else:
        run_name = _format_run_name(
            prune_args, search_args, reap_args.seed, prune_ratio, calib_dataset
        )

    pruned_model_dir = (
        pathlib.Path(search_args.output_dir)
        if search_args.output_dir
        else _get_non_uniform_dir(results_dir, run_name)
    )
    pruned_model_dir.mkdir(parents=True, exist_ok=True)
    run_name_base = _run_name_without_generation(run_name)

    resume_state = None
    resume_path = None
    if search_args.search_resume_from:
        resume_path = pathlib.Path(search_args.search_resume_from)
        if not resume_path.exists():
            raise ValueError(f"search_resume_from path not found: {resume_path}")
        resume_state = torch.load(resume_path, weights_only=False)
        if not isinstance(resume_state, dict):
            raise ValueError(f"Invalid resume state format in {resume_path}")
        saved_base = resume_state.get("run_name_base")
        saved_base_clean = normalize_run_name_base(saved_base)
        run_name_base_clean = normalize_run_name_base(run_name_base)
        if saved_base_clean and run_name_base_clean and saved_base_clean != run_name_base_clean:
            raise ValueError(
                "Resume run name mismatch: only generations may differ "
                f"(expected base={saved_base_clean}, got={run_name_base_clean})."
            )
        saved_target_generations = resume_state.get("target_generations")
        if saved_target_generations is not None and search_args.generations < int(saved_target_generations):
            logger.warning(
                "Resume requested with generations=%d < previous target_generations=%d; continuing.",
                int(search_args.generations),
                int(saved_target_generations),
            )
        logger.info("Loaded search resume state from %s", resume_path)

    checkpoint_dir = None
    if search_args.search_checkpoint_every > 0:
        checkpoint_dir = _resolve_search_checkpoint_dir(pruned_model_dir)

    # Reset RNGs before search dataset sampling to keep selection deterministic.
    set_seed(reap_args.seed)
    search_batches, search_example_preview = _prepare_search_batches(
        tokenizer, search_args, ds_args, obs_args
    )
    if not search_batches:
        raise RuntimeError("No search batches prepared; cannot run evolutionary search.")

    # Reset RNGs immediately before search to keep search sampling identical across modes.
    set_seed(reap_args.seed)
    rng = random.Random(reap_args.seed)
    device = next(model.parameters()).device
    baseline_cache = {}
    baseline_model = None
    if search_args.fitness == "spec-dec":
        logger.info("Loading baseline model for %s fitness.", search_args.fitness)
        baseline_device_map = "auto"
        if baseline_device:
            baseline_device_map = {"": baseline_device}
        baseline_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=baseline_device_map,
            torch_dtype="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        try:
            if getattr(baseline_model.config, "use_flash_attention_2", False) is not True:
                baseline_model.config.use_flash_attention_2 = True
                logger.info("Enabled flash attention for baseline model.")
        except Exception as e:
            logger.warning("Could not enable flash attention on baseline model: %s", e)

    # Microbatch search scoring (dataset batching).
    requested_mb = int(getattr(search_args, "search_microbatch_size", 0) or 0)
    if resume_state is not None:
        resume_mb = resume_state.get("search_microbatch_size")
        if resume_mb is not None:
            resume_mb = int(resume_mb)
            if requested_mb not in (0, resume_mb):
                raise ValueError(
                    "search_microbatch_size does not match resume checkpoint "
                    f"(requested={requested_mb}, checkpoint={resume_mb})"
                )
            requested_mb = resume_mb
    mb_cap = int(getattr(search_args, "search_microbatch_max", 0) or 0)
    if requested_mb < 0:
        raise ValueError("search_microbatch_size must be >= 0")
    if mb_cap < 0:
        raise ValueError("search_microbatch_max must be >= 0")
    pad_token_id = int(getattr(tokenizer, "pad_token_id", None) or 0)
    max_try = min(len(search_batches), mb_cap if mb_cap > 0 else 64)

    def _try_microbatch(mb: int) -> bool:
        probe = _collate_microbatch(list(search_batches[:mb]), pad_token_id=pad_token_id)
        input_ids, attention_mask, labels_mask = _batch_to_tensors(probe)
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        model_was_training = model.training
        baseline_was_training = baseline_model.training if baseline_model is not None else None
        model.eval()
        if baseline_model is not None:
            baseline_model.eval()
        try:
            with torch.inference_mode():
                if search_args.fitness in {"nll-full", "nll-assistant"}:
                    labels = input_ids.clone()
                    if attention_mask is not None:
                        labels = labels.masked_fill(attention_mask == 0, -100)
                    if search_args.fitness == "nll-assistant":
                        if labels_mask is None:
                            labels_mask = torch.ones_like(labels)
                        labels = labels.masked_fill(labels_mask.to(device) == 0, -100)
                    _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
                elif search_args.fitness in {"kl-baseline", "kl-pq-b", "esp-dataset", "sp-dataset"}:
                    if attention_mask is None:
                        attention_mask = torch.ones_like(input_ids, device=device)
                    base_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    cand_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                    if search_args.fitness == "kl-baseline":
                        base_log_probs = torch.log_softmax(base_logits, dim=-1)
                        cand_log_probs = torch.log_softmax(cand_logits, dim=-1)
                        cand_probs = cand_log_probs.exp()
                        _ = (cand_probs * (cand_log_probs - base_log_probs)).sum(dim=-1).mean().item()
                    elif search_args.fitness == "kl-pq-b":
                        base_log_probs = torch.log_softmax(base_logits, dim=-1)
                        cand_log_probs = torch.log_softmax(cand_logits, dim=-1)
                        base_probs = base_log_probs.exp()
                        _ = (base_probs * (base_log_probs - cand_log_probs)).sum(dim=-1).mean().item()
                    elif search_args.fitness == "esp-dataset":
                        base_probs = torch.softmax(base_logits, dim=-1)
                        cand_probs = torch.softmax(cand_logits, dim=-1)
                        _ = torch.minimum(base_probs, cand_probs).sum(dim=-1).mean().item()
                    else:
                        base_log_probs = torch.log_softmax(base_logits, dim=-1)
                        cand_log_probs = torch.log_softmax(cand_logits, dim=-1)
                        sampled = torch.distributions.Categorical(logits=cand_log_probs).sample()
                        q_lp = cand_log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
                        p_lp = base_log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
                        log_ratio = p_lp - q_lp
                        _ = torch.exp(torch.minimum(log_ratio, torch.zeros_like(log_ratio))).mean().item()
                    del base_logits, cand_logits
                elif search_args.fitness in {"esp-p", "kl-p"}:
                    base_for_gen = baseline_model or model
                    if attention_mask is None:
                        attention_mask = torch.ones_like(input_ids, device=device)
                    generated = base_for_gen.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=search_args.esp_p_max_new_tokens,
                        do_sample=False,
                    )
                    gen_attention = torch.ones_like(generated, device=device)
                    base_logits = base_for_gen(input_ids=generated, attention_mask=gen_attention).logits
                    cand_logits = model(input_ids=generated, attention_mask=gen_attention).logits
                    if search_args.fitness == "kl-p":
                        base_log_probs = torch.log_softmax(base_logits, dim=-1)
                        cand_log_probs = torch.log_softmax(cand_logits, dim=-1)
                        cand_probs = cand_log_probs.exp()
                        _ = (cand_probs * (cand_log_probs - base_log_probs)).sum(dim=-1).mean().item()
                    else:
                        base_probs = torch.softmax(base_logits, dim=-1)
                        cand_probs = torch.softmax(cand_logits, dim=-1)
                        _ = torch.minimum(base_probs, cand_probs).sum(dim=-1).mean().item()
                    del generated, base_logits, cand_logits
                else:
                    # spec-dec: batch won't help much; just test a forward.
                    if attention_mask is None:
                        attention_mask = torch.ones_like(input_ids, device=device)
                    _ = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :].mean().item()
            return True
        except RuntimeError as e:
            msg = str(e).lower()
            oom = ("out of memory" in msg) or ("cuda" in msg and "memory" in msg)
            if not oom:
                raise
            return False
        finally:
            if model_was_training:
                model.train()
            if baseline_model is not None and baseline_was_training:
                baseline_model.train()
            try:
                del input_ids, attention_mask, labels_mask
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    microbatch_size = 1
    if search_args.fitness == "spec-dec":
        microbatch_size = 1
    elif requested_mb == 0 and max_try >= 2:
        last_ok = 1
        mb = 2
        while mb <= max_try and _try_microbatch(mb):
            last_ok = mb
            mb *= 2
        lo, hi = last_ok, min(max_try, mb - 1)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if _try_microbatch(mid):
                lo = mid
            else:
                hi = mid - 1
        microbatch_size = lo
    elif requested_mb > 0:
        microbatch_size = min(requested_mb, max(1, len(search_batches)))

    if microbatch_size > 1:
        old_n = len(search_batches)
        search_batches = _rebatch_search_batches(list(search_batches), microbatch_size, pad_token_id=pad_token_id)
        logger.info(
            "Using search_microbatch_size=%d (auto=%s): %d -> %d scoring batches",
            microbatch_size,
            requested_mb == 0,
            old_n,
            len(search_batches),
        )
    else:
        logger.info(
            "Using search_microbatch_size=1 (auto=%s): %d scoring batches",
            requested_mb == 0,
            len(search_batches),
        )

    search_data_fingerprint = None
    if search_args.search_checkpoint_every > 0 or resume_state is not None:
        search_data_fingerprint = _fingerprint_search_batches(search_batches)
        if resume_state is not None:
            saved_fingerprint = resume_state.get("search_data_fingerprint")
            if saved_fingerprint and saved_fingerprint != search_data_fingerprint:
                raise ValueError("Search data fingerprint mismatch with resume checkpoint.")

    if search_args.fitness in {"esp-p", "kl-p"}:
        from reap.search_utils import precompute_esp_generation_cache
        logger.info("Precomputing baseline generation cache for %s fitness.", search_args.fitness)
        baseline_cache = precompute_esp_generation_cache(
            model,
            search_batches,
            max_new_tokens=search_args.esp_p_max_new_tokens,
        )
    if search_args.fitness in {"kl-baseline", "kl-pq-b", "esp-dataset", "sp-dataset"}:
        from reap.search_utils import precompute_baseline_cache
        logger.info("Precomputing baseline logits cache for %s fitness.", search_args.fitness)
        baseline_cache = precompute_baseline_cache(model, search_batches, search_args.fitness)

    model_attrs = MODEL_ATTRS[model.__class__.__name__]
    min_keep_per_layer = max(1, int(getattr(model.config, model_attrs["num_experts_per_tok"])))

    best_plan, best_score, history = evolutionary_search(
        model=model,
        baseline_model=baseline_model,
        search_batches=search_batches,
        observer_data=observer_data,
        prune_args=prune_args,
        search_args=search_args,
        tokenizer=tokenizer,
        example_preview=search_example_preview,
        layers=layers,
        rng=rng,
        device=device,
        min_keep_per_layer=min_keep_per_layer,
        baseline_cache=baseline_cache,
        # esp-p / kl-p specific controls
        esp_p_max_new_tokens=search_args.esp_p_max_new_tokens,
        resume_state=resume_state,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=search_args.search_checkpoint_every,
        search_data_fingerprint=search_data_fingerprint,
        search_microbatch_size=microbatch_size,
        run_name_base=run_name_base,
        history_path=pruned_model_dir / "search_history.json",
    )
    logger.info("Best plan found with score %.4f: %s", best_score, best_plan)
    search_duration_hours = None
    if history and isinstance(history[0], dict) and "search_start_time" in history[0]:
        search_duration_hours = (time.time() - history[0]["search_start_time"]) / 3600.0
        logger.info("Search duration: %.2f hours", search_duration_hours)
    if resume_state is not None:
        resumed_from_generation = resume_state.get("generation")
        if resumed_from_generation is not None:
            logger.info("Search resumed from generation %d.", resumed_from_generation)

    # persist plan summary
    plan_summary = []
    for layer in layers:
        total = len(observer_data[layer]["expert_frequency"])
        pruned = best_plan.get(layer, 0)
        plan_summary.append(
            {
                "layer": layer,
                "total_experts": total,
                "pruned_experts": pruned,
                "kept_experts": total - pruned,
                "prune_ratio": pruned / total if total > 0 else 0.0,
            }
        )
    # compute pruned indices preview for reuse
    from reap.main import _preview_experts_to_prune

    pruned_experts_info = _preview_experts_to_prune(observer_data, best_plan, prune_args)
    pruned_experts_info_str = {str(k): v for k, v in pruned_experts_info.items()}
    total_pruned = sum(item["pruned_experts"] for item in plan_summary)
    prune_ratio = total_pruned / total_experts if total_experts > 0 else 0.0

    # Apply pruning
    if search_args.structural_nonuniform:
        model_type = getattr(model.config, "model_type", "")
        if model_type == "olmoe":
            logger.info("Applying structural non-uniform pruning for OLMoE.")
            retained_per_layer = []
            for layer in layers:
                total = len(observer_data[layer]["expert_frequency"])
                pruned = len(pruned_experts_info_str.get(str(layer), []))
                retained_per_layer.append(total - pruned)
            logger.info(
                "Structural OLMoE retained_per_layer: min=%d max=%d",
                min(retained_per_layer) if retained_per_layer else -1,
                max(retained_per_layer) if retained_per_layer else -1,
            )
            # warn if any layer retains fewer experts than the configured top-k
            orig_top_k = model.config.num_experts_per_tok
            too_small = [(idx, kept) for idx, kept in enumerate(retained_per_layer) if kept < orig_top_k]
            if too_small:
                logger.warning(
                    "Some layers retain fewer experts (%s) than original top_k=%s; consider lowering top_k or adjusting the plan.",
                    too_small,
                    orig_top_k,
                )
            # build custom config/model with original expert counts, load weights, then prune structurally
            nu_config = _build_nonuniform_config(
                model.config, [model.config.num_experts] * model.config.num_hidden_layers
            )
            nu_model = NonUniformOlmoeForCausalLM(nu_config)
            nu_model.load_state_dict(model.state_dict(), strict=False)
            _structural_prune_olmoe(nu_model, pruned_experts_info_str, retained_per_layer)
            nu_model.config.num_experts_per_layer = retained_per_layer
            nu_model.config.num_experts_per_tok = min(nu_model.config.num_experts_per_tok, min(retained_per_layer))
            model = nu_model
            model.config = nu_model.config
        elif model_type == "ernie4_5_moe":
            logger.info("Applying structural non-uniform pruning for ERNIE4.5 MoE.")
            num_layers = int(model.config.num_hidden_layers)
            base_num_experts = int(model.config.moe_num_experts)
            retained_per_layer = [base_num_experts] * num_layers

            for layer_idx in range(num_layers):
                if not _is_ernie_moe_layer(model.config, layer_idx):
                    continue
                pruned = pruned_experts_info_str.get(str(layer_idx), [])
                retained_per_layer[layer_idx] = base_num_experts - len(pruned)
            moe_retained_preview = [
                kept
                for idx, kept in enumerate(retained_per_layer)
                if _is_ernie_moe_layer(model.config, idx)
            ]
            if moe_retained_preview:
                logger.info(
                    "Structural ERNIE retained MoE experts: min=%d max=%d (moe_layers=%d)",
                    min(moe_retained_preview),
                    max(moe_retained_preview),
                    len(moe_retained_preview),
                )

            # warn if any MoE layer retains fewer experts than the configured top-k
            orig_top_k = int(model.config.moe_k)
            too_small = [
                (idx, kept)
                for idx, kept in enumerate(retained_per_layer)
                if _is_ernie_moe_layer(model.config, idx) and kept < orig_top_k
            ]
            if too_small:
                logger.warning(
                    "Some MoE layers retain fewer experts (%s) than original top_k=%s; consider lowering top_k or adjusting the plan.",
                    too_small,
                    orig_top_k,
                )

            nu_config = _build_nonuniform_ernie_config(model.config, [base_num_experts] * num_layers)
            nu_model = NonUniformErnie4_5_MoeForCausalLM(nu_config)
            nu_model.load_state_dict(model.state_dict(), strict=False)
            _structural_prune_ernie4_5_moe(
                nu_model, pruned_experts_info_str, retained_per_layer
            )
            nu_model.config.num_experts_per_layer = retained_per_layer
            moe_retained = [kept for idx, kept in enumerate(retained_per_layer) if _is_ernie_moe_layer(model.config, idx)]
            if moe_retained:
                nu_model.config.moe_k = min(int(nu_model.config.moe_k), int(min(moe_retained)))
            model = nu_model
            model.config = nu_model.config
        elif model_type == "qwen3_moe":
            logger.info("Applying structural non-uniform pruning for Qwen3 MoE.")
            num_layers = int(model.config.num_hidden_layers)
            base_num_experts = int(model.config.num_experts)
            retained_per_layer = [base_num_experts] * num_layers

            for layer_idx in range(num_layers):
                if not _is_qwen3_moe_layer(model.config, layer_idx):
                    continue
                pruned = pruned_experts_info_str.get(str(layer_idx), [])
                retained_per_layer[layer_idx] = base_num_experts - len(pruned)

            moe_retained_preview = [
                kept
                for idx, kept in enumerate(retained_per_layer)
                if _is_qwen3_moe_layer(model.config, idx)
            ]
            if moe_retained_preview:
                logger.info(
                    "Structural Qwen3 retained MoE experts: min=%d max=%d (moe_layers=%d)",
                    min(moe_retained_preview),
                    max(moe_retained_preview),
                    len(moe_retained_preview),
                )

            orig_top_k = int(model.config.num_experts_per_tok)
            too_small = [
                (idx, kept)
                for idx, kept in enumerate(retained_per_layer)
                if _is_qwen3_moe_layer(model.config, idx) and kept < orig_top_k
            ]
            if too_small:
                logger.warning(
                    "Some MoE layers retain fewer experts (%s) than original top_k=%s; consider lowering top_k or adjusting the plan.",
                    too_small,
                    orig_top_k,
                )

            nu_config = _build_nonuniform_qwen3_config(
                model.config, [base_num_experts] * num_layers
            )
            nu_model = NonUniformQwen3MoeForCausalLM(nu_config)
            nu_model.load_state_dict(model.state_dict(), strict=False)
            _structural_prune_qwen3_moe(
                nu_model, pruned_experts_info_str, retained_per_layer
            )
            nu_model.config.num_experts_per_layer = retained_per_layer
            moe_retained = [
                kept
                for idx, kept in enumerate(retained_per_layer)
                if _is_qwen3_moe_layer(model.config, idx)
            ]
            if moe_retained:
                nu_model.config.num_experts_per_tok = min(
                    int(nu_model.config.num_experts_per_tok), int(min(moe_retained))
                )
                nu_model.config.num_experts = int(max(moe_retained))
            model = nu_model
            model.config = nu_model.config
        else:
            raise ValueError(f"Structural non-uniform pruning not supported for model_type={model_type!r}")
    else:
        if not prune_args.zero_out:
            prune_args.zero_out = True
        logger.info("Applying mask-based pruning (zero-out).")
        pruned_experts_info = apply_layerwise_pruning(
            model,
            observer_data,
            best_plan,
            prune_args,
        )

    if search_args.output_dir:
        pruned_model_dir = pathlib.Path(search_args.output_dir)
    pruned_model_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    # persist custom model code if structural
    if search_args.structural_nonuniform:
        if getattr(model.config, "model_type", "") == "olmoe":
            _copy_nonuniform_model_code(pruned_model_dir)
        elif getattr(model.config, "model_type", "") == "ernie4_5_moe":
            _copy_nonuniform_ernie_model_code(pruned_model_dir)
        elif getattr(model.config, "model_type", "") == "qwen3_moe":
            _copy_nonuniform_qwen3_model_code(pruned_model_dir)
    model.save_pretrained(pruned_model_dir)
    tokenizer.save_pretrained(pruned_model_dir)
    with open(pruned_model_dir / "pruned_experts.json", "w") as f:
        json.dump({str(k): v for k, v in pruned_experts_info.items()}, f, indent=2)
    with open(pruned_model_dir / "search_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(pruned_model_dir / "nonuniform_plan.json", "w") as f:
        json.dump({"plan": plan_summary, "best_score": best_score}, f, indent=2)
    with open(pruned_model_dir / "search_metadata.json", "w") as f:
        json.dump(
            {
                "run_name": run_name,
                "seed": reap_args.seed,
                "prune_method": prune_args.prune_method,
                "prune_ratio": prune_ratio,
                "total_pruned": total_pruned,
                "total_experts": total_experts,
                "best_score": best_score,
                "search_duration_hours": search_duration_hours,
                "search_checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
                "search_resume_from": str(resume_path) if resume_path else None,
                "search_checkpoint_every": search_args.search_checkpoint_every,
                "search_data_fingerprint": search_data_fingerprint,
                "search_microbatch_size": microbatch_size,
                "resumed_from_generation": resume_state.get("generation") if resume_state else None,
                "search_params": dataclasses.asdict(search_args),
            },
            f,
            indent=2,
        )
    end = time.time()
    logger.info(
        "Saved pruned model to %s in %.2f seconds", pruned_model_dir, end - start
    )
    logger.info("PRUNED_MODEL_DIR=%s", pruned_model_dir)
    print(f"PRUNED_MODEL_DIR={pruned_model_dir}")
    dump_args_to_yaml(
        pruned_model_dir,
        reap_args,
        ds_args,
        obs_args,
        model_args,
        eval_args,
        prune_args,
        cluster_args,
        search_args,
    )

    if reap_args.do_eval:
        remove_hook_from_module(model, recurse=True)
        model.to("cpu")
        del model
        del observer_data
        torch.cuda.empty_cache()
        gc.collect()
        model_args.model_name = pruned_model_dir
        results_subdir = "eval_vllm" if eval_args.use_server else "eval_hf"
        run_evaluate(model_args, pruned_model_dir / results_subdir, eval_args, reap_args.seed)


if __name__ == "__main__":
    main()
