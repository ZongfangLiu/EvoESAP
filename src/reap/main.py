from __future__ import annotations
import time
import pickle
import logging
import dataclasses
import pathlib
import re
import time
from typing import Any
import gc
import yaml
import shutil
import json

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser

from accelerate.utils import set_seed
from accelerate.hooks import remove_hook_from_module


from reap.args import (
    ReapArgs,
    ModelArgs,
    DatasetArgs,
    ObserverArgs,
    ClusterArgs,
    KdArgs,
    EvalArgs,
    MergeArgs,
    PruneArgs,
)
from reap.merge import MergeMethod, MoEExpertMerger
from reap.data import DATASET_REGISTRY
from reap.observer import OBSERVER_CONFIG_REGISTRY, MoETransformerObserver
from reap.cluster import (
    get_penalty_vector,
    hierarchical_clustering,
    dynamic_frequency_penalized_clustering,
    multi_layer_hierarchical_clustering,
    mc_smoe_clustering,
    multi_layer_kmeans_clustering,
    multi_layer_kmeans_clustering_on_ca,
    restricted_hierarchical_clustering,
    kmeans_clustering
)
from reap.model_util import (
    get_moe,
    assert_merge,
    MODEL_ATTRS,
    patched_model_map,
    get_super_expert_indices,
    install_runtime_router_mask,
    maybe_resolve_model_attrs,
    resolve_model_attrs,
)
from reap.eval import run_evaluate
from reap.cluster_plots import plot_cluster_analysis
from reap.metrics import get_distance_fn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> tuple[dataclasses.Dataclass]:
    parser = HfArgumentParser(
        (
            ReapArgs,
            ModelArgs,
            DatasetArgs,
            ObserverArgs,
        ClusterArgs,
        KdArgs,
        EvalArgs,
        MergeArgs,
        PruneArgs,
    )
)
    args = parser.parse_args_into_dataclasses()
    return args


def str_to_directory_name(s: str) -> str:
    """Convert a string to a valid directory name by replacing special characters."""
    return re.sub(r"[^\w\-_.]", "_", s)


def create_results_directory(model_name: str, dataset_name: str) -> pathlib.Path:
    """Create a clean directory name from model and dataset names."""
    model_clean = model_name.split("/")[-1]
    dataset_clean = dataset_name.split("/")[-1]

    # Create clean directory name by removing special characters
    model_clean = str_to_directory_name(model_clean)
    dataset_clean = str_to_directory_name(dataset_clean)

    results_dir = pathlib.Path("./artifacts") / model_clean / dataset_clean

    if results_dir.exists():
        logger.warning(f"Directory '{results_dir}' already exists")
    else:
        results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created artifacts directory: {results_dir}")

    return results_dir


def plan_hybrid_prune_merge(
    observer_data: dict[int, dict[str, Any]],
    cluster_args: ClusterArgs,
    merge_args: MergeArgs,
) -> tuple[dict[int, int], dict[int, int]]:
    """
    Build a layer-wise plan for pruning vs. merging while keeping the overall
    compression budget consistent with the provided compression_ratio/num_clusters.
    Returns:
        prune_plan: layer -> #experts to prune
        merge_clusters: layer -> #clusters (num experts after merge) for merging layers
    """
    layers = sorted(observer_data.keys())
    num_experts_per_layer = {
        layer: len(observer_data[layer]["expert_frequency"]) for layer in layers
    }
    prune_layers = set(merge_args.hybrid_prune_layers)
    invalid_layers = [l for l in prune_layers if l not in observer_data]
    if invalid_layers:
        raise ValueError(
            f"hybrid_prune_layers contains invalid indices not in observer data: {invalid_layers}"
        )
    if merge_args.hybrid_prune_counts is not None and len(
        merge_args.hybrid_prune_counts
    ) != len(merge_args.hybrid_prune_layers):
        raise ValueError(
            "Length of hybrid_prune_counts must match hybrid_prune_layers when provided."
        )

    # Helper for baseline removal per layer given current compression settings.
    def base_remove(num_experts: int) -> int:
        if cluster_args.num_clusters is not None:
            return max(num_experts - cluster_args.num_clusters, 0)
        if cluster_args.compression_ratio is not None:
            return int(num_experts * cluster_args.compression_ratio)
        raise ValueError(
            "Either num_clusters or compression_ratio must be set to derive hybrid plan."
        )

    merge_layers = [l for l in layers if l not in prune_layers]
    if merge_args.skip_first and merge_layers:
        merge_layers = merge_layers[1:]
    if merge_args.skip_last and merge_layers:
        merge_layers = merge_layers[:-1]

    compressible_layers = set(merge_layers) | prune_layers
    if not compressible_layers:
        raise ValueError(
            "Hybrid pruning requested but no layers remain for pruning or merging."
        )

    baseline_remove = {
        layer: base_remove(num_experts_per_layer[layer]) for layer in compressible_layers
    }
    total_budget = sum(baseline_remove.values())

    # Build prune plan
    prune_plan: dict[int, int] = {}
    if merge_args.hybrid_prune_counts is not None:
        for layer, count in zip(
            merge_args.hybrid_prune_layers, merge_args.hybrid_prune_counts
        ):
            prune_plan[layer] = count
    for layer in prune_layers:
        prune_plan.setdefault(layer, baseline_remove[layer])
        max_removal = num_experts_per_layer[layer] - 1
        if prune_plan[layer] < 0 or prune_plan[layer] > max_removal:
            raise ValueError(
                f"Invalid prune count {prune_plan[layer]} for layer {layer}; "
                f"must be in [0, {max_removal}]."
            )

    actual_pruned = sum(prune_plan.values())
    target_merge_remove = total_budget - actual_pruned
    if target_merge_remove < 0:
        raise ValueError(
            "Pruning removes more experts than the global budget allows. "
            "Reduce hybrid_prune_counts or compression_ratio."
        )

    merge_clusters: dict[int, int] = {}
    if merge_layers:
        base_merge_total = sum(baseline_remove[l] for l in merge_layers)
        max_merge_total = sum(
            num_experts_per_layer[l] - 1 for l in merge_layers
        )
        if target_merge_remove > max_merge_total:
            raise ValueError(
                "Not enough merge capacity to meet the remaining compression budget. "
                "Lower hybrid_prune_counts or compression_ratio."
            )

        if base_merge_total == 0:
            if target_merge_remove != 0:
                raise ValueError(
                    "Requested additional compression for merge layers, "
                    "but baseline merge removal is zero."
                )
            merge_removal = {l: 0 for l in merge_layers}
        else:
            scale = target_merge_remove / base_merge_total if base_merge_total else 0.0
            merge_removal = {}
            residuals = []
            for l in merge_layers:
                desired = baseline_remove[l] * scale
                max_rem = num_experts_per_layer[l] - 1
                alloc = min(int(desired), max_rem)
                merge_removal[l] = alloc
                residuals.append((l, desired - alloc, max_rem))
            remaining = target_merge_remove - sum(merge_removal.values())
            if remaining > 0:
                residuals.sort(key=lambda x: x[1], reverse=True)
                for layer, _, max_rem in residuals:
                    if remaining <= 0:
                        break
                    if merge_removal[layer] < max_rem:
                        merge_removal[layer] += 1
                        remaining -= 1
            elif remaining < 0:
                excess = -remaining
                residuals = sorted(
                    merge_removal.items(), key=lambda x: x[1], reverse=True
                )
                for layer, current in residuals:
                    if excess <= 0:
                        break
                    if current > 0:
                        delta = min(current, excess)
                        merge_removal[layer] -= delta
                        excess -= delta
            # Final sanity: must hit target exactly or fail fast.
            if sum(merge_removal.values()) != target_merge_remove:
                raise ValueError(
                    "Failed to allocate merge removals to satisfy the compression budget."
                )
        merge_clusters = {
            l: num_experts_per_layer[l] - merge_removal[l] for l in merge_layers
        }
    else:
        if target_merge_remove != 0:
            raise ValueError(
                "No merge layers available but compression budget remains. "
                "Adjust hybrid_prune_layers or compression_ratio."
            )

    logger.info(
        "Hybrid plan: total budget=%d, prune=%d, merge=%d",
        total_budget,
        actual_pruned,
        target_merge_remove,
    )
    logger.info(
        "Prune layers -> counts: %s",
        {k: prune_plan[k] for k in sorted(prune_plan.keys())},
    )
    logger.info(
        "Merge layers -> clusters: %s",
        {k: merge_clusters[k] for k in sorted(merge_clusters.keys())},
    )
    return prune_plan, merge_clusters


def record_activations(
    model, tokenizer, reap_args, model_args, ds_args, obs_args, results_dir
):
    if ds_args.dataset_name == "combined":
        # just return the combined data
        cat_dir = results_dir / "all"
        f_name = cat_dir / obs_args.output_file_name
        if f_name.exists():
            return torch.load(f_name, weights_only=False)
        else:
            raise RuntimeError(
                f"Combined dataset requested but no pre-recorded data found at {f_name}"
            )
    try:
        if ds_args.dataset_name == "allenai/c4":
            file_url = "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz"
            c4_single_file_dataset = load_dataset(
                "json", data_files={"train": file_url}, split="train", streaming=False
            )
            raw_ds = c4_single_file_dataset
        else:
            raw_ds = load_dataset(ds_args.dataset_name, split=ds_args.split)
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{ds_args.dataset_name}': {e}")

    # load dataset processor
    proc_cls = DATASET_REGISTRY.get(ds_args.dataset_name)
    if proc_cls is None:
        raise ValueError(
            f"No DatasetProcessor registered for '{ds_args.dataset_name}'. "
            f"Supported: {list(DATASET_REGISTRY.keys())}"
        )

    # init processor & process dataset
    processor = proc_cls(
        dataset=raw_ds,
        tokenizer=tokenizer,
        max_input_len=obs_args.model_max_length,
        split=ds_args.split,
        split_by_category=obs_args.split_by_category,
        return_vllm_tokens_prompt=obs_args.return_vllm_tokens_prompt,
        truncate=obs_args.truncate,
    )
    category_data_batches = processor.get_processed_dataset(
        samples_per_category=obs_args.samples_per_category,
    )
    logger.info(
        "Loaded and processed data for categories: %s",
        str(list(category_data_batches.keys())),
    )

    # load observer and hook model
    try:
        renormalize_router_weights = getattr(model.config, "norm_topk_prob", False) and obs_args.renormalize_router_weights
        if renormalize_router_weights:
            logger.info("Renormalizing topk router weights to sum to 1.")
        observer_config = OBSERVER_CONFIG_REGISTRY[model.__class__.__name__](
            # distance_measure=obs_args.distance_measure,
            distance_measure='cosine',
            renormalize_router_weights=renormalize_router_weights,
            record_pruning_metrics_only=obs_args.record_pruning_metrics_only,
        )
    except KeyError:
        raise ValueError(
            f"No observer configuration registered for model '{model.__class__.__name__}'. "
            f"Supported: {list(OBSERVER_CONFIG_REGISTRY.keys())}"
        )
    observer = MoETransformerObserver(
        model=model,
        hook_config=observer_config,
    )

    if reap_args.profile:
        # profile at max len
        with torch.no_grad():
            try:
                model_max_length = obs_args.model_max_length
                if model_max_length is None:
                    model_max_length = tokenizer.model_max_length
                logger.info(f"Profiling at model max length: {model_max_length}.")
                s = "hello " * model_max_length
                tokenized = tokenizer(
                    [s],
                    return_tensors="pt",
                    truncation=True,
                    max_length=model_max_length,
                )
                tokenized = {k: v.to(model.device) for k, v in tokenized.items()}
                for _ in range(2):
                    _ = model(**tokenized)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to run model with max input length {model_max_length}: {e}"
                )
        logger.info(
            f"Model {model_args.model_name} successfully loaded and profiled at max length {model_max_length}."
        )
        observer.reset()

    # run samples over model and save observer state
    with torch.no_grad():
        for category, cat_data in category_data_batches.items():
            logger.info(f"Processing category: {category}...")
            cat_dir = results_dir / str_to_directory_name(category)
            cat_dir.mkdir(parents=True, exist_ok=True)
            f_name = cat_dir / obs_args.output_file_name
            # Reload existing data if present unless overwrite is set
            if f_name.exists() and not obs_args.overwrite_observations:
                logger.info(
                    f"Category '{category}' previously processed. Skipping to next category..."
                )
                continue
            try:
                logger.info("No previous data found @ %s", f_name)
                for sample in tqdm(cat_data, desc=f"Processing {category} samples"):
                    model(sample.to(model.device))
            except Exception as e:
                logger.error(f"Error processing category '{category}'")
                logger.info(
                    f"Saving partial results for category '{category}' and exiting"
                )
                observer.save_state(cat_dir / "partial.pkl")
                logger.info(
                    f"{category} data processed and saved to "
                    f"{cat_dir / obs_args.output_file_name}"
                )
                raise e
            observer.save_state(cat_dir / obs_args.output_file_name)
            observer.reset()
            logger.info(
                f"{category} data processed and saved to "
                f"{cat_dir / obs_args.output_file_name}"
            )
    observer.close_hooks()
    with open(f"{cat_dir / obs_args.output_file_name}", "rb") as f:
        observer_data = torch.load(f, weights_only=False)
    return observer_data


def cluster(
    data: dict[int, dict[str, Any]],
    num_clusters: int | dict[int, int],
    cluster_args: ClusterArgs,
    distance_measure: str,
    results_dir: pathlib.Path,
) -> dict[int, torch.Tensor]:
    """Cluster the model's experts based on the specified clustering method."""
    logger.info(f"Clustering experts using settings:\n{cluster_args.__str__()}\n")

    cluster_labels = {}
    distances = {}
    all_layer_expert_proba = {}
    if cluster_args.singleton_super_experts or cluster_args.singleton_outlier_experts:
        super_expert_idx = get_super_expert_indices(data, include_last_layers=cluster_args.singleton_outlier_experts)
    for layer in tqdm(data, "Clustering experts..."):
        if isinstance(num_clusters, dict):
            if layer not in num_clusters:
                continue
            layer_num_clusters = num_clusters[layer]
        else:
            layer_num_clusters = num_clusters

        expert_prob = data[layer]["expert_frequency"] / data[layer]["total_tokens"]
        ttm_sim_matrix = None
        try:
            ttm_sim_matrix = data[layer]["ttm_similarity_matrix"]
        except KeyError:
            pass
        online_characteristic_activation_dist = None
        try:
            online_characteristic_activation_dist = data[layer][
                "online_characteristic_activation_dist"
            ]
        except KeyError:
            pass
        ca = data[layer]["characteristic_activation"]
        routed_ca = None
        try:
            routed_ca = data[layer]["routed_characteristic_activation"]
        except KeyError:
            pass
        router_logits = data[layer]["router_logit_similiarity"]

        expert_similarity_scores = {
            "ttm": ttm_sim_matrix,
            "dynamic_ttm": ttm_sim_matrix,
            "characteristic_activation": ca,
            "routed_characteristic_activation": routed_ca,
            "router_logits": router_logits,
            "online_characteristic_activation_dist": online_characteristic_activation_dist,
        }
        distance = expert_similarity_scores[cluster_args.expert_sim]

        if cluster_args.expert_sim in [
            "characteristic_activation",
            "routed_characteristic_activation",
            "router_logits",
        ] and cluster_args.cluster_method != "kmeans":
            # get NxN similarity matrix for vector metrics
            distance_fn = get_distance_fn(distance_measure)
            distance = distance_fn(distance.unsqueeze(0), distance.unsqueeze(1))

        
        if cluster_args.singleton_super_experts:
            # set super expert distance to max
            super_experts_in_layer = super_expert_idx[super_expert_idx[:, 0] == layer][:, 1]
            if len(super_experts_in_layer) > 0:
                max_value = torch.finfo(distance.dtype).max
                distance[:, super_experts_in_layer] = max_value
                distance[super_experts_in_layer, :] = max_value

        distances[layer] = distance
        all_layer_expert_proba[layer] = expert_prob
        if cluster_args.multi_layer or cluster_args.cluster_method == "mc_smoe":
            continue
        if cluster_args.frequency_penalty and cluster_args.expert_sim != "dynamic_ttm":
            penalty = get_penalty_vector(
                expert_prob,
                cluster_args.softmax_temperature,
            )
            penalty_matrix = penalty.unsqueeze(0) + penalty.unsqueeze(1)
            penalized_distance = distance * penalty_matrix
            penalized_distance[penalized_distance.isnan()] = float("inf")
            distance = penalized_distance

        if cluster_args.expert_sim == "dynamic_ttm":
            cluster_label = dynamic_frequency_penalized_clustering(
                distance,
                expert_prob,
                layer_num_clusters,
                cluster_args.softmax_temperature,
            )

        elif cluster_args.cluster_method == "agglomerative":
            if (
                hasattr(cluster_args, "max_cluster_size")
                and cluster_args.max_cluster_size is None
            ):
                cluster_label = hierarchical_clustering(
                    distance,
                    cluster_args.linkage_method,
                    layer_num_clusters,
                )
            else:
                cluster_label = restricted_hierarchical_clustering(
                    distance,
                    cluster_args.linkage_method,
                    layer_num_clusters,
                    max_cluster_size=cluster_args.max_cluster_size,
                )
            if isinstance(cluster_label, np.ndarray):
                cluster_label = torch.tensor(cluster_label)

        elif cluster_args.cluster_method == "kmeans":
            cluster_label = kmeans_clustering(distance, layer_num_clusters)

        else:
            raise NotImplementedError(
                f"Clustering method '{cluster_args.cluster_method}' is not implemented."
            )
        cluster_labels[layer] = cluster_label

    if cluster_args.multi_layer:
        if isinstance(num_clusters, dict):
            raise ValueError(
                "Per-layer num_clusters is not supported with multi_layer clustering."
            )
        # we have parsed distances, time to cluster across layers]
        logger.info(
            f"Multi layer clustering with multi_layer={cluster_args.multi_layer}"
        )
        if cluster_args.cluster_method == "agglomerative":
            cluster_labels = multi_layer_hierarchical_clustering(
                distances,
                cluster_args.multi_layer,
                cluster_args.linkage_method,
                num_clusters,
            )
        elif cluster_args.cluster_method == "kmeans": 
            # try v2:
            if cluster_args.expert_sim != 'characteristic_activation':
                raise ValueError("multi_layer kmeans clustering on ca only implemented for characteristic_activation expert sim")
            cluster_labels = multi_layer_kmeans_clustering_on_ca(
                distances,
                num_layers=cluster_args.multi_layer,
                n_clusters=num_clusters,
            )
            
            # cluster_labels = multi_layer_kmeans_clustering(
            #     distances,
            #     num_layers=cluster_args.multi_layer,
            #     n_clusters=num_clusters,
            # )

    if cluster_args.cluster_method == "mc_smoe":
        if isinstance(num_clusters, dict):
            raise ValueError("Per-layer num_clusters is not supported with mc_smoe.")
        logger.info(f"Performing MC-SMoE adpative layer-wise merging...")
        cluster_labels = mc_smoe_clustering(
            distances,
            all_layer_expert_proba,
            total_clusters=len(distances) * num_clusters,
        )
    return cluster_labels


def apply_layerwise_pruning(
    model: nn.Module,
    observer_data: dict[int, dict[str, Any]],
    prune_plan: dict[int, int],
    prune_args: "PruneArgs",
) -> dict[str, list[int]]:
    """Prune selected layers according to prune_plan."""
    model_attrs = None
    pruned_experts_info: dict[str, list[int]] = {}

    def _zero_out_expert(expert_module: nn.Module):
        for _, param in expert_module.named_parameters():
            param.data.zero_()

    for layer, prune_count in sorted(prune_plan.items()):
        if prune_count <= 0:
            continue
        num_experts = observer_data[layer]["expert_frequency"].shape[0]
        if prune_count >= num_experts:
            raise ValueError(
                f"Cannot prune {prune_count} experts from layer {layer} with {num_experts} experts."
            )

        if prune_args.prune_method == "ean_ca":
            ean = torch.zeros(num_experts, device=model.device, dtype=torch.float32)
            for i in range(num_experts):
                ean[i] = torch.linalg.norm(
                    observer_data[layer]["routed_characteristic_activation"][i],
                    dim=-1,
                ).sum()
            _, experts_to_prune = torch.topk(ean, prune_count, largest=False)
        else:
            prune_method = prune_args.prune_method
            if prune_method == "frequency":
                prune_method = "expert_frequency"
            elif prune_method == "weighted_frequency_sum":
                prune_method = "weighted_expert_frequency_sum"
            saliency_data = observer_data[layer].get(prune_method)
            if saliency_data is None:
                raise ValueError(
                    f"Prune method {prune_args.prune_method} not found in observer data for layer {layer}. "
                    f"Available keys: {list(observer_data[layer].keys())}"
                )
            _, experts_to_prune = torch.topk(
                saliency_data, prune_count, largest=False
            )

        retained_expert_indices = [
            i for i in range(num_experts) if i not in experts_to_prune
        ]
        pruned_experts_info[str(layer)] = [
            int(idx) for idx in experts_to_prune.tolist()
        ]

        moe = get_moe(model, layer)
        model_attrs = resolve_model_attrs(model, moe)
        if prune_args.zero_out:
            if not model_attrs["fused"]:
                all_experts = getattr(moe, model_attrs["experts"])
                for idx in experts_to_prune.tolist():
                    _zero_out_expert(all_experts[idx])
                router = getattr(moe, model_attrs["router"])
                install_runtime_router_mask(router, experts_to_prune)
                setattr(moe, model_attrs["router"], router)
            else:
                experts = getattr(moe, model_attrs["experts"])
                gate_proj_attr = model_attrs["gate_proj"]
                down_proj_attr = model_attrs["down_proj"]
                getattr(experts, gate_proj_attr).data[experts_to_prune] = 0
                getattr(experts, down_proj_attr).data[experts_to_prune] = 0
                router = getattr(moe, model_attrs["router"])
                install_runtime_router_mask(router, experts_to_prune)
                setattr(moe, model_attrs["router"], router)
            # keep num_experts/out_features unchanged
        else:
            if not model_attrs["fused"]:
                all_experts = getattr(moe, model_attrs["experts"])
                retained_experts = [all_experts[i] for i in retained_expert_indices]
                retained_experts = torch.nn.ModuleList(retained_experts)
                setattr(moe, model_attrs["experts"], retained_experts)

                router = getattr(moe, model_attrs["router"])
                router.weight.data = router.weight.data[retained_expert_indices, :]
                if getattr(router, "bias", None) is not None:
                    router.bias.data = router.bias.data[retained_expert_indices]
                router.out_features = len(retained_expert_indices)
                if hasattr(router, "e_score_correction_bias"):
                    router.e_score_correction_bias.data = (
                        router.e_score_correction_bias.data[retained_expert_indices]
                    )
                setattr(moe, model_attrs["router"], router)
                # keep module-level expert count in sync to avoid routing past the new size
                if hasattr(moe, model_attrs["num_experts"]):
                    setattr(moe, model_attrs["num_experts"], len(retained_expert_indices))
                if hasattr(router, "num_experts"):
                    router.num_experts = len(retained_expert_indices)
            else:
                experts = getattr(moe, model_attrs["experts"])
                gate_proj_attr = model_attrs["gate_proj"]
                down_proj_attr = model_attrs["down_proj"]
                gate_proj = getattr(experts, gate_proj_attr)
                down_proj = getattr(experts, down_proj_attr)
                gate_proj.data = gate_proj.data[retained_expert_indices]
                down_proj.data = down_proj.data[retained_expert_indices]
                setattr(experts, gate_proj_attr, gate_proj)
                setattr(experts, down_proj_attr, down_proj)
                if hasattr(experts, "num_experts"):
                    experts.num_experts = len(retained_expert_indices)
                if hasattr(moe, model_attrs["num_experts"]):
                    setattr(moe, model_attrs["num_experts"], len(retained_expert_indices))
                elif hasattr(moe, "num_experts"):
                    moe.num_experts = len(retained_expert_indices)

                router = getattr(moe, model_attrs["router"])
                if hasattr(router, "weight") and router.weight is not None:
                    router.weight.data = router.weight.data[retained_expert_indices]
                if getattr(router, "bias", None) is not None:
                    router.bias.data = router.bias.data[retained_expert_indices]
                if hasattr(router, "out_features"):
                    router.out_features = len(retained_expert_indices)
                if hasattr(router, "num_experts"):
                    router.num_experts = len(retained_expert_indices)
                if hasattr(router, "top_k"):
                    router.top_k = min(int(router.top_k), len(retained_expert_indices))
                if hasattr(moe, "top_k"):
                    moe.top_k = min(int(moe.top_k), len(retained_expert_indices))
                if hasattr(router, "e_score_correction_bias"):
                    router.e_score_correction_bias.data = (
                        router.e_score_correction_bias.data[retained_expert_indices]
                    )
                if hasattr(router, "moe_statics") and hasattr(
                    router.moe_statics, "e_score_correction_bias"
                ):
                    statics_bias = router.moe_statics.e_score_correction_bias
                    if statics_bias is not None:
                        if statics_bias.data.ndim == 2:
                            statics_bias.data = statics_bias.data[:, retained_expert_indices]
                        else:
                            statics_bias.data = statics_bias.data[retained_expert_indices]

    return pruned_experts_info


def prune_to_cluster_representatives(
    model: nn.Module,
    cluster_labels: dict[int, torch.Tensor],
    pruned_experts_info: dict[str, list[int]],
):
    """
    Reduce merged layers to one expert per cluster to make expert counts match
    the intended compression (avoids config/weight shape mismatches downstream).
    """
    for layer, cluster_label in cluster_labels.items():
        # already pruned? skip if we recorded this layer
        if str(layer) in pruned_experts_info:
            continue
        moe = get_moe(model, layer)
        model_attrs = resolve_model_attrs(model, moe)
        unique_clusters, first_indices = torch.unique(cluster_label, return_inverse=False, return_counts=False, sorted=True, return_indices=True)  # type: ignore[arg-type]
        # torch.unique returns a tuple when return_indices in some versions; compute manually for clarity
        uniq = torch.unique(cluster_label)
        retain_indices = []
        for c in uniq:
            idx = (cluster_label == c).nonzero(as_tuple=True)[0][0].item()
            retain_indices.append(idx)
        retain_indices = sorted(retain_indices)
        pruned = [i for i in range(len(cluster_label)) if i not in retain_indices]
        if not pruned:
            continue
        pruned_experts_info[str(layer)] = pruned

        if not model_attrs["fused"]:
            all_experts = getattr(moe, model_attrs["experts"])
            retained_experts = [all_experts[i] for i in retain_indices]
            retained_experts = torch.nn.ModuleList(retained_experts)
            setattr(moe, model_attrs["experts"], retained_experts)

            router = getattr(moe, model_attrs["router"])
            router.weight.data = router.weight.data[retain_indices, :]
            if getattr(router, "bias", None) is not None:
                router.bias.data = router.bias.data[retain_indices]
            router.out_features = len(retain_indices)
            if hasattr(router, "e_score_correction_bias"):
                router.e_score_correction_bias.data = (
                    router.e_score_correction_bias.data[retain_indices]
                )
            if hasattr(moe, model_attrs["num_experts"]):
                setattr(moe, model_attrs["num_experts"], len(retain_indices))
            if hasattr(router, "num_experts"):
                router.num_experts = len(retain_indices)
            setattr(moe, model_attrs["router"], router)
        else:
            experts = getattr(moe, model_attrs["experts"])
            gate_proj_attr = model_attrs["gate_proj"]
            down_proj_attr = model_attrs["down_proj"]
            gate_proj = getattr(experts, gate_proj_attr)
            down_proj = getattr(experts, down_proj_attr)
            gate_proj.data = gate_proj.data[retain_indices]
            down_proj.data = down_proj.data[retain_indices]
            setattr(experts, gate_proj_attr, gate_proj)
            setattr(experts, down_proj_attr, down_proj)
            if hasattr(experts, "num_experts"):
                experts.num_experts = len(retain_indices)
            if hasattr(moe, model_attrs["num_experts"]):
                setattr(moe, model_attrs["num_experts"], len(retain_indices))
            elif hasattr(moe, "num_experts"):
                moe.num_experts = len(retain_indices)

            router = getattr(moe, model_attrs["router"])
            if hasattr(router, "weight") and router.weight is not None:
                router.weight.data = router.weight.data[retain_indices]
            if getattr(router, "bias", None) is not None:
                router.bias.data = router.bias.data[retain_indices]
            if hasattr(router, "out_features"):
                router.out_features = len(retain_indices)
            if hasattr(router, "num_experts"):
                router.num_experts = len(retain_indices)
            if hasattr(router, "top_k"):
                router.top_k = min(int(router.top_k), len(retain_indices))
            if hasattr(moe, "top_k"):
                moe.top_k = min(int(moe.top_k), len(retain_indices))
            if hasattr(router, "e_score_correction_bias"):
                router.e_score_correction_bias.data = (
                    router.e_score_correction_bias.data[retain_indices]
                )
            if hasattr(router, "moe_statics") and hasattr(
                router.moe_statics, "e_score_correction_bias"
            ):
                statics_bias = router.moe_statics.e_score_correction_bias
                if statics_bias is not None:
                    if statics_bias.data.ndim == 2:
                        statics_bias.data = statics_bias.data[:, retain_indices]
                    else:
                        statics_bias.data = statics_bias.data[retain_indices]

    return pruned_experts_info


def _preview_experts_to_prune(
    observer_data: dict[int, dict[str, Any]],
    prune_plan: dict[int, int],
    prune_args: "PruneArgs",
) -> dict[int, list[int]]:
    """
    Dry-run helper: compute which experts would be pruned per layer without mutating the model.
    Mirrors the selection logic in apply_layerwise_pruning.
    """
    preview: dict[int, list[int]] = {}
    for layer, prune_count in sorted(prune_plan.items()):
        if prune_count <= 0:
            continue
        num_experts = observer_data[layer]["expert_frequency"].shape[0]
        if prune_count >= num_experts:
            raise ValueError(
                f"Cannot prune {prune_count} experts from layer {layer} with {num_experts} experts."
            )
        if prune_args.prune_method == "ean_ca":
            ean = torch.zeros(num_experts, dtype=torch.float32)
            for i in range(num_experts):
                ean[i] = torch.linalg.norm(
                    observer_data[layer]["routed_characteristic_activation"][i],
                    dim=-1,
                ).sum()
            _, experts_to_prune = torch.topk(ean, prune_count, largest=False)
        else:
            prune_method = prune_args.prune_method
            if prune_method == "frequency":
                prune_method = "expert_frequency"
            elif prune_method == "weighted_frequency_sum":
                prune_method = "weighted_expert_frequency_sum"
            saliency_data = observer_data[layer].get(prune_method)
            if saliency_data is None:
                raise ValueError(
                    f"Prune method {prune_args.prune_method} not found in observer data for layer {layer}. "
                    f"Available keys: {list(observer_data[layer].keys())}"
                )
            _, experts_to_prune = torch.topk(
                saliency_data, prune_count, largest=False
            )
        preview[layer] = [int(idx) for idx in experts_to_prune.tolist()]
    return preview


def _summarize_router_logits(
    router_logits: torch.Tensor,
    top_k: int | None,
    pruned_indices: list[int],
) -> dict[str, Any]:
    """Summarize how often experts are selected in router top-k."""
    probs = torch.softmax(router_logits, dim=-1)
    k = top_k if top_k is not None else min(1, probs.shape[-1])
    k = min(k, probs.shape[-1])
    topk = torch.topk(probs, k=k, dim=-1).indices
    counts = torch.bincount(topk.view(-1), minlength=probs.shape[-1])
    take = min(5, counts.numel())
    top = torch.topk(counts, k=take)
    pruned_selected = counts[pruned_indices].sum().item() if pruned_indices else 0
    total_selected = counts.sum().item()
    pruned_pct = (
        float(pruned_selected) / total_selected * 100 if total_selected > 0 else 0.0
    )
    return {
        "top_experts": [(int(idx), int(counts[idx])) for idx in top.indices.tolist()],
        "pruned_selected": int(pruned_selected),
        "total_selected": int(total_selected),
        "pruned_pct": pruned_pct,
    }


def _run_router_probe(
    model: nn.Module,
    tokenizer,
    probe_text: str,
    max_length: int,
    planned_pruned: dict[int, list[int]],
    prune_args: "PruneArgs",
    simulate: bool = True,
):
    """
    Run a single forward pass with output_router_logits and report selection stats
    before and after simulating zero-out/dropping on the planned pruned indices
    (simulate=True). If simulate=False, report actual routing on the current model
    state (useful after applying zero-out in-memory during dry run).
    """
    device = next(model.parameters()).device
    inputs = tokenizer(
        probe_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs, output_router_logits=True)
    router_logits = outputs.router_logits
    if not router_logits:
        logger.warning("Router logits not returned; skipping router probe.")
        return
    model_attr = maybe_resolve_model_attrs(model)
    if model_attr is None:
        logger.warning("Unknown model class for router probe; skipping.")
        return
    for layer, logits in enumerate(router_logits):
        pruned = planned_pruned.get(layer, [])
        if not pruned:
            continue
        moe = get_moe(model, layer)
        model_attr = resolve_model_attrs(model, moe)
        router = getattr(moe, model_attr["router"])
        top_k = getattr(moe, "top_k", getattr(moe, "num_experts_per_tok", None))
        if simulate:
            before = _summarize_router_logits(logits, top_k, pruned_indices=[])
            if prune_args.zero_out:
                masked_logits = logits.clone()
                if getattr(router, "bias", None) is not None:
                    masked_logits[:, pruned] = torch.finfo(masked_logits.dtype).min
                else:
                    masked_logits[:, pruned] = 0
                after = _summarize_router_logits(
                    masked_logits, top_k, pruned_indices=pruned
                )
            else:
                # simulate physical removal by dropping columns
                mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
                mask[pruned] = False
                masked_logits = logits[:, mask]
                after = _summarize_router_logits(
                    masked_logits, top_k, pruned_indices=[]
                )
            logger.info(
                f"[ROUTER PROBE] layer {layer}: "
                f"before top {before['top_experts']} | pruned_topk={before['pruned_selected']} ({before['pruned_pct']:.2f}%) ; "
                f"after top {after['top_experts']} | pruned_topk={after['pruned_selected']} ({after['pruned_pct']:.2f}%)"
            )
        else:
            actual = _summarize_router_logits(logits, top_k, pruned_indices=pruned)
            logger.info(
                f"[ROUTER PROBE actual] layer {layer}: top {actual['top_experts']} | "
                f"pruned_topk={actual['pruned_selected']} ({actual['pruned_pct']:.2f}%)"
            )


def _sync_config_num_experts_if_uniform(model: nn.Module):
    """
    Align model.config num_experts with the current modules if uniform across layers.
    Raises if counts differ, because downstream loaders (e.g., vLLM) assume a single
    value and will mis-shape modules otherwise.
    """
    model_attr = maybe_resolve_model_attrs(model)
    if not model_attr:
        return
    num_attr = model_attr["num_experts"]
    try:
        counts = []
        for i in range(len(model.model.layers)):
            moe = get_moe(model, i)
            layer_attrs = resolve_model_attrs(model, moe)
            experts = getattr(moe, layer_attrs["experts"])
            if layer_attrs.get("fused", False):
                if hasattr(experts, "num_experts"):
                    count = int(getattr(experts, "num_experts"))
                elif hasattr(moe, layer_attrs["num_experts"]):
                    count = int(getattr(moe, layer_attrs["num_experts"]))
                else:
                    router = getattr(moe, layer_attrs["router"])
                    count = int(getattr(router, "num_experts", router.weight.size(0)))
            else:
                count = len(experts)
            counts.append(count)
        # counts = []
        # for i in range(len(model.model.layers)):
        #     moe = get_moe(model, i)
        #     experts = getattr(moe, model_attr["experts"])
        #     if model_attr.get("fused", False):
        #         if hasattr(experts, "num_experts"):
        #             count = getattr(experts, "num_experts")
        #         elif hasattr(moe, num_attr):
        #             count = getattr(moe, num_attr)
        #         else:
        #             router = getattr(moe, model_attr["router"])
        #             count = getattr(router, "num_experts", router.weight.size(0))
        #     else:
        #         count = len(experts)
        #     counts.append(count)
    except Exception:
        return
    if len(set(counts)) == 1:
        new_num = counts[0]
        if hasattr(model.config, num_attr):
            setattr(model.config, num_attr, new_num)
            logger.info(f"Synced config.{num_attr} to {new_num} (uniform across layers).")
    else:
        raise ValueError(
            "Non-uniform expert counts across layers after prune/merge; "
            "vLLM and other loaders expect a single config.num_experts. "
            "Ensure hybrid plan produces uniform expert counts."
        )


def merge(
    model: nn.Module,
    cluster_labels: dict[int, torch.Tensor],
    observer_data: dict[int, dict[str, Any]],
    merge_args: MergeArgs,
):
    """Merge experts based on the clustering results."""
    logger.info(f"Merging experts using method '{merge_args.merge_method}'")

    try:
        merge_method = MergeMethod(merge_args.merge_method)
    except ValueError:
        raise NotImplementedError(
            f"Merge method '{merge_args.merge_method}' is not implemented. "
            f"Supported methods: {[method.value for method in MergeMethod]}"
        )

    for layer_idx, layer in enumerate(tqdm(cluster_labels, "Merging layers...")):
        if merge_args.skip_first and layer_idx == 0:
            logger.info(
                f"Skipping merging for layer {layer_idx} as per 'skip_first' argument."
            )
            continue

        if merge_args.skip_last and layer_idx == len(cluster_labels) - 1:
            logger.info(
                f"Skipping merging for layer {layer_idx} as per 'skip_last' argument."
            )
            continue

        expert_proba = (
            observer_data[layer]["expert_frequency"]
            / observer_data[layer]["total_tokens"]
        )
        cluster_label = cluster_labels[layer]
        moe = get_moe(model, layer)
        model_attrs = resolve_model_attrs(model, moe)
        merger = MoEExpertMerger(
            moe=moe,
            cluster_label=cluster_label,
            expert_proba=expert_proba,
            model_attrs=model_attrs,
            merge_method=merge_method,
            dom_as_base=merge_args.dom_as_base,
            select_top_k=merge_args.select_top_k,
            permute=merge_args.permute,
            tie_tensors=merge_args.save_as_tied_params,
        )
        merger.merge_experts()
        # in case of non-uniform compression, update num_experts
        # TODO deal with router too
        # setattr(getattr(moe, model_attrs["num_experts"]), model_attrs["num_experts"], len(cluster_label.unique()))
        assert_merge(model, moe, cluster_label)


def save_merged_model(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    merged_model_dir: pathlib.Path,
    safe_serialization,
) -> pathlib.Path:
    logger.info("Saving merged model...")
    merged_model_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    model.save_pretrained(merged_model_dir, safe_serialization=safe_serialization)
    tokenizer.save_pretrained(merged_model_dir)
    end = time.time()
    logger.info(
        f"Merged model saved to {merged_model_dir} in {end - start:.2f} seconds"
    )
    return merged_model_dir


@torch.no_grad()
def smoke_test(model: nn.Module, tokenizer: AutoTokenizer):
    """Run a smoke test to ensure the model is functioning correctly."""
    prompt = "What is your name?"
    test_input = [
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        test_input,
        return_tensors="pt",
        add_generation_prompt=True,
        tokenize=True,
        # enable_thinking=False,
    ).to(model.device)
    outputs = model.generate(
        inputs,
        max_new_tokens=50,
        do_sample=True,
    )
    response = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    logger.info("Smoke test response: %s", response[0])


def get_model_dir(
    results_dir,
    num_clusters,
    cluster_labels,
    cluster_args,
    obs_args,
    merge_args,
    prune_args,
    seed: int,
) -> pathlib.Path:
    cluster_desc = cluster_args.cluster_description
    num_clusters_repr = "per_layer" if isinstance(num_clusters, dict) else num_clusters
    if not cluster_desc:
        cluster_desc = (
            f"{cluster_args.expert_sim}_{obs_args.distance_measure}_{num_clusters_repr}_"
            f"{cluster_args.linkage_method}_freq-penalty-{cluster_args.frequency_penalty}"
            f"_softmax-{cluster_args.softmax_temperature}_multi_layer-{cluster_args.multi_layer}"
        )
        if cluster_args.max_cluster_size is not None:
            cluster_desc += f"_max_size-{cluster_args.max_cluster_size}"
    merge_model_subdir_name = merge_args.merged_model_dir_name
    
    if not merge_model_subdir_name:
        merge_model_subdir_name = f"{merge_args.merge_method}-permute_{merge_args.permute}-skip_first_{merge_args.skip_first}-skip_last_{merge_args.skip_last}-multilayer_{cluster_args.multi_layer}"
        if merge_args.hybrid_prune_layers:
            merge_model_subdir_name += f"-hybrid_layers_{len(merge_args.hybrid_prune_layers)}"

    # Check for non uniform compression
    non_uniform_cluster_labels = (
        len(
            torch.unique(
                torch.tensor(
                    [
                        len(torch.unique(clusters))
                        for clusters in cluster_labels.values()
                    ]
                )
            )
        )
        > 1
    )
    is_hybrid = bool(merge_args.hybrid_prune_layers) or ("hybrid" in str(cluster_desc).lower()) or ("hybrid" in str(merge_model_subdir_name).lower())
    if is_hybrid:
        ratio_tag = cluster_args.compression_ratio
        if ratio_tag is None:
            ratio_tag = "per_layer"
        if merge_args.hybrid_prune_layers:
            layer_tag = str_to_directory_name("_".join(map(str, merge_args.hybrid_prune_layers)))
        else:
            layer_tag = "no_prune"
        merge_model_parent_dir_name = "hybrid_models"
        merge_model_subdir_name = merge_args.merged_model_dir_name
        if not merge_model_subdir_name:
            merge_model_subdir_name = f"{merge_args.merge_method}_{prune_args.prune_method}_seed_{seed}_{ratio_tag}_{layer_tag}"
    elif (
        non_uniform_cluster_labels
        or cluster_args.multi_layer
        or merge_args.skip_first
        or merge_args.skip_last
    ):
        logger.info("Detected non-uniform compression across layers.")
        merge_model_parent_dir_name = "non_uniform_merged_models"
    else:
        merge_model_parent_dir_name = "merged_models"

    merged_model_dir = (
        results_dir
        / merge_model_parent_dir_name
        / merge_model_subdir_name
        / cluster_desc
    )
    return merged_model_dir


def dump_args_to_yaml(
    merged_model_dir: pathlib.Path,
    reap_args: ReapArgs,
    model_args: ModelArgs,
    ds_args: DatasetArgs,
    obs_args: ObserverArgs,
    cluster_args: ClusterArgs,
    kd_args: KdArgs,
    eval_args: EvalArgs,
    merge_args: MergeArgs,
    prune_args: PruneArgs,
):
    """Dump all arguments to a YAML file."""
    all_args = {
        "reap_args": dataclasses.asdict(reap_args),
        "model_args": dataclasses.asdict(model_args),
        "ds_args": dataclasses.asdict(ds_args),
        "obs_args": dataclasses.asdict(obs_args),
        "cluster_args": dataclasses.asdict(cluster_args),
        "kd_args": dataclasses.asdict(kd_args),
        "eval_args": dataclasses.asdict(eval_args),
        "merge_args": dataclasses.asdict(merge_args),
        "prune_args": dataclasses.asdict(prune_args),
    }

    def convert_paths_to_str(data):
        if isinstance(data, dict):
            return {k: convert_paths_to_str(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [convert_paths_to_str(i) for i in data]
        elif isinstance(data, pathlib.Path):
            return str(data)
        else:
            return data

    serializable_args = convert_paths_to_str(all_args)

    output_path = merged_model_dir / "reap_args.yaml"
    with open(output_path, "w") as f:
        yaml.dump(serializable_args, f, default_flow_style=False)
    logger.info(f"All arguments saved to {output_path}")


def _log_next_token_probe(
    model: nn.Module,
    tokenizer,
    probe_text: str,
    max_length: int,
    tag: str,
):
    """
    Log the top next-token predictions for the given probe text.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(
        probe_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    logits = outputs.logits  # (bsz, seq, vocab)
    next_logits = logits[:, -1, :]
    probs = torch.softmax(next_logits, dim=-1)
    top = torch.topk(probs, k=5, dim=-1)
    tokens = [tokenizer.convert_ids_to_tokens(i) for i in top.indices[0].tolist()]
    logger.info(
        f"[OUTPUT PROBE {tag}] top tokens: {list(zip(tokens, top.values[0].tolist()))}"
    )


def main():
    (
        reap_args,
        model_args,
        ds_args,
        obs_args,
        cluster_args,
        kd_args,
        eval_args,
        merge_args,
        prune_args,
    ) = parse_args()
    # default hybrid to zero-out so slot count stays consistent with merging
    if len(merge_args.hybrid_prune_layers) > 0:
        prune_args.zero_out = True
    set_seed(reap_args.seed)
    results_dir = create_results_directory(model_args.model_name, ds_args.dataset_name)

    if cluster_args.singleton_super_experts and cluster_args.singleton_outlier_experts:
        raise ValueError(
            "Both 'singleton_super_experts' in clustering and 'perserve_super_experts' in merging cannot be set to True."
        )
    # get local patched model if req'd
    model_name = patched_model_map(model_args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        # local_files_only=True,
    )

    # record activations or load previously recorded activations
    logger.info(
        f"Running observer to collect activation data for model {model_args.model_name} on dataset {ds_args.dataset_name}."
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

    # Decide hybrid mode early (used for dry-run decision)
    hybrid_mode = len(merge_args.hybrid_prune_layers) > 0

    # If we're going to do a dry-run and exit, keep the current dispatched model (no saving)
    if hybrid_mode and merge_args.hybrid_dry_run:
        pass
    else:
        # (optional) move observer_data tensors to CPU so we can free GPU cleanly
        for layer in observer_data:
            for k, v in list(observer_data[layer].items()):
                if torch.is_tensor(v):
                    observer_data[layer][k] = v.cpu()

        # Remove accelerate offload hooks and drop the dispatched model
        remove_hook_from_module(model, recurse=True)
        del model
        torch.cuda.empty_cache()
        gc.collect()

        # Reload WITHOUT device_map (no offload hooks) for any shape-changing ops + saving
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=None,          # IMPORTANT (no dispatch/offload)
            torch_dtype="auto",
            trust_remote_code=True,
            # local_files_only=True,
        )
        model.eval()

    # plan hybrid prune/merge if requested
    hybrid_mode = len(merge_args.hybrid_prune_layers) > 0
    pruned_experts_info: dict[str, list[int]] = {}
    logger.info("Start of clustering")
    if hybrid_mode:
        prune_plan, per_layer_clusters = plan_hybrid_prune_merge(
            observer_data, cluster_args, merge_args
        )
        if merge_args.hybrid_dry_run:
            logger.info("Hybrid dry run requested. Reporting planned changes and exiting.")
            planned_pruned = _preview_experts_to_prune(
                observer_data, prune_plan, prune_args
            )
            for layer, pruned in planned_pruned.items():
                num_experts = observer_data[layer]["expert_frequency"].shape[0]
                moe = get_moe(model, layer)
                top_k = getattr(moe, "top_k", getattr(moe, "num_experts_per_tok", None))
                planned_count = len(pruned)
                logger.info(
                    f"[PRUNE] layer {layer}: experts={num_experts}, top_k={top_k}, "
                    f"will prune {planned_count} -> "
                    f"{'zero-out (routing still over ' + str(num_experts) + ')' if prune_args.zero_out else str(num_experts - planned_count) + ' experts'}; "
                    f"indices: {pruned}"
                )
            for layer, clusters in sorted(per_layer_clusters.items()):
                num_experts = observer_data[layer]["expert_frequency"].shape[0]
                logger.info(
                    f"[MERGE] layer {layer}: experts={num_experts} -> clusters={clusters}"
                )
            if merge_args.hybrid_router_probe_text:
                logger.info(
                    f"Running router probe with text: '{merge_args.hybrid_router_probe_text}' "
                    f"(max_length={merge_args.hybrid_router_probe_max_length})"
                )
                _log_next_token_probe(
                    model,
                    tokenizer,
                    merge_args.hybrid_router_probe_text,
                    merge_args.hybrid_router_probe_max_length,
                    tag="baseline",
                )
                # Simulated before/after on unmodified model
                _run_router_probe(
                    model,
                    tokenizer,
                    merge_args.hybrid_router_probe_text,
                    merge_args.hybrid_router_probe_max_length,
                    planned_pruned,
                    prune_args,
                    simulate=True,
                )
                # Optionally apply in-memory pruning for a true zero-out probe
                if prune_plan and prune_args.zero_out:
                    logger.info("Applying in-memory zero-out for probe.")
                    apply_layerwise_pruning(
                        model, observer_data, prune_plan, prune_args
                    )
                    _log_next_token_probe(
                        model,
                        tokenizer,
                        merge_args.hybrid_router_probe_text,
                        merge_args.hybrid_router_probe_max_length,
                        tag="zero_out",
                    )
                _run_router_probe(
                    model,
                    tokenizer,
                    merge_args.hybrid_router_probe_text,
                    merge_args.hybrid_router_probe_max_length,
                    planned_pruned,
                    prune_args,
                    simulate=not prune_args.zero_out,  # if zero_out applied, report actual routing
                )
            return
        if prune_plan:
            logger.info("Applying hybrid pruning...")
            pruned_experts_info = apply_layerwise_pruning(
                model, observer_data, prune_plan, prune_args
            )
        num_clusters = per_layer_clusters
    else:
        num_clusters = cluster_args.num_clusters
        if num_clusters is None:
            if cluster_args.compression_ratio is None:
                raise ValueError(
                    "Either num_clusters or compression_ratio must be set for clustering."
                )
            else:
                # Calculate num_clusters from compression_ratio
                if not merge_args.skip_first and not merge_args.skip_last:
                    total_experts = len(
                        observer_data[next(iter(observer_data))]["expert_frequency"]
                    )
                    num_clusters = int(
                        total_experts * (1 - cluster_args.compression_ratio)
                    )
                else:
                    # If skipping first or last layer, adjust total_experts accordingly
                    experts_per_layer = len(
                        observer_data[next(iter(observer_data))]["expert_frequency"]
                    )
                    layers = len(observer_data)
                    total_experts = layers * experts_per_layer
                    total_clusters = int(
                        total_experts * (1 - cluster_args.compression_ratio)
                    )
                    total_layers = len(observer_data)
                    if merge_args.skip_first:
                        total_layers -= 1
                    if merge_args.skip_last:
                        total_layers -= 1
                    num_clusters = int(total_clusters / total_layers)
                logger.info(
                    f"Calculated num_clusters: {num_clusters} from compression_ratio: {cluster_args.compression_ratio}"
                )

    cluster_labels = cluster(
        observer_data,
        num_clusters,
        cluster_args,
        obs_args.distance_measure,
        results_dir,
    )
    logger.info("Clustering completed.")

    # merging
    logging.info("Start of merging")
    merged_model_dir = get_model_dir(
        results_dir,
        num_clusters,
        cluster_labels,
        cluster_args,
        obs_args,
        merge_args,
        prune_args,
        reap_args.seed,
    )
    if (
        merged_model_dir.exists()
        and list(merged_model_dir.glob("*.safetensors"))
        and not merge_args.overwrite_merged_model
    ):
        logger.info(
            f"Merged model files already exist in {merged_model_dir}. Skipping merging."
        )
    else:
        merge(
            model,
            cluster_labels,
            # num_clusters,
            observer_data,
            merge_args,
        )
        logger.info("Merging completed.")
        if hybrid_mode and not prune_args.zero_out:
            pruned_experts_info = prune_to_cluster_representatives(
                model, cluster_labels, pruned_experts_info
            )
        _sync_config_num_experts_if_uniform(model)
        logger.info("Saving merged model...")
        merged_model_dir = save_merged_model(
            model,
            tokenizer,
            merged_model_dir,
            safe_serialization=True if not merge_args.save_as_tied_params else False,
        )
        logger.info(f"Merged model saved to {merged_model_dir}.")

        # save clustering results
        logger.info("Saving clustering results...")
        cluster_analysis_dir = merged_model_dir / "clusters"
        cluster_analysis_dir.mkdir(parents=True, exist_ok=True)
        with open(cluster_analysis_dir / "clusters.pkl", "wb") as f:
            pickle.dump(cluster_labels, f)

        if pruned_experts_info:
            pruned_path = merged_model_dir / "pruned_experts.json"
            with open(pruned_path, "w") as f:
                json.dump(pruned_experts_info, f, indent=2)
            logger.info(f"Pruned experts indices saved to {pruned_path}")

        if reap_args.plot_clusters:
            logger.info("Plotting clusters analysis...")
            plot_cluster_analysis(
                cluster_labels,
                cluster_analysis_dir,
                merge_args.skip_first,
                merge_args.skip_last,
            )
        logger.info(
            f"Clustering results saved to {cluster_analysis_dir}"
        )

        # smoke test
        if reap_args.smoke_test:
            logger.info("Running smoke test on the merged model...")
            try:
                smoke_test(model, tokenizer)
            except Exception as e:
                logger.error(f"Smoke test failed: {e}")
                pass

        dump_args_to_yaml(
            merged_model_dir,
            reap_args,
            model_args,
            ds_args,
            obs_args,
            cluster_args,
            kd_args,
            eval_args,
            merge_args,
            prune_args,
        )

        if model_name == "artifacts/models/GLM-4.5-Air":
            # move modelling file
            source_file = pathlib.Path(model_name) / "modeling_glm4_moe.py"
            target_file = merged_model_dir / "modeling_glm4_moe.py"
            if source_file.exists():
                shutil.copy2(source_file, target_file)
                logger.info(f"Copied modeling_glm4_moe.py to {merged_model_dir}")
            else:
                raise RuntimeError(
                    f"Source file {source_file} does not exist. Cannot copy to {target_file}."
                )

    # eval
    if reap_args.do_eval:
        remove_hook_from_module(model, recurse=True)
        model.to("cpu")
        del model
        del observer_data
        del cluster_labels
        torch.cuda.empty_cache()
        gc.collect()
        model_args.model_name = merged_model_dir
        results_subdir = "eval_vllm" if eval_args.use_server else "eval_hf"
        run_evaluate(model_args, merged_model_dir / results_subdir, eval_args, reap_args.seed)


if __name__ == "__main__":
    main()
