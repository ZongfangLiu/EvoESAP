from __future__ import annotations
import time
import logging
import dataclasses
import pathlib
import time
from typing import Any
import gc
import yaml
import json

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser

from accelerate.utils import set_seed
from accelerate.hooks import remove_hook_from_module


from reap.main import record_activations, smoke_test, create_results_directory
from reap.args import (
    ReapArgs,
    ModelArgs,
    EvalArgs,
    PruneArgs,
    ObserverArgs,
    DatasetArgs,
    ClusterArgs,
)
from reap.data import DATASET_REGISTRY
from reap.cluster import (
    get_penalty_vector,
    hierarchical_clustering,
    dynamic_frequency_penalized_clustering,
)
from reap.model_util import (
    assert_merge,
    get_moe,
    get_super_expert_indices,
    patched_model_map,
    resolve_model_attrs,
)
from reap.eval import run_evaluate
import shutil

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def dump_args_to_yaml(
    pruned_model_dir: pathlib.Path,
    reap_args: ReapArgs,
    ds_args: DatasetArgs,
    obs_args: ObserverArgs,
    model_args: ModelArgs,
    eval_args: EvalArgs,
    prune_args: PruneArgs,
    cluster_args: ClusterArgs,
    search_args=None,
):
    """Dump all arguments to a YAML file."""
    all_args = {
        "reap_args": dataclasses.asdict(reap_args),
        "ds_args": dataclasses.asdict(ds_args),
        "obs_args": dataclasses.asdict(obs_args),
        "model_args": dataclasses.asdict(model_args),
        "eval_args": dataclasses.asdict(eval_args),
        "prune_args": dataclasses.asdict(prune_args),
        "cluster_args": dataclasses.asdict(cluster_args),
    }
    if search_args is not None:
        all_args["search_args"] = dataclasses.asdict(search_args)

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

    output_path = pruned_model_dir / "reap_args.yaml"
    with open(output_path, "w") as f:
        yaml.dump(serializable_args, f, default_flow_style=False)
    logger.info(f"All arguments saved to {output_path}")


def prune(
    observer_data,
    model,
    tokenizer,
    reap_args,
    prune_args,
    n_experts_to_prune,
    pruned_model_dir,
):
    """
    Prune the model based on the observer data and clustering.
    """
    model_attrs = None
    pruned_experts_info: dict[str, list[int]] = {}

    for layer in observer_data:
        if "expert_proba" not in observer_data[layer]:
            # Calculate expert probabilities if not already present
            observer_data[layer]["expert_proba"] = (
                observer_data[layer]["expert_frequency"]
                / observer_data[layer]["total_tokens"]
            )

    if prune_args.perserve_super_experts or prune_args.perserve_outliers:
        super_expert_idx = get_super_expert_indices(observer_data, include_last_layers=prune_args.perserve_outliers)
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
            super_experts_in_layer = super_expert_idx[super_expert_idx[:, 0] == layer][:, 1]
            if len(super_experts_in_layer) > 0:
                for metric in metrics:
                    if metric in observer_data[layer]:
                        observer_data[layer][metric][super_experts_in_layer] = float("inf")

    for layer in tqdm(observer_data, "Pruning layers..."):
        num_experts = observer_data[layer]["expert_frequency"].shape[0]
        if prune_args.prune_method == "ean_ca":
            ean = torch.zeros(num_experts, device=model.device, dtype=torch.float32)
            for i in range(num_experts):
                ean[i] = torch.linalg.norm(
                    observer_data[layer]["routed_characteristic_activation"][i], dim=-1
                ).sum()
            _, experts_to_prune = torch.topk(ean, n_experts_to_prune, largest=False)
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
                saliency_data, n_experts_to_prune, largest=False
            )

        retained_expert_indicies = [
            i for i in range(num_experts) if i not in experts_to_prune
        ]
        pruned_experts_info[str(layer)] = [int(idx) for idx in experts_to_prune.tolist()]
        # prune experts (physical removal)
        moe = get_moe(model, layer)
        model_attrs = resolve_model_attrs(model, moe)
        if not model_attrs["fused"]:
            all_experts = getattr(moe, model_attrs["experts"])
            retained_experts = [all_experts[i] for i in retained_expert_indicies]
            retained_experts = torch.nn.ModuleList(retained_experts)
            setattr(moe, model_attrs["experts"], retained_experts)

            # prune router
            router = getattr(moe, model_attrs["router"])
            router.weight.data = router.weight.data[retained_expert_indicies, :]
            if getattr(router, "bias", None):
                router.bias.data = router.bias.data[retained_expert_indicies]
            router.out_features = len(retained_expert_indicies)
            if hasattr(router, "e_score_correction_bias"):
                router.e_score_correction_bias.data = (
                    router.e_score_correction_bias.data[retained_expert_indicies]
                )
            setattr(moe, model_attrs["router"], router)
            # keep module metadata in sync to avoid routing past the pruned list
            if hasattr(moe, model_attrs["num_experts"]):
                setattr(moe, model_attrs["num_experts"], len(retained_expert_indicies))
            if hasattr(router, "num_experts"):
                router.num_experts = len(retained_expert_indicies)
        else:
            experts = getattr(moe, model_attrs["experts"])
            gate_proj_attr = model_attrs["gate_proj"]
            down_proj_attr = model_attrs["down_proj"]
            gate_proj = getattr(experts, gate_proj_attr)
            down_proj = getattr(experts, down_proj_attr)
            gate_proj.data = gate_proj.data[retained_expert_indicies]
            down_proj.data = down_proj.data[retained_expert_indicies]
            setattr(experts, gate_proj_attr, gate_proj)
            setattr(experts, down_proj_attr, down_proj)
            if hasattr(experts, "num_experts"):
                experts.num_experts = len(retained_expert_indicies)
            if hasattr(moe, model_attrs["num_experts"]):
                setattr(moe, model_attrs["num_experts"], len(retained_expert_indicies))
            elif hasattr(moe, "num_experts"):
                moe.num_experts = len(retained_expert_indicies)

            router = getattr(moe, model_attrs["router"])
            if hasattr(router, "weight") and router.weight is not None:
                router.weight.data = router.weight.data[retained_expert_indicies]
            if getattr(router, "bias", None) is not None:
                router.bias.data = router.bias.data[retained_expert_indicies]
            if hasattr(router, "out_features"):
                router.out_features = len(retained_expert_indicies)
            if hasattr(router, "num_experts"):
                router.num_experts = len(retained_expert_indicies)
            if hasattr(router, "top_k"):
                router.top_k = min(int(router.top_k), len(retained_expert_indicies))
            if hasattr(moe, "top_k"):
                moe.top_k = min(int(moe.top_k), len(retained_expert_indicies))
            if hasattr(router, "e_score_correction_bias"):
                router.e_score_correction_bias.data = (
                    router.e_score_correction_bias.data[retained_expert_indicies]
                )
            if hasattr(router, "moe_statics") and hasattr(
                router.moe_statics, "e_score_correction_bias"
            ):
                statics_bias = router.moe_statics.e_score_correction_bias
                if statics_bias is not None:
                    if statics_bias.data.ndim == 2:
                        statics_bias.data = statics_bias.data[:, retained_expert_indicies]
                    else:
                        statics_bias.data = statics_bias.data[retained_expert_indicies]
            setattr(moe, model_attrs["router"], router)

    # patch config and dump
    logger.info("Saving pruned model...")
    retained_experts = len(retained_expert_indicies)
    setattr(model.config, model_attrs["num_experts"], retained_experts)
    if model.__class__.__name__ == "Ernie4_5_MoeForCausalLM":  # remote-code verson
        model.config.moe_capacity = [
            retained_experts,
            retained_experts,
            retained_experts,
        ]

    pruned_model_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    model.save_pretrained(pruned_model_dir)
    pruned_experts_path = pruned_model_dir / "pruned_experts.json"
    with open(pruned_experts_path, "w") as f:
        json.dump(pruned_experts_info, f, indent=2)
    logger.info(f"Pruned experts indices saved to {pruned_experts_path}")
    end = time.time()
    logger.info(
        f"Pruned model saved to {pruned_model_dir} in {end - start:.2f} seconds"
    )
    return pruned_model_dir


def get_pruned_model_dir(
    results_dir,
    n_experts_to_prune: str,
    total_experts: int,
    prune_args,
    seed: int,
    renorm: bool,
) -> pathlib.Path:
    # keep enough precision so 0.125 doesn't round to 0.12
    compression_ratio_str = f"{(n_experts_to_prune / total_experts):.3f}"
    pruned_model_name = f"{prune_args.prune_method}"
    if prune_args.perserve_super_experts:
        pruned_model_name += "-perserve_super"
    elif prune_args.perserve_outliers:
        pruned_model_name += "-perserve_outlier"
    if renorm:
        pruned_model_name += f"-renorm_{str(renorm).lower()}"
    pruned_model_name += f"-seed_{seed}"
    pruned_model_name += f"-{compression_ratio_str}"
    pruned_model_dir = results_dir / "pruned_models" / pruned_model_name
    logger.info(f"Using seed {seed}, pruned model dir: {pruned_model_dir}")
    return pruned_model_dir


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
        )
    )
    reap_args, ds_args, obs_args, model_args, eval_args, prune_args, cluster_args = (
        parser.parse_args_into_dataclasses()
    )
    if prune_args.perserve_super_experts and prune_args.perserve_outliers:
        raise ValueError("Only one of perserve_super_experts or perserve_outliers can be set to True.")
    set_seed(reap_args.seed)
    results_dir = create_results_directory(model_args.model_name, ds_args.dataset_name)

    # get local patched model if req'd
    model_name = patched_model_map(model_args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    # Log dtype/device to verify how parameters were loaded
    first_param = next(model.parameters())
    logger.info(f"Loaded model param dtype: {first_param.dtype}, device: {first_param.device}")
    # Inspect a specific expert parameter if it exists
    try:
        target_param = model.model.layers[0].mlp.experts[0].down_proj.weight
        logger.info(
            "model.layers.0.mlp.experts.0.down_proj.weight "
            f"dtype: {target_param.dtype}, device: {target_param.device}, shape: {tuple(target_param.shape)}"
        )
    except Exception as e:
        logger.warning(f"Could not access model.layers.0.mlp.experts.0.down_proj.weight: {e}")
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

    
    # >>> INSERT HERE (before "Start of pruning") <<<
    # (1) move observer_data tensors to cpu (optional but good)
    for layer in observer_data:
        for k, v in list(observer_data[layer].items()):
            if torch.is_tensor(v):
                observer_data[layer][k] = v.cpu()

    # (2) kill accelerate offload hooks + free the dispatched model
    remove_hook_from_module(model, recurse=True)
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # (3) reload WITHOUT device_map so no offload hook exists during prune/save
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        trust_remote_code=True,
        local_files_only=True,
        device_map=None,          # IMPORTANT
    )
    model.eval()
    # <<< END INSERT >>>
    # pruning
    logger.info("Start of pruning")
    n_experts_to_prune = prune_args.n_experts_to_prune
    if n_experts_to_prune is None:
        if cluster_args.compression_ratio is None:
            raise ValueError(
                "Either n_experts_to_prune or compression_ratio must be set for pruning."
            )
        else:
            # Calculate n_experts_to_prune from compression_ratio
            total_experts = len(
                observer_data[next(iter(observer_data))]["expert_frequency"]
            )
            n_experts_to_prune = int(total_experts * cluster_args.compression_ratio)
            logger.info(
                f"Calculated n_experts to prune: {n_experts_to_prune} from compression_ratio: {cluster_args.compression_ratio}"
            )

    pruned_model_dir = get_pruned_model_dir(
        results_dir, n_experts_to_prune, total_experts, prune_args, reap_args.seed, obs_args.renormalize_router_weights
    )
    if (
        pruned_model_dir.exists()
        and list(pruned_model_dir.glob("*.safetensors"))
        and not prune_args.overwrite_pruned_model
    ):
        logger.info(
            f"Pruned model directory {pruned_model_dir} already exists and contains pruned model files. "
            "Skipping pruning step."
        )
    else:
        logger.info(f"Pruning model to {total_experts - n_experts_to_prune} experts...")
        prune(
            observer_data,
            model,
            tokenizer,
            reap_args,
            prune_args,
            n_experts_to_prune,
            pruned_model_dir,
        )
        logger.info("pruning completed.")

        # smoke test
        if reap_args.smoke_test:
            logger.info("Running smoke test on the merged model...")
            try:
                smoke_test(model, tokenizer)
            except Exception as e:
                logger.error(f"Smoke test failed: {e}")
                pass

        tokenizer.save_pretrained(pruned_model_dir)
        if model_name == "artifacts/models/GLM-4.5-Air":
            # move modelling file
            source_file = pathlib.Path(model_name) / "modeling_glm4_moe.py"
            target_file = pruned_model_dir / "modeling_glm4_moe.py"
            if source_file.exists():
                shutil.copy2(source_file, target_file)
                logger.info(f"Copied modeling_glm4_moe.py to {pruned_model_dir}")
            else:
                raise RuntimeError(
                    f"Source file {source_file} does not exist. Cannot copy to {target_file}."
                )

        logger.info("Pruning completed.")

        dump_args_to_yaml(
            pruned_model_dir,
            reap_args,
            ds_args,
            obs_args,
            model_args,
            eval_args,
            prune_args,
            cluster_args,
        )

    # eval
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
