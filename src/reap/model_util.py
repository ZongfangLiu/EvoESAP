import logging
from typing import Any, Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


MODEL_ATTRS = {
    "Qwen3MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Qwen3-Coder-30B-A3B-Instruct": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "NonUniformQwen3MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Llama4ForCausalLM": {
        "moe_block": "feed_forward",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": True,
        "router": "gate",
        "num_experts": "num_local_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "MixtralForCausalLM": {
        "moe_block": "block_sparse_moe",
        "gate_proj": "w3",
        "up_proj": "w1",
        "down_proj": "w2",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_local_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "DeepseekV2ForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Ernie4_5_MoEForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "moe_num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Ernie4_5_MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "moe_num_experts",
        "num_experts_per_tok": "moe_k",
    },
    "NonUniformErnie4_5_MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "moe_num_experts",
        "num_experts_per_tok": "moe_k",
    },
    # "gpt-oss-20b": {
    #     "moe_block": "mlp",
    #     "gate_proj": "gate_proj",
    #     "up_proj": "up_proj",
    #     "down_proj": "down_proj",
    #     "experts": "experts",
    #     "fused": True,
    #     "router": "gate",
    #     "num_experts": "num_experts",
    #     "num_experts_per_tok": "num_experts_per_tok",
    # },
    "GptOssForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": True,
        "router": "router",
        "num_experts": "num_local_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "Glm4MoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "OlmoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "NonUniformOlmoeForCausalLM": {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
}

def _tensor_ndim(value: Any) -> int | None:
    if isinstance(value, nn.Parameter):
        value = value.data
    if torch.is_tensor(value):
        return value.ndim
    return None


def _looks_like_fused_experts(experts: Any) -> bool:
    if experts is None:
        return False
    gate_up_proj = getattr(experts, "gate_up_proj", None)
    down_proj = getattr(experts, "down_proj", None)
    if gate_up_proj is None or down_proj is None:
        return False
    gate_dim = _tensor_ndim(gate_up_proj)
    down_dim = _tensor_ndim(down_proj)
    if gate_dim is None or down_dim is None:
        return False
    return gate_dim >= 3 and down_dim >= 3


def resolve_model_attrs(model: nn.Module, moe: nn.Module | None = None) -> dict[str, Any]:
    base = MODEL_ATTRS[model.__class__.__name__]
    if base.get("fused", False):
        return base
    if moe is None:
        try:
            moe = get_moe(model, 0)
        except Exception:
            moe = None
    experts = None
    if moe is not None:
        try:
            experts = getattr(moe, base["experts"])
        except Exception:
            experts = None
    if not _looks_like_fused_experts(experts):
        return base
    updated = dict(base)
    updated["fused"] = True
    if getattr(experts, "gate_up_proj", None) is not None:
        updated["gate_proj"] = "gate_up_proj"
        updated["up_proj"] = "gate_up_proj"
    if getattr(experts, "down_proj", None) is not None:
        updated["down_proj"] = "down_proj"
    if hasattr(model, "config") and hasattr(model.config, "num_local_experts"):
        updated["num_experts"] = "num_local_experts"
    return updated


def maybe_resolve_model_attrs(
    model: nn.Module, moe: nn.Module | None = None
) -> dict[str, Any] | None:
    if model.__class__.__name__ not in MODEL_ATTRS:
        return None
    return resolve_model_attrs(model, moe)


def get_moe(model, layer):
    moe_attr_name = MODEL_ATTRS.get(model.__class__.__name__)["moe_block"]
    return getattr(model.model.layers[layer], moe_attr_name)


def assert_merge(model, merged_moe, cluster_label):
    model_attr = resolve_model_attrs(model, merged_moe)
    assert hasattr(merged_moe, "experts"), (
        "The merged module must have an 'experts' attribute."
    )

    gate_proj = model_attr["gate_proj"]
    down_proj = model_attr["down_proj"]

    if model_attr["fused"]:
        for cluster_id in cluster_label.unique():
            expert_indices = torch.where(cluster_label == cluster_id)[0]
            dom_expert = expert_indices[0]
            for expert in expert_indices[1:]:
                assert torch.allclose(
                    getattr(merged_moe.experts, gate_proj)[dom_expert],
                    getattr(merged_moe.experts, gate_proj)[expert],
                ), f"Experts {expert_indices} are not merged correctly."
                assert torch.allclose(
                    getattr(merged_moe.experts, down_proj)[dom_expert],
                    getattr(merged_moe.experts, down_proj)[expert],
                ), f"Experts {expert_indices} are not merged correctly."
    else:
        up_proj = model_attr["up_proj"]
        for cluster_id in cluster_label.unique():
            expert_indices = torch.where(cluster_label == cluster_id)[0]
            dom_expert = expert_indices[0]
            for expert in expert_indices[1:]:
                assert (
                    getattr(merged_moe.experts[dom_expert], up_proj).weight
                    == getattr(merged_moe.experts[expert], up_proj).weight
                ).all(), f"Experts {expert_indices} are not merged correctly."
                assert (
                    getattr(merged_moe.experts[dom_expert], down_proj).weight
                    == getattr(merged_moe.experts[expert], down_proj).weight
                ).all(), f"Experts {expert_indices} are not merged correctly."
                assert (
                    getattr(merged_moe.experts[dom_expert], gate_proj).weight
                    == getattr(merged_moe.experts[expert], gate_proj).weight
                ).all(), f"Experts {expert_indices} are not merged correctly."


def patched_model_map(model: str):
    patched = False
    model_name = model

    if model == "deepseek-ai/DeepSeek-V2-Lite-Chat":
        patched = True
        model_name = "artifacts/models/DeepSeek-V2-Lite-Chat"

    # until hf version lands
    if model == "baidu/ERNIE-4.5-21B-A3B-PT":
        patched = True
        # model_name = "artifacts/models/ERNIE-4.5-21B-A3B-PT"
        model_name = "baidu/ERNIE-4.5-21B-A3B-PT"

    if model == "Qwen/NonUniformQwen3-30B-A3B":
        patched = True
        model_name = "artifacts/models/NonUniformQwen3-30B-A3B"

    if model == "zai-org/GLM-4.5-Air":
        patched = True
        model_name = "artifacts/models/GLM-4.5-Air"

    if model == "zai-org/GLM-4.5-Air-FP8":
        patched = True
        model_name = "artifacts/models/GLM-4.5-Air-FP8"

    if model == "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8":
        patched = True
        model_name = "artifacts/models/Qwen3-Coder-480B-A35B-Instruct-FP8"

    if model == "allenai/OLMoE-1B-7B-0125":
        patched = True
        model_name = "allenai/OLMoE-1B-7B-0125"
        
    if model == "allenai/OLMoE-1B-7B-0125-Instruct":
        patched = True
        model_name = "allenai/OLMoE-1B-7B-0125-Instruct"
    
    if model == "RedHatAI/Qwen3-30B-A3B-FP8-dynamic":
        patched = True
        model_name = "RedHatAI/Qwen3-30B-A3B-FP8-dynamic"
        
    if model == "Qwen/Qwen3-30B-A3B-Instruct-2507":
        patched = True
        model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

    if patched:
        logger.info(f"Using patched model for {model} from: {model_name}")
    return model_name


def install_runtime_router_mask(router: nn.Module, indices: Sequence[int] | torch.Tensor):
    """
    Runtime-only mask: prevents routing to the given expert indices without changing shapes.
    Registers a non-persistent buffer and a forward hook that adds dtype-min to pruned logits.
    """
    if not torch.is_tensor(indices):
        indices = torch.tensor(indices, device=router.weight.device, dtype=torch.long)
    else:
        indices = indices.to(router.weight.device)
    if indices.numel() == 0:
        indices = indices.reshape(-1)
    if hasattr(router, "out_features"):
        out_features = int(getattr(router, "out_features"))
    elif hasattr(router, "weight") and router.weight is not None:
        out_features = int(router.weight.shape[0])
    elif hasattr(router, "num_experts"):
        out_features = int(getattr(router, "num_experts"))
    else:
        raise AttributeError(
            f"Router module {router.__class__.__name__} has no out_features/weight/num_experts to infer expert dim."
        )

    # filter invalid indices (e.g., when experts were physically removed)
    valid = indices[(indices >= 0) & (indices < out_features)]
    invalid = indices.numel() - valid.numel()
    if invalid > 0:
        logger.debug(
            "Dropping %d invalid pruned indices (router.out_features=%d)",
            invalid,
            out_features,
        )

    # Zero weights so logits are input-independent; push bias/corrections to dtype-min.
    if valid.numel() > 0 and hasattr(router, "weight") and router.weight is not None:
        router.weight.data[valid] = 0
    if getattr(router, "bias", None) is not None and valid.numel() > 0:
        # router.bias.data[valid] = torch.finfo(router.bias.dtype).min
        router.bias.data[valid] = 0
    if hasattr(router, "e_score_correction_bias") and valid.numel() > 0:
        # router.e_score_correction_bias.data[valid] = torch.finfo(
        #     router.e_score_correction_bias.dtype
        # ).min
        router.e_score_correction_bias.data[valid] = 0

    # Remove any prior hook/buffer and install a non-persistent mask buffer + hook.
    prior = getattr(router, "_prune_mask_hook", None)
    if prior is not None:
        try:
            prior.remove()
        except Exception:
            pass
    router._buffers.pop("prune_mask", None)
    router._parameters.pop("prune_mask", None)

    mask = torch.zeros(
        out_features, device=router.weight.device, dtype=router.weight.dtype
    )
    if valid.numel() > 0:
        mask[valid] = torch.finfo(mask.dtype).min
    router.register_buffer("prune_mask", mask, persistent=False)

    def _mask_fn(module, _, out):
        if isinstance(out, tuple):
            logits = out[0] + module.prune_mask
            return (logits, *out[1:])
        return out + module.prune_mask

    router._prune_mask_hook = router.register_forward_hook(_mask_fn)

    # ERNIE-style correction bias: also mask correction scores so pruned experts
    # can never be selected even if correction bias is positive.
    if hasattr(router, "moe_statics"):
        statics = getattr(router, "moe_statics", None)
        if statics is not None:
            prior = getattr(statics, "_prune_mask_hook", None)
            if prior is not None:
                try:
                    prior.remove()
                except Exception:
                    pass
            statics._buffers.pop("prune_mask", None)
            statics._parameters.pop("prune_mask", None)
            statics_mask = torch.zeros(out_features, device=router.weight.device, dtype=router.weight.dtype)
            if valid.numel() > 0:
                statics_mask[valid] = torch.finfo(statics_mask.dtype).min
            statics.register_buffer("prune_mask", statics_mask, persistent=False)

            def _mask_statics(module, _, out):
                return out + module.prune_mask

            statics._prune_mask_hook = statics.register_forward_hook(_mask_statics)

# This is wrong, for olmoe in new transformer v5.0, we can't directly get the logits, we should modify the forward function of router.
def apply_runtime_router_masks(model: nn.Module, pruned_experts_info: dict[str, list[int]]):
    """
    Given a mapping of layer -> pruned indices, install runtime-only masks on each router.
    """
    model_attrs = MODEL_ATTRS[model.__class__.__name__]
    for layer_str, pruned_indices in pruned_experts_info.items():
        layer_idx = int(layer_str)
        moe = get_moe(model, layer_idx)
        router = getattr(moe, model_attrs["router"])
        install_runtime_router_mask(router, pruned_indices)
        setattr(moe, model_attrs["router"], router)


def verify_runtime_pruning(model: nn.Module, pruned_experts_info: dict[str, list[int]]):
    """
    Sanity-check that runtime masks/zeroing are present for the given pruned experts.
    Logs warnings if anything looks off.
    """
    model_attrs = MODEL_ATTRS[model.__class__.__name__]
    for layer_str, pruned_indices in pruned_experts_info.items():
        layer_idx = int(layer_str)
        moe = get_moe(model, layer_idx)
        router = getattr(moe, model_attrs["router"])
        if not hasattr(router, "prune_mask"):
            logger.warning(
                "Layer %s router missing prune_mask buffer; mask hook may not be installed.",
                layer_idx,
            )
            continue
        mask = router.prune_mask
        bad_mask = [i for i in pruned_indices if i < mask.numel() and mask[i] != torch.finfo(mask.dtype).min]
        if bad_mask:
            logger.warning(
                "Layer %s router mask entries not dtype-min at indices: %s", layer_idx, bad_mask
            )
        if hasattr(router, "weight") and router.weight is not None:
            weight = router.weight
            bad_weight = []
            for i in pruned_indices:
                if i >= weight.size(0):
                    continue
                if torch.any(weight[i].abs() > 0):
                    bad_weight.append(i)
            if bad_weight:
                logger.warning(
                    "Layer %s router weights not zeroed at pruned rows: %s", layer_idx, bad_weight
                )
        # Check experts if present and not fused
        if not model_attrs.get("fused", False) and hasattr(moe, model_attrs["experts"]):
            experts = getattr(moe, model_attrs["experts"])
            bad_experts = []
            for i in pruned_indices:
                if i >= len(experts):
                    continue
                exp_mod = experts[i]
                nonzero = False
                for _, p in exp_mod.named_parameters():
                    if torch.any(p.data.abs() > 0):
                        nonzero = True
                        break
                if nonzero:
                    bad_experts.append(i)
            if bad_experts:
                logger.warning(
                    "Layer %s pruned experts not zeroed: %s", layer_idx, bad_experts
                )
        logger.info("Verified runtime mask/zeroing for layer %s (pruned %s)", layer_idx, pruned_indices)


def assert_tied_weights(model, clusters_labels):
    model_attrs = MODEL_ATTRS.get(model.__class__.__name__)
    for layer_idx in clusters_labels:
        clusters = clusters_labels[layer_idx]
        moe = get_moe(model, layer_idx)
        experts = getattr(moe, model_attrs["experts"])
        for cluster_idx in torch.unique(clusters):
            experts_in_cluster = torch.where(clusters == cluster_idx)[0].tolist()
            dom_expert = experts[experts_in_cluster[0]]
            for attr in ["up_proj", "down_proj", "gate_proj"]:
                for expert_idx in experts_in_cluster:
                    if expert_idx == dom_expert:
                        continue
                    expert = experts[expert_idx]
                    proj = getattr(expert, attr)
                    weight = proj.weight
                    dom_proj = getattr(dom_expert, attr)
                    dom_weight = dom_proj.weight
                    if not torch.allclose(weight, dom_weight):
                        print(
                            f"Weights for expert {expert_idx} in cluster {cluster_idx} for layer {layer_idx} and attr {attr} are not tied!"
                        )
                        print(f"Max diff: {torch.abs(weight - dom_weight).max()}")
                    # check adapters
                    for lora_adapter in ["lora_A", "lora_B"]:
                        if hasattr(proj, lora_adapter):
                            lora_weight = getattr(proj, lora_adapter).default.weight
                            dom_lora_weight = getattr(
                                dom_proj, lora_adapter
                            ).default.weight
                            if not torch.allclose(lora_weight, dom_lora_weight):
                                print(
                                    f"LoRA Weights for expert {expert_idx} in cluster {cluster_idx} for layer {layer_idx} and adapter {lora_adapter} are not tied!"
                                )
                                print(
                                    f"Max diff: {torch.abs(lora_weight - dom_lora_weight).max()}"
                                )

def get_super_expert_indices(observer_data, include_last_layers: bool = False):
    logger.info("Identifying super experts to preserve...")
    quantile = 99.5
    times = 10
    all_max_activations = [layer['max_activations'] for layer in observer_data.values()]
    num_layers = len(all_max_activations)
    all_max_activations = torch.cat(all_max_activations).flatten()
    percentile_threshold = torch.quantile(all_max_activations, quantile / 100.0).item()
    abs_threshold = all_max_activations.max().item() / times
    final_threshold = max(percentile_threshold, abs_threshold)
    # reshape back into per layer data
    all_max_activations = all_max_activations.reshape(num_layers, -1)
    super_experts_mask = all_max_activations > final_threshold
    if not include_last_layers:
        # only consider first 75% of layers for super experts
        logger.info(
            "Only considering first 75% of layers for super expert "
            "identification since perserve_outliers is False"
        )
        num_layers = int(num_layers * 0.75)
        super_experts_mask[num_layers:, :] = False
    super_expert_idx = torch.argwhere(super_experts_mask)
    logger.info(f"Identified {super_experts_mask.sum().item()} super experts with threshold: {final_threshold:.4f}")
    return super_expert_idx

def register_llama_with_vllm():
    from vllm.model_executor.models import ModelRegistry
    print("Registering Llama4ForCausalLM with vLLM")
    ModelRegistry.register_model("Llama4ForCausalLM", "vllm.model_executor.models.llama4:Llama4ForCausalLM")
