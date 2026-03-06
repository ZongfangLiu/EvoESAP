from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional
import gc
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from dataclasses import dataclass
import logging
import pathlib
from functools import reduce

from reap.metrics import (
    ttm_online,
    get_routed_characteristic_activation,
    ca_dist_online,
    OnlineStatsTracker,
    get_distance_fn,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseTransformerObserverHookConfig:
    state_attr_name: str = "hook_state"
    hook_attr_name: str = "hooks"
    module_name_to_hook_regex: Optional[str] = None
    module_class_name_to_hook_regex: Optional[nn.Module] = None


class BaseTransformerObserver(ABC):
    def __init__(
        self,
        model,
        hook_config: Optional[BaseTransformerObserverHookConfig] = None,
    ):
        self.model = model
        self.hook_config = hook_config
        self.hooks = []
        self.state: dict[Any, Any] = {}
        self._hook_model()
        logger.info(
            "%s initialized for %s.",
            self.__class__.__name__,
            self.model.__class__.__name__,
        )

    @abstractmethod
    def _hook_factory(self, module: nn.Module, layer_number: int) -> callable:
        """
        Factory method to create a hook function for the given module.
        This method should be implemented by subclasses to define how the
        hook function should behave.
        """
        raise NotImplementedError("Subclasses must implement _hook_factory method.")

    def report_state(self) -> dict[str, Any]:
        """
        Method to report the current state of the observer. Can be overridden to inject
        custom behaviours.
        """
        return self.state

    def close_hooks(self):
        """Close all hooks registered to the model."""
        self.reset()  # Reset the state before closing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        logger.debug("All hooks closed for %s.", self.model.__class__.__name__)

    def reset(self):
        """Reset the observer state."""
        del self.state
        gc.collect()
        self.state = {}
        logger.debug("Observer state reset for %s.", self.model.__class__.__name__)

    def save_state(self, file_path: str | pathlib.Path):
        self._move_state_tensors_to_cpu()
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = self.report_state()
        with open(file_path, "wb") as f:
            torch.save(state_dict, f)
        logger.info("State saved to %s", file_path)

    def _move_state_tensors_to_cpu(self):
        """
        Move all tensors in the state dictionary to CPU.
        This is useful before saving the state to avoid GPU memory issues.
        """
        for layer_number, layer_state in self.state.items():
            for key, value in layer_state.items():
                if isinstance(value, torch.Tensor):
                    self.state[layer_number][key] = value.cpu()

    def _validate_hook_config(self):
        if self.hook_config is None:
            return
        if (
            self.hook_config.module_name_to_hook_regex is None
            and self.hook_config.module_class_name_to_hook_regex is None
        ):
            raise ValueError(
                "At least one of 'module_n`ame_to_hook_regex' or "
                "'module_type_to_hook_regex' must be provided in the hook config."
            )
        if (
            self.hook_config.module_name_to_hook_regex is not None
            and self.hook_config.module_class_name_to_hook_regex is not None
        ):
            logger.warning(
                "Both 'module_name_to_hook_regex' and 'module_type_to_hook_regex' are "
                "provided. Both conditions must be satisfied to hook the module."
            )

    def _hook_model(self):
        for name, module in self.model.named_modules():
            hook_module = False
            if (
                self.hook_config.module_name_to_hook_regex
                and re.search(self.hook_config.module_name_to_hook_regex, name)
            ) or (
                self.hook_config.module_class_name_to_hook_regex
                and module.__class__.__name__
                == self.hook_config.module_class_name_to_hook_regex
            ):
                hook_module = True
            if hook_module:
                layer_number = int(re.search(r"\d+", name).group(0))
                hook_fn = self._hook_factory(module, layer_number)
                hook = module.register_forward_hook(hook_fn)
                self.hooks.append(hook)
                logger.info("Hooked module: %s at layer %d", name, layer_number)
        if len(self.hooks) == 0:
            raise ValueError(
                "No modules matched the provided hook configuration. "
                "Check your hook configuration settings."
            )

    @classmethod
    def _get_registry_for_cls(cls) -> dict[str, type[BaseTransformerObserver]]:
        """Helper to get the registry from the specific class 'cls'."""
        if not hasattr(cls, "_architecture_registry") or not isinstance(
            cls._architecture_registry, dict
        ):
            raise AttributeError(
                f"Class {cls.__name__} must define its own "
                "`_architecture_registry: dict[str, type] = {{}}` "
                f"to use the common registration/creation methods."
            )
        return cls._architecture_registry

    @classmethod
    def register_implementation(cls, *arch_names: str):
        """
        Class method decorator to register a concrete observer implementation.
        'cls' is the class on which this decorator's factory is called (e.g.,
        MoEExpertObserver) 'sub_cls' is the class being decorated
        (e.g., Llama4MoEExpertObserver).
        """

        def decorator(sub_cls: type[BaseTransformerObserver]):
            registry = cls._get_registry_for_cls()

            for name in arch_names:
                if name in registry:
                    raise RuntimeError(
                        f"Architecture {name} already registered with "
                        f"{registry[name].__name__} for {cls.__name__}."
                    )
                registry[name] = sub_cls
            return sub_cls

        return decorator

    @classmethod
    def create_from_registry(
        cls,
        model: nn.Module,
        hook_config: Optional[BaseTransformerObserverHookConfig] = None,
        return_rank_0_only: bool = True,
        **kwargs: Any,
    ) -> BaseTransformerObserver:
        registry = cls._get_registry_for_cls()
        model_cls_name = model.__class__.__name__

        specific_observer_cls = registry.get(model_cls_name)

        if specific_observer_cls:
            return specific_observer_cls(
                model,
                hook_config=hook_config,
                return_rank_0_only=return_rank_0_only,
                **kwargs,
            )
        else:
            raise ValueError(
                "Unsupported architecture for "
                f"{cls.__name__}: {model_cls_name}. "
                "Registered architectures in "
                f"{cls.__name__}._architecture_registry: "
                f"{list(registry.keys())}"
            )


# --- MoE Transformer Observer ---------------------------------------------------------


@dataclass
class MoETransformerObserverConfig(BaseTransformerObserverHookConfig):
    num_experts_attr_name: str = "num_experts"
    top_k_attr_name: str = "top_k"
    fused_experts: bool = False
    distance_measure: str = "angular"
    renormalize_router_weights: bool = False
    record_pruning_metrics_only: bool = False


class MoETransformerObserver(BaseTransformerObserver):
    """MoE Transformer Observer for all methods including both pruning and merging."""

    def report_state(self) -> dict[str, Any]:
        """
        Method to report the current state of the observer. Can be overridden to inject
        custom behaviours.
        """
        return {
            layer_num: {
                k: v.mean if isinstance(v, OnlineStatsTracker) else v
                for k, v in layer_state.items()
            }
            for layer_num, layer_state in self.state.items()
        }

    def _initialize_state(self, output: torch.Tensor, num_experts: int):
        # get device and shape info
        output_hidden_states = output[0]
        device = "cpu"
        hidden_dim = output_hidden_states.shape[-1]
        layer_state = {}

        # unnormalized states (counts)
        layer_state["total_tokens"] = torch.tensor(0, device=device, dtype=torch.long)
        layer_state["expert_frequency"] = torch.zeros(
            num_experts, device=device, dtype=torch.long
        )
        layer_state["pairwise_expert_frequency"] = torch.zeros(
            num_experts, num_experts, dtype=torch.long, device=device
        )

        if not self.hook_config.record_pruning_metrics_only:
            # per routed token normalized states
            layer_state["ttm_similarity_matrix"] = OnlineStatsTracker(
                shape=(num_experts, num_experts),
                count_shape=(num_experts, num_experts),
                device=device,
                dtype=torch.float32,
            )
            layer_state["routed_characteristic_activation"] = OnlineStatsTracker(
                shape=(num_experts, hidden_dim),
                count_shape=(num_experts, hidden_dim),
                device=device,
                dtype=torch.float32,
            )
            # HC-SMoE
            layer_state["characteristic_activation"] = OnlineStatsTracker(
                shape=(num_experts, hidden_dim),
                count_shape=1,
                device=device,
                dtype=torch.float32,
            )
            # SubMoE
            layer_state["online_characteristic_activation_dist"] = OnlineStatsTracker(
                shape=(num_experts, num_experts),
                count_shape=1,
                device=device,
                dtype=torch.float32,
            )
            # per total token normalized states -> MC-SMoE
            layer_state["router_logit_similiarity"] = OnlineStatsTracker(
                shape=(num_experts, num_experts),
                count_shape=1,
                device=device,
                dtype=torch.float32,
            )

        # Expert Activation Norm
        layer_state["ean_sum"] = torch.zeros(
            (num_experts,), device=device, dtype=torch.float64, requires_grad=False
        )
        layer_state["weighted_ean_sum"] = torch.zeros(
            (num_experts,), device=device, dtype=torch.float64, requires_grad=False
        )
        layer_state["ean_mean"] = OnlineStatsTracker(
            shape=(num_experts,),
            count_shape=(num_experts,),
            device=device,
            dtype=torch.float32,
        )
        layer_state["reap"] = OnlineStatsTracker(
            shape=(num_experts,),
            count_shape=(num_experts,),
            device=device,
            dtype=torch.float32,
        )

        # Weighted frequency
        layer_state["weighted_expert_frequency_sum"] = torch.zeros(
            (num_experts,), device=device, dtype=torch.float64, requires_grad=False
        )

        # super experts
        layer_state["max_activations"] = torch.zeros(
            (num_experts,), device=device, dtype=torch.float32, requires_grad=False
        )

        return layer_state

    def _hook_factory(self, module: nn.Module, layer_number: int) -> callable:
        distance_fn = get_distance_fn("cosine") # always use cosine for online dist. metrics
        def _get_nested_attr(root: object, dotted: str):
            cur = root
            for part in dotted.split("."):
                cur = getattr(cur, part)
            return cur

        def _get_attr_with_fallbacks(root: object, primary: str, fallbacks: list[str]):
            for path in [primary, *fallbacks]:
                try:
                    return _get_nested_attr(root, path)
                except AttributeError:
                    continue
            raise AttributeError(
                f"Could not resolve attribute '{primary}' (fallbacks={fallbacks}) on {root.__class__.__name__}."
            )

        num_experts = _get_attr_with_fallbacks(
            module,
            self.hook_config.num_experts_attr_name,
            ["num_experts", "gate.num_experts", "router.num_experts"],
        )
        top_k = _get_attr_with_fallbacks(
            module,
            self.hook_config.top_k_attr_name,
            ["top_k", "gate.top_k", "router.top_k"],
        )

        experts_module = getattr(module, "experts", None)
        if experts_module is not None and hasattr(experts_module, "num_experts"):
            try:
                maybe_num_experts = int(getattr(experts_module, "num_experts"))
                if maybe_num_experts > 0:
                    num_experts = maybe_num_experts
            except Exception:
                pass

        try:
            top_k = int(top_k)
            num_experts = int(num_experts)
        except Exception as e:
            raise ValueError(
                f"Invalid num_experts/top_k for {module.__class__.__name__} at layer {layer_number}: "
                f"num_experts={num_experts!r}, top_k={top_k!r}"
            ) from e

        if top_k > num_experts:
            top_k = num_experts
        if num_experts is None or top_k is None:
            raise ValueError(
                f"Module {module.__class__.__name__} at layer {layer_number} "
                "does not have expected 'num_experts' or 'top_k' attributes. Check "
                "HookConfig settings."
            )

        @torch.no_grad()
        def _hook_fn(module, args, output):
            input = args[0]  # (batch_size, seq_len, hidden_dim)
            device = input.device
            if layer_number not in self.state:
                self.state[layer_number] = self._initialize_state(output, num_experts)
            batch_size, sequence_length, hidden_dim = input.shape
            flat_input = input.view(-1, hidden_dim)  # total_seq_len, hidden
            activations = torch.zeros(
                (num_experts, *flat_input.shape), device=device, dtype=flat_input.dtype
            )

            router_logits = None
            selected_experts = None

            if self.hook_config.fused_experts:
                router_out = module.router(flat_input)  # (total_tokens, num_experts)
                if isinstance(router_out, (tuple, list)) and len(router_out) >= 1:
                    router_logits = router_out[0]
                else:
                    router_logits = router_out
                if not torch.is_tensor(router_logits) or router_logits.ndim != 2:
                    raise ValueError(
                        f"Unexpected router output for {module.__class__.__name__} at layer {layer_number}: "
                        f"type={type(router_out)}"
                    )
                if router_logits.shape == (num_experts, batch_size * sequence_length):
                    router_logits = router_logits.T
                _, selected_experts = torch.topk(router_logits, top_k, dim=-1)

                _, router_scores = output  # (num_experts, total_tokens)
                selected_experts = selected_experts.to(device)
                router_indices = (
                    torch.arange(batch_size * sequence_length, device=device)
                    .view(1, -1)
                    .expand(router_scores.size(0), -1)
                )
                router_indices = router_indices.reshape(-1, 1).expand(-1, hidden_dim)
                routed_in = torch.gather(
                    input=flat_input,
                    dim=0,
                    index=router_indices,
                ).to(device)
                # we do not apply router_scores
                # record unweighted activations for all experts
                routed_out = module.experts(routed_in)
                activations = routed_out.view(num_experts, *flat_input.shape)

            else:  # loop based MoE execution
                if isinstance(output, (tuple, list)) and len(output) > 0:
                    maybe_router_logits = output[-1]
                    if torch.is_tensor(maybe_router_logits) and maybe_router_logits.ndim == 2:
                        router_logits = maybe_router_logits

                if router_logits is None:
                    router_module = None
                    if hasattr(module, "router"):
                        router_module = getattr(module, "router")
                    elif hasattr(module, "gate"):
                        router_module = getattr(module, "gate")

                    if router_module is not None and callable(router_module):
                        router_out = router_module(flat_input)
                        if isinstance(router_out, (tuple, list)) and len(router_out) >= 1:
                            router_logits = router_out[0]
                        else:
                            router_logits = router_out

                if router_logits is None or not torch.is_tensor(router_logits):
                    raise ValueError(
                        f"Could not infer router logits for {module.__class__.__name__} at layer {layer_number}."
                    )
                if router_logits.shape == (num_experts, batch_size * sequence_length):
                    router_logits = router_logits.T
                _, selected_experts = torch.topk(router_logits, top_k, dim=-1)

                experts = getattr(module, "experts", None)
                if isinstance(experts, (nn.ModuleList, list, tuple)):
                    for idx, expert in enumerate(experts):
                        if idx >= num_experts:
                            break
                        activations[idx] = expert(flat_input).to(device)
                elif experts is not None and hasattr(experts, "gate_up_proj") and hasattr(
                    experts, "down_proj"
                ):
                    act_fn = getattr(experts, "act_fn", None)
                    if act_fn is None:
                        raise ValueError(
                            f"{experts.__class__.__name__} has no act_fn; cannot compute per-expert activations."
                        )
                    for idx in range(num_experts):
                        gate, up = F.linear(flat_input, experts.gate_up_proj[idx]).chunk(
                            2, dim=-1
                        )
                        hidden = act_fn(gate) * up
                        activations[idx] = F.linear(hidden, experts.down_proj[idx]).to(
                            device
                        )
                else:
                    raise ValueError(
                        f"Unsupported experts container for {module.__class__.__name__} at layer {layer_number}: "
                        f"type={type(experts)}"
                    )

            del flat_input
            num_tokens = batch_size * sequence_length
            num_tokens = torch.tensor(num_tokens, device="cpu", dtype=torch.long)

            # --- PRUNE/MERGE SALIENCY CRITERIA --------------------------------
            # expert frequency
            expert_frequency = torch.bincount(
                selected_experts.view(-1), minlength=num_experts
            ).to(device)
            pairwise_expert_frequency = expert_frequency.unsqueeze(
                0
            ) + expert_frequency.unsqueeze(1)
            pairwise_expert_frequency = pairwise_expert_frequency.to(device)

            self.state[layer_number]["total_tokens"] += num_tokens
            self.state[layer_number]["expert_frequency"] += expert_frequency.to(
                "cpu", torch.long
            )
            self.state[layer_number]["pairwise_expert_frequency"] += (
                pairwise_expert_frequency.to("cpu", torch.long)
            )

            # Merging critera
            if not self.hook_config.record_pruning_metrics_only:
                ttm_similarity_matrix = ttm_online(
                    activations,
                    selected_experts,
                    distance_callable=distance_fn,
                    num_experts=num_experts,
                    pairwise_expert_frequency=pairwise_expert_frequency,
                )

                # ttm_similarity_matrix with pairwise frequency counts
                self.state[layer_number]["ttm_similarity_matrix"].update(
                    ttm_similarity_matrix, pairwise_expert_frequency
                )
                del ttm_similarity_matrix

                routed_characteristic_activation = get_routed_characteristic_activation(
                    activations,
                    selected_experts,
                    expert_frequency,
                    device,
                    hidden_dim,
                    num_experts,
                )

                # routed_characteristic_activation with expert frequency counts
                expert_freq_expanded = expert_frequency.unsqueeze(-1).expand(
                    (-1, hidden_dim)
                )
                self.state[layer_number]["routed_characteristic_activation"].update(
                    routed_characteristic_activation, expert_freq_expanded
                )
                del expert_freq_expanded, routed_characteristic_activation

                online_characteristic_activation_dist = ca_dist_online(
                    activations,
                    distance_callable=distance_fn,
                ).to(device="cpu")

                # online_characteristic_activation_dist with expert frequency counts
                self.state[layer_number]["online_characteristic_activation_dist"].update(
                    online_characteristic_activation_dist, num_tokens
                )
                del online_characteristic_activation_dist
 
                # router logit similarity -> must align with distance_fn shape expectations
                # dim 0 "batch" dim, dims 1,2 expert pairwise, dim 3 token logits
                router_logit_sim = (
                    distance_fn(
                        router_logits.permute(1, 0).view(
                            1, num_experts, 1, -1
                        ),  # 1, num_experts, 1, logits
                        router_logits.permute(1, 0).view(
                            1, 1, num_experts, -1
                        ),  # 1, 1, num_experts, logits
                    )
                    .squeeze()
                    .to(device="cpu")
                )  # yields (num_experts, num_experts)

                # router_logit_similarity with total tokens count
                self.state[layer_number]["router_logit_similiarity"].update(
                    router_logit_sim, num_tokens
                )
                del router_logit_sim

                # characteristic_activation with total tokens count
                self.state[layer_number]["characteristic_activation"].update(
                    activations.mean(dim=1), num_tokens
                )

            # Pruning criteria
            ean_sum = torch.zeros(num_experts, device=device, dtype=torch.float64)
            ean_mean = torch.zeros(num_experts, device=device, dtype=torch.float32)
            weighted_ean_sum = torch.zeros(
                num_experts, device=device, dtype=torch.float64
            )
            reap = torch.zeros(
                num_experts, device=device, dtype=torch.float32
            )
            weighted_expert_frequency_sum = torch.zeros(
                num_experts, device=device, dtype=torch.float64
            )
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float).to(
                device
            )  # tok, num_experts
            prior_max_activations = self.state[layer_number]["max_activations"]
            # renormalize
            if self.hook_config.renormalize_router_weights:
                topk_weights = torch.gather(
                    routing_weights,
                    1,
                    selected_experts,
                )  # (total_tokens, top_k)
                routing_weights = routing_weights / topk_weights.sum(
                    dim=-1, keepdim=True
                )
                routing_weights = torch.clamp(
                    routing_weights, min=torch.finfo(routing_weights.dtype).eps
                )
                # routing_weights = routing_weights.to(device)

            for i in range(num_experts):
                active_mask = (selected_experts == i).any(dim=-1).to(device)
                if not active_mask.any():
                    continue
                active_router_weights = routing_weights[active_mask, i]
                ean_norm = torch.linalg.norm(activations[i, active_mask, :], dim=-1)
                ean_sum[i] = ean_norm.sum().to(device)
                ean_mean[i] = ean_norm.mean().to(device)
                weighted_expert_frequency_sum[i] = active_router_weights.sum().to(
                    device
                )
                weighted_ean_sum[i] = (
                    (ean_norm * active_router_weights).sum().to(device)
                )
                reap[i] = (
                    (ean_norm * active_router_weights).mean().to(device)
                )

                # super experts
                selected_activations = activations[i, active_mask, :]
                selected_activations_max = selected_activations.max().to(device="cpu")
                if selected_activations_max > prior_max_activations[i]:
                    self.state[layer_number]["max_activations"][i] = (
                        selected_activations_max
                    )
                    prior_max_activations[i] = selected_activations_max

            # ean
            self.state[layer_number]["ean_sum"] += ean_sum.to(device="cpu")
            self.state[layer_number]["ean_mean"].update(ean_mean, expert_frequency)
            self.state[layer_number]["weighted_ean_sum"] += weighted_ean_sum.to(
                device="cpu"
            )
            if reap.sum() == 0:
                print("debug")
            self.state[layer_number]["reap"].update(
                reap, expert_frequency
            )

            # weighted_expert_frequency_sum
            
            self.state[layer_number]["weighted_expert_frequency_sum"] += (
                weighted_expert_frequency_sum.to(device="cpu")
            )

            # --- CLEAN UP -------------------------------------------------------------
            del (
                activations,
                selected_experts,
                router_logits,
                expert_frequency,
                pairwise_expert_frequency,
                prior_max_activations,
            )
            gc.collect()

        return _hook_fn


# --- Concrete Config Implementations ----


@dataclass
class Qwen3MoEObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: Optional[str] = "Qwen3MoeSparseMoeBlock"
    num_experts_attr_name: str = "gate.num_experts"
    top_k_attr_name: str = "gate.top_k"


@dataclass
class Llama4MoEObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: Optional[str] = "Llama4TextMoe"
    fused_experts: bool = True  # Llama4 uses fused experts


@dataclass
class MixtralMoEObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: Optional[str] = "MixtralSparseMoeBlock"


@dataclass
class DeepSeekMoEObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: Optional[str] = "DeepseekV2MoE"
    num_experts_attr_name: str = "experts_per_rank"  # only for ep=1!
    top_k_attr_name: str = "num_experts_per_tok"
    fused_experts: bool = False


@dataclass
class Ernie4_5MoEObserverHookConfig(MoETransformerObserverConfig):
    # Hook the actual MoE block used by the HF implementation; older patched variants
    # used `Ernie4_5_MoeMLP`, but the in-tree model exposes experts on
    # `Ernie4_5_MoeSparseMoeBlock`.
    module_class_name_to_hook_regex: Optional[str] = "Ernie4_5_MoeSparseMoeBlock"
    num_experts_attr_name: str = "num_experts"
    top_k_attr_name: str = "top_k"
    fused_experts: bool = False


@dataclass
class Glm44MoEObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: Optional[str] = "Glm4MoeMoE"
    num_experts_attr_name: str = "config.n_routed_experts"
    top_k_attr_name: str = "config.num_experts_per_tok"
    fused_experts: bool = False

@dataclass
class GptOssMoEObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: Optional[str] = "GptOssMLP"
    num_experts_attr_name: str = "router.num_experts"
    top_k_attr_name: str = "router.top_k"
    fused_experts: bool = True

@dataclass
class OlmoeObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: Optional[str] = "OlmoeSparseMoeBlock"
    num_experts_attr_name: str = "gate.num_experts"
    top_k_attr_name: str = "gate.top_k"
    fused_experts: bool = False


OBSERVER_CONFIG_REGISTRY = {
    "Qwen3MoeForCausalLM": Qwen3MoEObserverHookConfig,
    "NonUniformQwen3MoeForCausalLM": Qwen3MoEObserverHookConfig,
    "Llama4ForCausalLM": Llama4MoEObserverHookConfig,
    "MixtralForCausalLM": MixtralMoEObserverHookConfig,
    "DeepseekV2ForCausalLM": DeepSeekMoEObserverHookConfig,
    "Ernie4_5_MoEForCausalLM": Ernie4_5MoEObserverHookConfig,
    "Ernie4_5_MoeForCausalLM": Ernie4_5MoEObserverHookConfig,
    "NonUniformErnie4_5_MoeForCausalLM": Ernie4_5MoEObserverHookConfig,
    "Glm4MoeForCausalLM": Glm44MoEObserverHookConfig,
    "GptOssForCausalLM": GptOssMoEObserverHookConfig,
    "OlmoeForCausalLM": OlmoeObserverHookConfig,
}
