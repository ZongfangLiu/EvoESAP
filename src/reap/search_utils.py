from __future__ import annotations

import copy
import dataclasses
import json
import logging
import random
import re
import time
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch import nn
from tqdm import tqdm

from reap.model_util import MODEL_ATTRS, get_moe

logger = logging.getLogger(__name__)


def normalize_run_name_base(run_name: str | None) -> str | None:
    if not run_name:
        return None
    base = re.sub(r"-gen\\d+", "", str(run_name))
    base = re.sub(r"-struct(?=-|$)", "", base)
    base = base.replace("calib_tulu-3-sft-personas-math", "calib_tulu-3")
    base = re.sub(r"--+", "-", base).strip("-")
    return base


def _allocate_budget(
    weights: Sequence[float], capacities: Sequence[int], budget: int
) -> List[int]:
    """
    Allocate an integer budget across layers according to weights, respecting per-layer
    capacity (max experts that can be pruned).
    """
    n = len(capacities)
    if n == 0:
        return []
    total_weight = float(sum(weights)) or float(n)
    desired = [budget * (w / total_weight) for w in weights]
    alloc = [min(int(d), capacities[i]) for i, d in enumerate(desired)]
    used = sum(alloc)
    # Distribute leftover based on fractional parts, then round-robin for any remainder.
    fractional = [
        (i, desired[i] - alloc[i])
        for i in range(n)
        if alloc[i] < capacities[i]
    ]
    fractional.sort(key=lambda x: x[1], reverse=True)
    remaining = budget - used
    for i, _ in fractional:
        if remaining <= 0:
            break
        if alloc[i] < capacities[i]:
            alloc[i] += 1
            remaining -= 1
    if remaining > 0:
        # Round-robin to fill any remainder within capacity.
        while remaining > 0:
            made_progress = False
            for i in range(n):
                if remaining <= 0:
                    break
                if alloc[i] < capacities[i]:
                    alloc[i] += 1
                    remaining -= 1
                    made_progress = True
            if not made_progress:
                # Cannot place further budget without exceeding capacity.
                logger.warning(
                    "Unable to allocate full budget (%d remaining); capacities may be too small.",
                    remaining,
                )
                break
    # ensure exact budget when capacity allows; otherwise fail fast
    if sum(alloc) != budget:
        raise ValueError(
            f"Failed to allocate exact budget. Requested={budget}, allocated={sum(alloc)}, "
            f"capacities={list(capacities)}"
        )
    return alloc


def _pattern_weights(pattern: str, n: int, rng: random.Random) -> List[float]:
    if pattern == "random":
        return [rng.random() + 1e-4 for _ in range(n)]
    if pattern == "up":
        return [i + 1 for i in range(n)]
    if pattern == "down":
        return [n - i for i in range(n)]
    if pattern == "mid":
        # Triangular peak toward the middle layers.
        mid = (n - 1) / 2
        return [mid - abs(i - mid) + 1 for i in range(n)]
    if pattern == "mid-neg":
        # Inverse triangular: emphasize edges, de-emphasize middle.
        mid = (n - 1) / 2
        return [abs(i - mid) + 1 for i in range(n)]
    if pattern == "front-heavy":
        # Exponential decay from early layers.
        return [pow(0.7, i) + 1e-4 for i in range(n)]
    if pattern == "back-heavy":
        # Exponential growth toward later layers.
        return [pow(1.3, i) for i in range(n)]
    if pattern == "block-early":
        # Concentrate on the first quarter of layers.
        cutoff = max(1, n // 4)
        return [2.0 if i < cutoff else 1.0 for i in range(n)]
    if pattern == "block-late":
        # Concentrate on the last quarter of layers.
        cutoff = max(1, n // 4)
        return [2.0 if i >= n - cutoff else 1.0 for i in range(n)]
    if pattern == "alt":
        # Checkerboard emphasis across layers.
        return [2.0 if i % 2 == 0 else 1.0 for i in range(n)]
    if pattern == "deep-flat":
        # Gentle taper at ends, flat in the middle.
        return [0.5 + 0.5 * (1 - abs((2 * i / max(n - 1, 1)) - 1)) for i in range(n)]
    # default to uniform
    return [1.0] * n


def _normalize_parities(
    parities: Sequence[int],
    capacities: Sequence[int],
    budget: int,
) -> List[int]:
    n = len(capacities)
    if len(parities) != n:
        raise ValueError("Parity constraints must match the number of layers.")
    base = [int(p) & 1 for p in parities]
    for i, parity in enumerate(base):
        if parity > capacities[i]:
            raise ValueError(
                f"Parity constraint requires at least {parity} removals for layer {i}, "
                f"but capacity is {capacities[i]}."
            )
    base_sum = sum(base)
    if budget < base_sum:
        raise ValueError(
            f"Budget {budget} is too small for parity-constrained minimum {base_sum}."
        )
    if (budget - base_sum) % 2 != 0:
        raise ValueError(
            f"Budget {budget} is incompatible with parity constraints (min={base_sum})."
        )
    max_budget = sum(
        base[i] + 2 * ((capacities[i] - base[i]) // 2) for i in range(n)
    )
    if budget > max_budget:
        raise ValueError(
            f"Budget {budget} exceeds parity-constrained capacity {max_budget}."
        )
    return base


def _plan_matches_parity(plan: Sequence[int], parities: Sequence[int]) -> bool:
    if len(plan) != len(parities):
        return False
    return all(
        (int(value) - int(parity)) % 2 == 0
        for value, parity in zip(plan, parities)
    )


def init_pattern(
    pattern: str,
    capacities: Sequence[int],
    budget: int,
    rng: random.Random,
    parities: Sequence[int] | None = None,
) -> List[int]:
    weights = _pattern_weights(pattern, len(capacities), rng)
    if parities is None:
        return _allocate_budget(weights, capacities, budget)
    base = _normalize_parities(parities, capacities, budget)
    remaining_budget = budget - sum(base)
    even_capacities = [
        (capacities[i] - base[i]) // 2 for i in range(len(capacities))
    ]
    alloc = _allocate_budget(weights, even_capacities, remaining_budget // 2)
    return [base[i] + 2 * alloc[i] for i in range(len(capacities))]


def plan_to_dict(plan: Sequence[int], layers: Sequence[int]) -> Dict[int, int]:
    return {layer: int(plan[idx]) for idx, layer in enumerate(layers)}


def _format_ratio_list(plan: Sequence[int], totals: Sequence[int]) -> str:
    ratios = []
    for idx, total in enumerate(totals):
        ratio = (plan[idx] / total) if total else 0.0
        ratios.append(f"{ratio:.3f}")
    return "[" + ", ".join(ratios) + "]"


def _atomic_torch_save(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.name}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(target)


def _atomic_json_dump(path: str | Path, payload: Any, indent: int = 2) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f"{target.name}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=indent)
    tmp_path.replace(target)


def _append_jsonl_log(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        return "out of memory" in str(exc).lower()
    return False


def _filtered_search_args(search_args: Any) -> Dict[str, Any]:
    try:
        args_dict = dataclasses.asdict(search_args)
    except Exception:
        return {}
    for key in {
        "output_dir",
        "generations",
        "search_resume_from",
        "search_checkpoint_every",
        "search_dataset_name",
        "search_dataset_weights",
        "search_dataset_config_name",
        "search_dataset_split",
        "search_samples_per_category",
        "search_pack_samples",
        "search_microbatch_size",
        "search_microbatch_max",
    }:
        args_dict.pop(key, None)
    return args_dict


def _validate_resume_state(
    resume_state: Dict[str, Any],
    layers: Sequence[int],
    capacities: Sequence[int],
    budget: int,
    min_keep_per_layer: int,
    search_args: Any,
    search_data_fingerprint: str | None,
    population_size: int,
    survivors: int,
    search_microbatch_size: int | None,
    run_name_base: str | None,
) -> None:
    errors: List[str] = []
    if resume_state.get("layers") != list(layers):
        errors.append("layers mismatch")
    if resume_state.get("capacities") != list(capacities):
        errors.append("capacities mismatch")
    if resume_state.get("budget") != budget:
        errors.append("budget mismatch")
    if resume_state.get("min_keep_per_layer") != min_keep_per_layer:
        errors.append("min_keep_per_layer mismatch")
    if resume_state.get("population_size") != population_size:
        errors.append("population_size mismatch")
    if resume_state.get("survivors") != survivors:
        errors.append("topk/survivors mismatch")
    saved_fingerprint = resume_state.get("search_data_fingerprint")
    if saved_fingerprint and search_data_fingerprint and saved_fingerprint != search_data_fingerprint:
        errors.append("search_data_fingerprint mismatch")
    saved_run_base = resume_state.get("run_name_base")
    if saved_run_base and run_name_base:
        saved_run_base_clean = normalize_run_name_base(str(saved_run_base))
        run_name_base_clean = normalize_run_name_base(str(run_name_base))
        if saved_run_base_clean and run_name_base_clean and saved_run_base_clean != run_name_base_clean:
            errors.append("run_name_base mismatch")
    saved_target_generations = resume_state.get("target_generations")
    if saved_target_generations is not None and int(search_args.generations) < int(saved_target_generations):
        logger.warning(
            "Resume requested with generations=%d < previous target_generations=%d; continuing.",
            int(search_args.generations),
            int(saved_target_generations),
        )
    saved_microbatch = resume_state.get("search_microbatch_size")
    if saved_microbatch is not None and search_microbatch_size is not None:
        if int(saved_microbatch) != int(search_microbatch_size):
            errors.append("search_microbatch_size mismatch")
    saved_args = resume_state.get("search_args")
    if saved_args:
        current_args = _filtered_search_args(search_args)
        for key, value in current_args.items():
            if key in saved_args and saved_args[key] != value:
                errors.append(f"search_args.{key} mismatch")
    history = resume_state.get("history")
    if history:
        last_gen = history[-1].get("generation")
        if last_gen is not None and int(last_gen) != int(resume_state.get("generation", -1)):
            errors.append("history generation mismatch")
    if errors:
        raise ValueError("Cannot resume search: " + "; ".join(errors))

def mutate_plan(
    plan: Sequence[int],
    capacities: Sequence[int],
    rng: random.Random,
    max_delta: int,
    mutation_times: int,
    max_attempts: int,
    even_delta: bool = False,
) -> Tuple[List[int], int, int, int]:
    """
    Apply a random number of valid level-switch mutations (sampled from mutation_times),
    retrying up to max_attempts total. Returns
    (child_plan, successful_mutations, target_mutations, attempts_made).
    """
    child = list(plan)
    successes = 0
    attempts = 0
    desired = max(mutation_times, 1)
    target_mutations = min(rng.randint(1, desired), rng.randint(1, desired))
    max_even_delta = max_delta if max_delta % 2 == 0 else max_delta - 1
    while successes < target_mutations and attempts < max_attempts:
        attempts += 1
        decrease_candidates = [i for i, v in enumerate(child) if v > 0]
        increase_candidates = [i for i, cap in enumerate(capacities) if child[i] < cap]
        if not decrease_candidates or not increase_candidates:
            break
        src = rng.choice(decrease_candidates)
        dst = rng.choice(increase_candidates)
        if src == dst and len(increase_candidates) > 1:
            continue
        if even_delta:
            if max_even_delta < 2:
                break
            delta = rng.randrange(2, max_even_delta + 1, 2)
        else:
            delta = rng.randint(1, max(max_delta, 1))
        if child[src] - delta < 0:
            continue
        if child[dst] + delta > capacities[dst]:
            continue
        child[src] -= delta
        child[dst] += delta
        successes += 1
    return child, successes, target_mutations, attempts


def precompute_baseline_cache(
    model: nn.Module,
    search_batches: Sequence[Any],
    fitness_mode: str,
) -> Dict[bytes, torch.Tensor]:
    """
    Precompute baseline logits for offline fitnesses that compare against a baseline
    model (KL, expected speculative acceptance). Returns a cache keyed by batch
    tensors.
    """
    if fitness_mode not in {"kl-baseline", "kl-pq-b", "esp-dataset", "sp-dataset"}:
        return {}
    device = next(model.parameters()).device
    cache: Dict[bytes, torch.Tensor] = {}
    model.eval()
    iterable = search_batches
    try:
        iterable = tqdm(
            search_batches,
            desc="Baseline logits (cache)",
            leave=False,
        )
    except Exception:
        iterable = search_batches
    with torch.inference_mode():
        for batch in iterable:
            if isinstance(batch, torch.Tensor):
                input_ids = batch.to(device)
                attention_mask = torch.ones_like(input_ids, device=device)
            elif isinstance(batch, dict):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids, device=device)
                else:
                    attention_mask = attention_mask.to(device)
            else:
                input_ids = batch.input_ids.to(device)
                attention_mask = getattr(batch, "attention_mask", None)
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids, device=device)
                else:
                    attention_mask = attention_mask.to(device)
            key = (
                input_ids.detach().cpu().numpy().tobytes()
                + attention_mask.detach().cpu().numpy().tobytes()
            )
            if key in cache:
                continue
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
            cache[key] = logits.detach().cpu()
    return cache


def precompute_esp_generation_cache(
    model: nn.Module,
    search_batches: Sequence[Any],
    max_new_tokens: int | None = None,
) -> Dict[bytes, Dict[str, torch.Tensor]]:
    """
    Precompute baseline continuations + logits for esp-p/kl-p using a single model.
    Stores tensors on CPU to avoid duplicating GPU memory during search.
    """
    cache: Dict[bytes, Dict[str, torch.Tensor]] = {}
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    gen_max_new_tokens = max_new_tokens if max_new_tokens is not None and max_new_tokens > 0 else 64
    try:
        iterable = tqdm(
            search_batches,
            desc="Baseline generation (cache)",
            leave=False,
        )
    except Exception:
        iterable = search_batches
    with torch.inference_mode():
        for batch in iterable:
            if isinstance(batch, torch.Tensor):
                input_ids = batch.to(device)
                attention_mask = torch.ones_like(input_ids, device=device)
            elif isinstance(batch, dict):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids, device=device)
                else:
                    attention_mask = attention_mask.to(device)
            else:
                input_ids = batch.input_ids.to(device)
                attention_mask = getattr(batch, "attention_mask", None)
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids, device=device)
                else:
                    attention_mask = attention_mask.to(device)
        
            prompt_lens = attention_mask.to(dtype=torch.long).sum(dim=-1)  # shape: (bs,)
            
            key = (
                input_ids.detach().cpu().numpy().tobytes()
                + attention_mask.detach().cpu().numpy().tobytes()
            )
            if key in cache:
                continue
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_max_new_tokens,
                do_sample=False,
            )
            gen_attention = torch.ones_like(generated, device=device)
            base_logits = model(input_ids=generated, attention_mask=gen_attention).logits
            cache[key] = {
                "generated": generated.detach().cpu(),
                "base_logits": base_logits.detach().cpu(),
            }
    if was_training:
        model.train()
    return cache


def _install_temp_router_masks(
    model: nn.Module, pruned_experts_info: Dict[int, List[int]]
) -> List[Any]:
    """
    Temporarily prevents pruned experts from being selected by the router.

    - OLMoE / Qwen3 MoE: if the router is a TopKRouter that computes logits internally
      (via `F.linear` + `softmax`), forward hooks are too late. We monkey-patch its
      `forward` to mask logits right before the `softmax`, then restore afterwards.
    - ERNIE 4.5 MoE: top-k uses `moe_statics(routing_weights)` where `moe_statics` adds
      `e_score_correction_bias`, so we can mask pruned experts by setting the
      corresponding bias entries to `finfo.min`, then restore afterwards.

    For other routers, we best-effort hook the module that produces full router logits
    (typically a `nn.Linear`) and mask its output.
    """
    model_attrs = MODEL_ATTRS[model.__class__.__name__]
    top_k_default = int(getattr(model.config, model_attrs["num_experts_per_tok"]))
    num_experts_default = int(getattr(model.config, model_attrs["num_experts"]))
    handles: List[Any] = []
    warned_layers: set[tuple[int, str]] = set()

    def _as_valid_indices(pruned: List[int] | None, num_experts: int) -> List[int]:
        if not pruned:
            return []
        return sorted({int(i) for i in pruned if 0 <= int(i) < int(num_experts)})

    def _warn_masked_selected(
        layer: int,
        router_name: str,
        selected: torch.Tensor,
        valid: Sequence[int],
    ) -> None:
        if not valid:
            return
        key = (int(layer), str(router_name))
        if key in warned_layers:
            return
        if (not torch.is_tensor(selected)) or selected.numel() == 0:
            return
        if selected.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            return
        idx = torch.tensor(valid, device=selected.device, dtype=selected.dtype)
        if idx.numel() == 0:
            return
        try:
            bad_mask = torch.isin(selected, idx)
        except Exception:
            bad_mask = (selected.unsqueeze(-1) == idx).any(dim=-1)
        if not bool(bad_mask.any()):
            return
        bad_vals = torch.unique(selected[bad_mask]).detach().cpu().tolist()
        preview = bad_vals[:8]
        logger.warning(
            "Masked experts were still activated during search (layer=%d, router=%s, masked=%s, count=%d).",
            layer,
            router_name,
            preview,
            len(bad_vals),
        )
        warned_layers.add(key)

    def _collect_tensors(value: Any, bucket: List[torch.Tensor]) -> None:
        if torch.is_tensor(value):
            bucket.append(value)
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                _collect_tensors(item, bucket)
            return
        if isinstance(value, dict):
            for item in value.values():
                _collect_tensors(item, bucket)
            return

    def _inspect_router_output(
        output: Any,
        warn_fn,
        top_k: int,
        num_experts: int,
    ) -> None:
        if warn_fn is None:
            return
        warn_key = getattr(warn_fn, "_warn_key", None)
        if warn_key is not None and warn_key in warned_layers:
            return
        tensors: List[torch.Tensor] = []
        _collect_tensors(output, tensors)
        if not tensors:
            return
        int_types = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
        for tensor in tensors:
            if (not torch.is_tensor(tensor)) or tensor.numel() == 0:
                continue
            if tensor.dtype in int_types:
                try:
                    if int(tensor.min().item()) < 0 or int(tensor.max().item()) >= int(num_experts):
                        continue
                except Exception:
                    continue
                if tensor.ndim > 0 and int(tensor.shape[-1]) <= max(int(top_k), 1):
                    warn_fn(tensor)
                    return
        for tensor in tensors:
            if (not torch.is_tensor(tensor)) or tensor.numel() == 0:
                continue
            if (not torch.is_floating_point(tensor)) or tensor.ndim == 0:
                continue
            if int(tensor.shape[-1]) != int(num_experts):
                continue
            k = min(int(top_k), int(num_experts))
            if k <= 0:
                return
            try:
                indices = torch.topk(tensor, k=k, dim=-1).indices
            except Exception:
                return
            warn_fn(indices)
            return
    
    # def _mask_logits_inplace_(logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    #     if indices.numel() == 0:
    #         return logits
    #     if (not torch.is_floating_point(logits)) or logits.shape[-1] <= int(indices.max().item()):
    #         return logits
    #     logits.index_fill_(-1, indices, torch.finfo(logits.dtype).min)  # in-place
    #     return logits

    def _mask_logits_inplace(logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if indices.numel() == 0:
            return logits
        if (not torch.is_floating_point(logits)) or logits.shape[-1] <= int(indices.max().item()):
            return logits
        masked = logits.clone()
        masked.index_fill_(-1, indices, torch.finfo(masked.dtype).min)
        return masked

    def _try_install_ernie_bias_mask(moe: nn.Module, router: nn.Module, valid: List[int]) -> bool:
        if not valid:
            return False
        statics = getattr(moe, "moe_statics", None)
        if statics is None:
            statics = getattr(router, "moe_statics", None)
        bias = getattr(statics, "e_score_correction_bias", None) if statics is not None else None
        if not isinstance(bias, (nn.Parameter, torch.Tensor)):
            return False

        bias_tensor = bias.data if isinstance(bias, nn.Parameter) else bias
        original = bias_tensor.detach().clone()
        idx = torch.tensor(valid, device=bias_tensor.device, dtype=torch.long)
        bias_tensor.index_fill_(-1, idx, torch.finfo(bias_tensor.dtype).min)

        def _restore():
            try:
                bias_tensor.copy_(original)
            except Exception:
                pass

        handles.append(_FnHandle(_restore))
        return True

    def _try_install_olmoe_topkrouter_patch(
        router: nn.Module, valid: List[int], warn_fn
    ) -> bool:
        if not valid:
            return False
        # OLMoE TopKRouter (HF-style) uses a weight Parameter and calls F.linear + softmax.
        if router.__class__.__name__ not in {"OlmoeTopKRouter", "Qwen3MoeTopKRouter"}:
            return False
        if not hasattr(router, "weight") or not hasattr(router, "top_k"):
            return False


        original_forward = router.forward
        idx_cpu = torch.tensor(valid, dtype=torch.long)

        def patched_forward(hidden_states, *args, experts_to_prune=None, **kwargs):
            # Keep logic identical to HF's OlmoeTopKRouter, only mask before softmax.
            # print("Patched OlmoeTopKRouter forward called.")
            hidden_dim = int(getattr(router, "hidden_dim", hidden_states.shape[-1]))
            top_k_local = int(getattr(router, "top_k"))
            norm_topk_prob = bool(getattr(router, "norm_topk_prob", False))

            hs = hidden_states.reshape(-1, hidden_dim)
            router_logits = torch.nn.functional.linear(hs, router.weight)

            experts = experts_to_prune
            if experts is None:
                experts = idx_cpu
            if isinstance(experts, (list, tuple)):
                experts = torch.tensor(list(experts), dtype=torch.long)
            if torch.is_tensor(experts) and experts.numel() > 0:
                experts = experts.to(device=router_logits.device, dtype=torch.long)
                router_logits = _mask_logits_inplace(router_logits, experts)

            router_logits = torch.nn.functional.softmax(
                router_logits, dtype=torch.float, dim=-1
            )
            router_top_value, router_indices = torch.topk(
                router_logits, top_k_local, dim=-1
            )
            if norm_topk_prob:
                router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
            router_top_value = router_top_value.to(router_logits.dtype)
            router_scores = router_top_value

            if warn_fn is not None:
                warn_fn(router_indices)

            return router_logits, router_scores, router_indices

        router.forward = patched_forward  # type: ignore[assignment]

        def _restore():
            try:
                router.forward = original_forward  # type: ignore[assignment]
            except Exception:
                pass

        handles.append(_FnHandle(_restore))
        return True

    def _try_install_ernie_topkrouter_warning(router: nn.Module, warn_fn) -> bool:
        if warn_fn is None:
            return False
        if router.__class__.__name__ not in {"Ernie4_5_MoeTopKRouter"}:
            return False
        original_forward = router.forward

        def patched_forward(*args, **kwargs):
            out = original_forward(*args, **kwargs)
            try:
                if isinstance(out, (tuple, list)) and len(out) >= 2:
                    selected = out[1]
                    if torch.is_tensor(selected):
                        warn_fn(selected)
            except Exception:
                pass
            return out

        router.forward = patched_forward  # type: ignore[assignment]

        def _restore():
            try:
                router.forward = original_forward  # type: ignore[assignment]
            except Exception:
                pass

        handles.append(_FnHandle(_restore))
        return True

    def _find_logits_producer(router: nn.Module, num_experts: int) -> nn.Module:
        if isinstance(router, nn.Linear) and int(router.out_features) == num_experts:
            return router
        for attr in ("gate", "router", "linear", "proj"):
            sub = getattr(router, attr, None)
            if isinstance(sub, nn.Linear) and int(sub.out_features) == num_experts:
                return sub
        for _name, mod in router.named_modules():
            if isinstance(mod, nn.Linear) and int(mod.out_features) == num_experts:
                return mod
        return router

    get_num_experts = getattr(model.config, "get_num_experts", None)
    per_layer = getattr(model.config, "num_experts_per_layer", None)

    for layer, pruned in pruned_experts_info.items():
        moe = get_moe(model, layer)
        router = getattr(moe, model_attrs["router"])
        if callable(get_num_experts):
            num_experts_layer = int(get_num_experts(layer))
        elif isinstance(per_layer, list) and 0 <= layer < len(per_layer):
            num_experts_layer = int(per_layer[layer])
        else:
            num_experts_layer = num_experts_default
        valid = _as_valid_indices(pruned, num_experts_layer)
        top_k = min(top_k_default, num_experts_layer)

        if valid:
            assert (num_experts_layer - len(valid)) >= top_k, (
                f"Layer {layer}: pruned too many experts for top_k={top_k} "
                f"(num_experts={num_experts_layer}, pruned={len(valid)})."
            )

        warn_fn = None
        if valid:
            warn_key = (int(layer), str(router.__class__.__name__))

            def _warn_selected(
                selected,
                layer=layer,
                router_name=router.__class__.__name__,
                valid=valid,
            ):
                _warn_masked_selected(layer, router_name, selected, valid)

            _warn_selected._warn_key = warn_key
            warn_fn = _warn_selected

        if _try_install_ernie_bias_mask(moe, router, valid):
            pass
        elif _try_install_olmoe_topkrouter_patch(router, valid, warn_fn):
            pass
        else:
            raise RuntimeError(
                "Router mask installation failed for "
                f"{model.__class__.__name__} layer {layer} "
                f"(router={router.__class__.__name__}). "
                "No supported masking strategy found."
            )

        if warn_fn is not None:
            _try_install_ernie_topkrouter_warning(router, warn_fn)

            def _warn_hook(_module, _inp, out, warn_fn=warn_fn, top_k=top_k, num_experts_layer=num_experts_layer):
                try:
                    _inspect_router_output(out, warn_fn, top_k, num_experts_layer)
                except Exception:
                    pass

            handles.append(router.register_forward_hook(_warn_hook))

    return handles

def _cleanup_handles(handles: Sequence[Any]):
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

class _FnHandle:
    def __init__(self, fn):
        self._fn = fn
    def remove(self):
        self._fn()


def _resolve_model_max_length(model: nn.Module) -> int | None:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    candidates: List[int] = []
    for attr in (
        "max_position_embeddings",
        "max_sequence_length",
        "model_max_length",
        "max_length",
    ):
        val = getattr(cfg, attr, None)
        if val is None:
            continue
        try:
            val_int = int(val)
        except (TypeError, ValueError):
            continue
        if val_int > 0:
            candidates.append(val_int)
    if not candidates:
        return None
    return min(candidates)


@dataclasses.dataclass
class _SpecDecStats:
    accepted: int = 0
    proposed: int = 0


def _record_spec_dec_stats(stats: _SpecDecStats, scores: Any, num_matches: Any) -> None:
    proposed = 0
    if scores is not None:
        try:
            proposed = int(scores.shape[1] - 1)
        except Exception:
            proposed = 0
    if proposed > 0:
        stats.proposed += proposed
    try:
        accepted = int(num_matches.item()) if torch.is_tensor(num_matches) else int(num_matches)
    except Exception:
        accepted = 0
    if accepted > 0:
        stats.accepted += accepted


class _SpecDecStatsTracker:
    def __init__(self, stats: _SpecDecStats, expected_assistant: nn.Module | None = None):
        self._stats = stats
        self._expected_assistant = expected_assistant
        self._orig_get_candidate_generator = None
        self._installed = False
        self._saw_assisted = False

    @property
    def installed(self) -> bool:
        return self._installed

    @property
    def saw_assisted(self) -> bool:
        return self._saw_assisted

    def __enter__(self):
        try:
            from transformers.generation.utils import GenerationMixin
        except Exception:
            return self
        if not hasattr(GenerationMixin, "_get_candidate_generator"):
            return self
        self._orig_get_candidate_generator = GenerationMixin._get_candidate_generator
        stats = self._stats
        orig_get = self._orig_get_candidate_generator
        expected_assistant = self._expected_assistant

        def _get_candidate_generator(instance, *args, **kwargs):
            candidate_generator = orig_get(instance, *args, **kwargs)
            if expected_assistant is not None:
                if not hasattr(candidate_generator, "assistant_model"):
                    raise RuntimeError(
                        "Spec-dec candidate generator did not expose assistant_model; masked model not used."
                    )
                if candidate_generator.assistant_model is not expected_assistant:
                    raise RuntimeError(
                        "Spec-dec assistant model mismatch; masked model not used for assisted generation."
                    )
                self._saw_assisted = True
            orig_prepare = getattr(candidate_generator, "_prepare_generation_args", None)
            if callable(orig_prepare):
                def _prepare(self_cg, *p_args, **p_kwargs):
                    generation_args = orig_prepare(*p_args, **p_kwargs)
                    gen_cfg = generation_args.get("generation_config")
                    if gen_cfg is not None:
                        clean_cfg = copy.deepcopy(gen_cfg)
                        clean_cfg.max_length = None
                        clean_cfg.min_length = None
                        clean_cfg.max_new_tokens = None
                        clean_cfg.min_new_tokens = None
                        generation_args["generation_config"] = clean_cfg
                    return generation_args

                candidate_generator._prepare_generation_args = MethodType(_prepare, candidate_generator)
            orig_update = candidate_generator.update_candidate_strategy

            def _update(self_cg, input_ids, scores, num_matches):
                _record_spec_dec_stats(stats, scores, num_matches)
                return orig_update(input_ids, scores, num_matches)

            candidate_generator.update_candidate_strategy = MethodType(_update, candidate_generator)
            return candidate_generator

        GenerationMixin._get_candidate_generator = _get_candidate_generator
        self._installed = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._installed and self._orig_get_candidate_generator is not None:
            from transformers.generation.utils import GenerationMixin

            GenerationMixin._get_candidate_generator = self._orig_get_candidate_generator


def _spec_dec_stats_transformers(
    target_model: nn.Module,
    assistant_model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    max_new_tokens: int,
    num_assistant_tokens: int,
) -> Tuple[int, int]:
    if max_new_tokens <= 0:
        return 0, 0
    try:
        target_device = target_model.device
    except Exception:
        try:
            target_device = next(target_model.parameters()).device
        except StopIteration:
            target_device = input_ids.device
    if input_ids.device != target_device:
        input_ids = input_ids.to(target_device)
    if attention_mask is not None and attention_mask.device != target_device:
        attention_mask = attention_mask.to(target_device)
    input_len = int(input_ids.shape[1])
    target_max_ctx = _resolve_model_max_length(target_model)
    assistant_max_ctx = _resolve_model_max_length(assistant_model)
    if target_max_ctx is not None and assistant_max_ctx is not None:
        max_ctx = min(target_max_ctx, assistant_max_ctx)
    else:
        max_ctx = target_max_ctx or assistant_max_ctx
    if max_ctx is not None and input_len + max_new_tokens > max_ctx:
        raise RuntimeError(
            "Spec-dec prompt length plus max_new_tokens exceeds model context "
            f"(prompt={input_len}, max_new_tokens={max_new_tokens}, max_ctx={max_ctx})."
        )
    stats = _SpecDecStats()
    tracker = _SpecDecStatsTracker(stats, expected_assistant=assistant_model)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=input_ids.device)
    assistant_cfg = getattr(assistant_model, "generation_config", None)
    sentinel = object()
    saved_num = sentinel
    saved_schedule = sentinel
    saved_threshold = sentinel
    try:
        gen_cfg = copy.deepcopy(target_model.generation_config)
    except Exception as exc:
        raise RuntimeError("Spec-dec requires a target generation_config.") from exc
    gen_cfg.max_new_tokens = int(max_new_tokens)
    gen_cfg.max_length = None
    gen_cfg.min_length = 0
    gen_cfg.min_new_tokens = None
    if assistant_cfg is not None:
        if hasattr(assistant_cfg, "num_assistant_tokens"):
            saved_num = assistant_cfg.num_assistant_tokens
            assistant_cfg.num_assistant_tokens = max(1, int(num_assistant_tokens))
        if hasattr(assistant_cfg, "num_assistant_tokens_schedule"):
            saved_schedule = assistant_cfg.num_assistant_tokens_schedule
            assistant_cfg.num_assistant_tokens_schedule = "constant"
        if hasattr(assistant_cfg, "assistant_confidence_threshold"):
            saved_threshold = assistant_cfg.assistant_confidence_threshold
            assistant_cfg.assistant_confidence_threshold = 0.0
    try:
        with tracker:
            if not tracker.installed:
                raise RuntimeError(
                    "Spec-dec tracking unavailable: GenerationMixin._get_candidate_generator not found."
                )
            with torch.inference_mode():
                target_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=True,
                    temperature=1.0,
                    top_k=0,
                    top_p=1.0,
                    assistant_model=assistant_model,
                    use_cache=True,
                    generation_config=gen_cfg,
                )
    finally:
        if assistant_cfg is not None:
            if saved_num is not sentinel:
                assistant_cfg.num_assistant_tokens = saved_num
            if saved_schedule is not sentinel:
                assistant_cfg.num_assistant_tokens_schedule = saved_schedule
            if saved_threshold is not sentinel:
                assistant_cfg.assistant_confidence_threshold = saved_threshold
    if not tracker.saw_assisted:
        raise RuntimeError("Spec-dec did not engage assisted generation with the masked model.")
    if stats.proposed <= 0:
        raise RuntimeError(
            "Spec-dec produced no candidate tokens; check input length, max_new_tokens, and model context."
        )
    return stats.accepted, stats.proposed



def score_candidate(
    model: nn.Module,
    baseline_model: nn.Module | None,
    baseline_cache: Dict[bytes, Any] | None,
    search_batches: Sequence[Any],
    observer_data: Dict[int, Dict[str, Any]],
    prune_plan: Dict[int, int],
    prune_args,
    spec_dec_chunk_size: int,
    fitness_mode: str,
    device: torch.device,
    tokenizer=None,
    example_preview=None,
    log_example: bool = False,
    example_max_new_tokens: int = 64,
    esp_p_max_new_tokens: int | None = None,
    spec_dec_oom_log_path: str | Path | None = None,
    spec_dec_oom_context: Dict[str, Any] | None = None,
) -> Tuple[float, Dict[str, str] | None]:
    from reap.main import _preview_experts_to_prune

    pruned_indices = _preview_experts_to_prune(observer_data, prune_plan, prune_args)
    handles = _install_temp_router_masks(model, pruned_indices)
    model.eval()
    if baseline_model is not None:
        baseline_model.eval()
    total_loss = 0.0
    total_tokens = 0
    spec_proposed = 0
    spec_accepted = 0
    example_log = None
    esp_p_logged = False
    batch_iter: Sequence[Any] | Any = search_batches
    oom_log_path = Path(spec_dec_oom_log_path) if spec_dec_oom_log_path else None
    oom_context = spec_dec_oom_context or {}
    try:
        for idx, batch in enumerate(batch_iter):
            if isinstance(batch, torch.Tensor):
                input_ids = batch.to(device)
                attention_mask = None
                labels_mask = None
            elif isinstance(batch, dict):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                labels_mask = batch.get("labels_mask")
                if labels_mask is not None:
                    labels_mask = labels_mask.to(device)
            else:
                # BatchEncoding or similar
                input_ids = batch.input_ids.to(device)
                attention_mask = getattr(batch, "attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                labels_mask = getattr(batch, "labels_mask", None)
                if labels_mask is not None:
                    labels_mask = labels_mask.to(device)
            if fitness_mode == "spec-dec":
                if baseline_model is None:
                    raise ValueError("baseline_model is required for spec-dec fitness.")
                # ensure batch dimension
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                    if attention_mask is not None:
                        attention_mask = attention_mask.unsqueeze(0)
                bs = input_ids.shape[0]
                chunk_size = max(1, int(spec_dec_chunk_size))
                for b in range(bs):
                    seq = input_ids[b : b + 1]
                    attn = attention_mask[b : b + 1] if attention_mask is not None else None
                    try:
                        accepted, proposed = _spec_dec_stats_transformers(
                            target_model=baseline_model,
                            assistant_model=model,
                            input_ids=seq,
                            attention_mask=attn,
                            max_new_tokens=example_max_new_tokens,
                            num_assistant_tokens=chunk_size,
                        )
                    except Exception as exc:
                        if _is_cuda_oom(exc):
                            logger.warning(
                                "Spec-dec OOM on batch %d item %d; skipping. %s",
                                idx,
                                b,
                                exc,
                            )
                            if oom_log_path is not None:
                                try:
                                    payload = dict(oom_context)
                                    payload.update(
                                        {
                                            "time": time.time(),
                                            "batch_idx": int(idx),
                                            "item_idx": int(b),
                                            "prompt_len": int(seq.shape[1]),
                                            "max_new_tokens": int(example_max_new_tokens),
                                            "num_assistant_tokens": int(chunk_size),
                                            "device": str(seq.device),
                                            "error": repr(exc),
                                        }
                                    )
                                    _append_jsonl_log(
                                        oom_log_path,
                                        payload,
                                    )
                                except Exception as log_exc:
                                    logger.warning("Failed to write spec-dec OOM log: %s", log_exc)
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue
                        raise
                    finally:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    spec_accepted += int(accepted)
                    spec_proposed += int(proposed)
            elif fitness_mode in {"kl-baseline", "kl-pq-b", "esp-dataset", "sp-dataset"}:
                if baseline_model is None and (baseline_cache is None or not baseline_cache):
                    raise ValueError("Baseline logits cache is empty and no baseline_model provided for baseline-dependent fitness.")
                if labels_mask is None:
                    labels_mask = torch.ones_like(input_ids)
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids, device=input_ids.device)
                # cache key: tensors content hash
                key = (
                    input_ids.detach().cpu().numpy().tobytes()
                    + attention_mask.detach().cpu().numpy().tobytes()
                )
                if baseline_cache is not None and key in baseline_cache:
                    base_logits = baseline_cache[key].to(device)
                elif baseline_model is not None:
                    with torch.inference_mode():
                        base_logits = baseline_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        ).logits
                    if baseline_cache is not None:
                        baseline_cache[key] = base_logits.detach().cpu()
                else:
                    raise ValueError("Baseline logits for batch not precomputed and no baseline_model available.")
                with torch.inference_mode():
                    cand_logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    ).logits
                mask = attention_mask.bool() & (labels_mask.to(device).bool())
                if fitness_mode == "kl-baseline":
                    # KL(cand || base) over masked tokens
                    base_log_probs = torch.log_softmax(base_logits, dim=-1)
                    cand_log_probs = torch.log_softmax(cand_logits, dim=-1)
                    cand_probs = torch.exp(cand_log_probs)
                    kl = torch.sum(
                        cand_probs * (cand_log_probs - base_log_probs), dim=-1
                    )
                    kl = kl * mask
                    total_loss += float(kl.sum().item())
                    total_tokens += int(mask.sum().item())
                elif fitness_mode == "kl-pq-b":
                    # KL(base || cand) over masked tokens
                    base_log_probs = torch.log_softmax(base_logits, dim=-1)
                    cand_log_probs = torch.log_softmax(cand_logits, dim=-1)
                    base_probs = torch.exp(base_log_probs)
                    kl = torch.sum(
                        base_probs * (base_log_probs - cand_log_probs), dim=-1
                    )
                    kl = kl * mask
                    total_loss += float(kl.sum().item())
                    total_tokens += int(mask.sum().item())
                elif fitness_mode == "esp-dataset":
                    # Expected speculative acceptance (overlap of next-token distributions).
                    base_probs = torch.softmax(base_logits, dim=-1)
                    cand_probs = torch.softmax(cand_logits, dim=-1)
                    overlap = torch.sum(torch.minimum(base_probs, cand_probs), dim=-1)
                    overlap = overlap * mask
                    total_loss += float(overlap.sum().item())
                    total_tokens += int(mask.sum().item())
                else:
                    # Sampled-token expected acceptance (uses candidate proposal only).
                    base_log_probs = torch.log_softmax(base_logits, dim=-1)
                    cand_log_probs = torch.log_softmax(cand_logits, dim=-1)
                    sampled = torch.distributions.Categorical(logits=cand_log_probs).sample()
                    q_lp = cand_log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
                    p_lp = base_log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
                    log_ratio = p_lp - q_lp
                    acceptance = torch.exp(torch.minimum(log_ratio, torch.zeros_like(log_ratio)))
                    acceptance = acceptance * mask
                    total_loss += float(acceptance.sum().item())
                    total_tokens += int(mask.sum().item())
            elif fitness_mode in {"esp-p", "kl-p"}:
                if baseline_model is None and (baseline_cache is None or not baseline_cache):
                    raise ValueError(f"Baseline cache is empty and no baseline_model is available for {fitness_mode} fitness.")
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids, device=input_ids.device)
                # Cache key for baseline self-generation + teacher-forced scoring.
                key = (
                    input_ids.detach().cpu().numpy().tobytes()
                    + attention_mask.detach().cpu().numpy().tobytes()
                )
                cache_hit = False
                generated: torch.Tensor
                base_logits: torch.Tensor
                if baseline_cache is not None and key in baseline_cache:
                    cached_val = baseline_cache[key]
                    if (
                        isinstance(cached_val, dict)
                        and "generated" in cached_val
                        and "base_logits" in cached_val
                    ):
                        generated = cached_val["generated"].to(device)
                        base_logits = cached_val["base_logits"].to(device)
                        cache_hit = True
                if not cache_hit:
                    if baseline_model is None:
                        raise ValueError("Baseline logits for batch not precomputed and no baseline_model available.")
                    if not esp_p_logged:
                        logger.info(
                            "%s: generating baseline continuation for batch %d (max_new_tokens=%d)",
                            fitness_mode,
                            idx,
                            esp_p_max_new_tokens or example_max_new_tokens,
                        )
                        esp_p_logged = True
                    # Generate continuation with baseline model (teacher forcing on its own output).
                    with torch.inference_mode():
                        generated = baseline_model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=esp_p_max_new_tokens or example_max_new_tokens,
                            do_sample=False,
                        )
                    with torch.inference_mode():
                        base_logits = baseline_model(
                            input_ids=generated,
                            attention_mask=torch.ones_like(generated, device=device),
                        ).logits
                    if baseline_cache is not None:
                        baseline_cache[key] = {
                            "generated": generated.detach().cpu(),
                            "base_logits": base_logits.detach().cpu(),
                        }
                prompt_len = input_ids.shape[1]
                gen_len = generated.shape[1]
                if gen_len <= prompt_len:
                    # Nothing new generated; skip.
                    continue
                # Build mask to score only generated tokens.
                gen_attention = torch.ones_like(generated, device=device)
                score_mask = torch.zeros_like(generated, device=device)
                score_mask[:, prompt_len:] = 1
                with torch.inference_mode():
                    cand_logits = model(
                        input_ids=generated,
                        attention_mask=gen_attention,
                    ).logits
                if fitness_mode == "esp-p":
                    # Expected speculative acceptance (overlap of next-token distributions).
                    base_probs = torch.softmax(base_logits, dim=-1)
                    cand_probs = torch.softmax(cand_logits, dim=-1)
                    overlap = torch.sum(torch.minimum(base_probs, cand_probs), dim=-1)
                    overlap = overlap * score_mask
                    total_loss += float(overlap.sum().item())
                    total_tokens += int(score_mask.sum().item())
                else:
                    # KL(cand || base) over baseline-generated continuation.
                    base_log_probs = torch.log_softmax(base_logits, dim=-1)
                    cand_log_probs = torch.log_softmax(cand_logits, dim=-1)
                    cand_probs = torch.exp(cand_log_probs)
                    kl = torch.sum(
                        cand_probs * (cand_log_probs - base_log_probs), dim=-1
                    )
                    kl = kl * score_mask.to(dtype=kl.dtype)
                    total_loss += float(kl.sum().item())
                    total_tokens += int(score_mask.sum().item())
            else:
                labels = input_ids.clone()
                if attention_mask is not None:
                    labels = labels.masked_fill(attention_mask == 0, -100)
                if fitness_mode == "nll-assistant":
                    if labels_mask is None:
                        labels_mask = torch.ones_like(labels)
                    labels = labels.masked_fill(labels_mask == 0, -100)
                with torch.inference_mode():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                loss = float(outputs.loss)
                token_count = (
                    int(attention_mask.sum().item())
                    if attention_mask is not None
                    else labels.numel()
                )
                total_loss += loss * token_count
                total_tokens += token_count
        if log_example and tokenizer is not None and example_preview is not None:
            try:
                prompt_ids = example_preview["prompt_ids"].to(device)
                attention_mask = torch.ones_like(prompt_ids, device=device)
                with torch.inference_mode():
                    generated = model.generate(
                        input_ids=prompt_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=example_max_new_tokens,
                        do_sample=False,
                    )
                gen_text = tokenizer.decode(
                    generated[0][prompt_ids.shape[1] :], skip_special_tokens=True
                )
                prompt_text = example_preview.get("prompt_text", "")
                target_text = example_preview.get("target_text", None)
                example_log = {
                    "prompt": prompt_text,
                    "target": target_text,
                    "model_answer": gen_text,
                }
            except Exception as e:
                logger.warning("Failed to log example response: %s", e)
    finally:
        _cleanup_handles(handles)
    if fitness_mode == "spec-dec":
        if spec_proposed == 0:
            return 0.0, example_log
        return spec_accepted / spec_proposed, example_log  # higher is better
    if total_tokens == 0:
        # For overlap/acceptance we maximize; for loss metrics, inf is a safe fallback.
        return (0.0 if fitness_mode in {"esp-dataset", "sp-dataset", "esp-p"} else float("inf")), example_log
    return total_loss / total_tokens, example_log


def evolutionary_search(
    model: nn.Module,
    baseline_model: nn.Module | None,
    search_batches: Sequence[Any],
    observer_data: Dict[int, Dict[str, Any]],
    prune_args,
    search_args,
    tokenizer,
    example_preview,
    layers: Sequence[int],
    rng: random.Random,
    device: torch.device,
    min_keep_per_layer: int = 1,
    baseline_cache: Dict[bytes, Any] | None = None,
    esp_p_max_new_tokens: int | None = None,
    resume_state: Dict[str, Any] | None = None,
    checkpoint_dir: str | Path | None = None,
    checkpoint_every: int = 0,
    search_data_fingerprint: str | None = None,
    search_microbatch_size: int | None = None,
    run_name_base: str | None = None,
    history_path: str | Path | None = None,
) -> Tuple[Dict[int, int], float, List[Dict[str, Any]]]:
    """
    Run mutation-only evolutionary search. Returns (best_plan, best_score, history).
    """
    search_start_time = time.time()
    capacities = [
        max(observer_data[layer]["expert_frequency"].shape[0] - min_keep_per_layer, 0)
        for layer in layers
    ]
    totals = [
        observer_data[layer]["expert_frequency"].shape[0]
        for layer in layers
    ]
    budget = search_args.int_sparsity * len(layers)
    total_capacity = sum(capacities)
    if total_capacity < budget:
        raise ValueError(
            f"Insufficient capacity to satisfy budget. total_capacity={total_capacity}, "
            f"budget={budget}, min_keep_per_layer={min_keep_per_layer}"
        )
    parities: List[int] | None = None
    if getattr(search_args, "mutation_even_experts", False):
        parities = _normalize_parities(
            [total % 2 for total in totals],
            capacities,
            budget,
        )
    # Ensure we can seed one slot per init pattern.
    population_size = max(search_args.population_size, len(search_args.init_patterns), 1)
    survivors = max(1, min(search_args.topk, population_size))
    if baseline_cache is None:
        baseline_cache = {}
    maximize_score = search_args.fitness in {"spec-dec", "esp-dataset", "sp-dataset", "esp-p"}
    best_score = float("-inf") if maximize_score else float("inf")
    best_plan = None
    history: List[Dict[str, Any]] = []
    score_cache: Dict[Tuple[int, ...], float] = {}
    population: List[Dict[str, Any]]
    start_gen = 0

    if resume_state:
        missing = [key for key in ("generation", "population", "rng_state") if key not in resume_state]
        if missing:
            raise ValueError(f"resume_state missing required keys: {missing}")
        _validate_resume_state(
            resume_state=resume_state,
            layers=layers,
            capacities=capacities,
            budget=budget,
            min_keep_per_layer=min_keep_per_layer,
            search_args=search_args,
            search_data_fingerprint=search_data_fingerprint,
            population_size=population_size,
            survivors=survivors,
            search_microbatch_size=search_microbatch_size,
            run_name_base=run_name_base,
        )
        history = list(resume_state.get("history") or [])
        score_cache = resume_state.get("score_cache") or {}
        best_score = resume_state.get("best_score", best_score)
        best_plan = resume_state.get("best_plan", best_plan)
        population = resume_state.get("population", [])
        if not isinstance(population, list):
            raise ValueError("resume_state population must be a list")
        search_start_time = resume_state.get("search_start_time", search_start_time)
        if resume_state["rng_state"] is None:
            raise ValueError("resume_state rng_state is None")
        rng.setstate(resume_state["rng_state"])
        start_gen = int(resume_state.get("generation", -1)) + 1
        if len(population) != population_size:
            raise ValueError(
                f"resume_state population_size={len(population)} does not match expected {population_size}"
            )
    else:
        # Seed initial population
        population = []
        for pattern in search_args.init_patterns:
            population.append(
                {
                    "plan": init_pattern(pattern, capacities, budget, rng, parities=parities),
                    "origin": "init",
                }
            )
        uniform_only = bool(search_args.init_patterns) and all(
            pattern == "uniform" for pattern in search_args.init_patterns
        )
        fill_pattern = "uniform" if uniform_only else "random"
        while len(population) < population_size:
            population.append(
                {
                    "plan": init_pattern(fill_pattern, capacities, budget, rng, parities=parities),
                    "origin": "init",
                }
            )

    if parities is not None:
        for idx, entry in enumerate(population):
            plan = entry.get("plan")
            if plan is None or not _plan_matches_parity(plan, parities):
                raise ValueError(
                    f"Population plan {idx} violates parity constraints required by even mode."
                )

    if resume_state and start_gen >= search_args.generations:
        logger.info(
            "Resume checkpoint already reached target generation %d; returning saved results.",
            search_args.generations,
        )
        if best_plan is None and population:
            best_plan = plan_to_dict(population[0]["plan"], layers)
        if history_path is not None:
            _atomic_json_dump(history_path, history)
        return best_plan, best_score, history
    if resume_state:
        logger.info("Resuming evolutionary search from generation %d.", start_gen)

    if checkpoint_every > 0 and checkpoint_dir is None:
        logger.warning("Checkpointing enabled but checkpoint_dir is None; disabling checkpoints.")
        checkpoint_every = 0

    def _checkpoint_payload(generation: int) -> Dict[str, Any]:
        return {
            "generation": generation,
            "population": population,
            "best_score": best_score,
            "best_plan": best_plan,
            "score_cache": score_cache,
            "history": history,
            "rng_state": rng.getstate(),
            "search_start_time": search_start_time,
            "layers": list(layers),
            "capacities": list(capacities),
            "budget": budget,
            "min_keep_per_layer": min_keep_per_layer,
            "population_size": population_size,
            "survivors": survivors,
            "maximize_score": maximize_score,
            "search_args": _filtered_search_args(search_args),
            "search_data_fingerprint": search_data_fingerprint,
            "search_microbatch_size": search_microbatch_size,
            "run_name_base": run_name_base,
            "target_generations": int(search_args.generations),
        }

    for gen in range(start_gen, search_args.generations):
        scored: List[Dict[str, Any]] = []
        for idx, entry in enumerate(population):
            plan = entry["plan"]
            origin = entry.get("origin", "init")
            prune_plan = plan_to_dict(plan, layers)
            plan_key = tuple(int(x) for x in plan)
            if plan_key in score_cache:
                score = score_cache[plan_key]
                example_log = None
            else:
                oom_log_path = None
                if history_path is not None:
                    try:
                        oom_log_path = Path(history_path).parent / "spec_dec_oom.log"
                    except Exception:
                        oom_log_path = None
                score, example_log = score_candidate(
                    model,
                    baseline_model,
                    baseline_cache,
                    search_batches,
                    observer_data,
                    prune_plan,
                    prune_args,
                    spec_dec_chunk_size=search_args.spec_dec_chunk_size,
                    fitness_mode=search_args.fitness,
                    device=device,
                    tokenizer=tokenizer,
                    example_preview=example_preview,
                    log_example=search_args.log_eval_example,
                    example_max_new_tokens=search_args.example_max_new_tokens,
                    esp_p_max_new_tokens=esp_p_max_new_tokens or search_args.esp_p_max_new_tokens,
                    spec_dec_oom_log_path=oom_log_path,
                    spec_dec_oom_context={"generation": int(gen), "individual": int(idx)},
                )
                score_cache[plan_key] = score
            logger.info(
                "Gen %d indiv %d [%s] score=%.4f ratios=%s",
                gen,
                idx,
                "P" if origin == "parent" else ("C" if origin == "child" else "I"),
                score,
                _format_ratio_list(plan, totals),
            )
            if example_log:
                prompt_preview = example_log.get("prompt", "")
                target_preview = example_log.get("target", "")
                model_preview = example_log.get("model_answer", "")
                logger.info(
                    (
                        "Gen %d indiv %d example:\n"
                        "  [Question] %s\n"
                        "  [Answer]   %s\n"
                        "  [Model Output] %s"
                    ),
                    gen,
                    idx,
                    prompt_preview[:500],
                    target_preview[:500] if target_preview is not None else "",
                    model_preview[:500],
                )
            scored.append({"score": score, "plan": plan, "origin": origin})
        scored.sort(key=lambda x: x["score"], reverse=maximize_score)

        if scored:
            candidate_best = scored[0]["score"]
            if (maximize_score and candidate_best > best_score) or (not maximize_score and candidate_best < best_score):
                best_score = candidate_best
                best_plan = plan_to_dict(scored[0]["plan"], layers)
        if scored:
            logger.info(
                "Gen %d: best score=%.4f ratios=%s",
                gen,
                scored[0]["score"],
                _format_ratio_list(scored[0]["plan"], totals),
            )
            logger.info(
                "Gen %d: parents (top %d): %s",
                gen,
                survivors,
                "; ".join(
                    [
                        f"score={item['score']:.4f} ratios={_format_ratio_list(item['plan'], totals)}"
                        for item in scored[:survivors]
                    ]
                ),
            )
        history.append(
            {
                "generation": gen,
                "best_score": scored[0]["score"] if scored else None,
                "best_plan": plan_to_dict(scored[0]["plan"], layers) if scored else None,
                "population": [
                    {
                        "score": item["score"],
                        "plan": plan_to_dict(item["plan"], layers),
                        "origin": item["origin"],
                    }
                    for item in scored
                ],
                "search_start_time": search_start_time,
            }
        )
        if history_path is not None:
            _atomic_json_dump(history_path, history)
        elites = scored[:survivors]
        new_population = [{"plan": item["plan"], "origin": "parent"} for item in elites]
        while len(new_population) < population_size:
            parent = rng.choice(elites)["plan"]
            child, successes, target_mutations, attempts = mutate_plan(
                parent,
                capacities,
                rng=rng,
                max_delta=search_args.mutation_max_delta,
                mutation_times=search_args.mutation_times,
                max_attempts=search_args.mutation_max_attempts,
                even_delta=bool(getattr(search_args, "mutation_even_experts", False)),
            )
            if successes < target_mutations and attempts >= search_args.mutation_max_attempts:
                logger.warning(
                    "Generation %d: mutation only achieved %d/%d valid steps before hitting attempt limit",
                    gen,
                    successes,
                    target_mutations,
                )
            new_population.append({"plan": child, "origin": "child"})
        population = new_population
        if checkpoint_every > 0 and (gen + 1) % checkpoint_every == 0:
            checkpoint_file = Path(checkpoint_dir) / f"search_state_gen_{gen:04d}.pt"
            _atomic_torch_save(checkpoint_file, _checkpoint_payload(gen))
            logger.info("Saved search checkpoint to %s (generation %d).", checkpoint_file, gen)
        if gen == search_args.generations - 1:
            break

    if best_plan is None:
        # Fallback to first population member if everything failed.
        best_plan = plan_to_dict(population[0]["plan"], layers)
    return best_plan, best_score, history
