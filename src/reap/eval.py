import logging
from typing import Tuple
import pathlib
import os
import json
import random
import shutil
import subprocess
import time
import inspect
import requests
from contextlib import contextmanager, nullcontext
import yaml

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, HfArgumentParser
from transformers.utils import is_flash_attn_2_available
from lm_eval import evaluator
from lm_eval.utils import make_table
from evalplus.evaluate import evaluate as evalplus_evaluator

from reap.args import ReapArgs, ModelArgs, EvalArgs
from reap.model_util import (
    patched_model_map,
    MODEL_ATTRS,
    apply_runtime_router_masks,
    verify_runtime_pruning,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _normalize_generation_config(eval_args: EvalArgs) -> dict[str, float | int]:
    """
    Single source-of-truth generation config used across evaluation backends.
    When greedy=True, determinism is enforced (temperature=0, do_sample=False).
    """
    cfg: dict[str, float | int] = {
        "temperature": float(eval_args.temperature),
        "top_p": float(eval_args.top_p),
        "top_k": int(eval_args.top_k),
        "min_p": float(eval_args.min_p),
    }

    if eval_args.greedy:
        return {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
        }
    return cfg


def _format_generation_config_for_log(cfg: dict) -> str:
    try:
        return json.dumps(cfg, sort_keys=True)
    except Exception:
        return str(cfg)


@contextmanager
def ensure_evalscope_deterministic():
    """
    evalscope's fix_do_sample_warning resets temperature to 1.0 when do_sample=False.
    For HF math evals we want deterministic decoding (temperature=0). Temporarily
    monkeypatch the helper to preserve caller-provided temperature/top_p.
    """
    try:
        import evalscope.utils.model_utils as _mu
    except Exception:
        yield
        return

    orig = getattr(_mu, "fix_do_sample_warning", None)

    def _no_op_fix(gen_cfg):
        if getattr(gen_cfg, "temperature", None) == 0:
            gen_cfg.do_sample = False
        # leave temperature/top_p/top_k untouched to honor caller settings

    if orig is None:
        yield
        return

    _mu.fix_do_sample_warning = _no_op_fix
    try:
        yield
    finally:
        _mu.fix_do_sample_warning = orig


@contextmanager
def patch_hf_from_pretrained(apply_pruned_masks: bool, use_server: bool):
    """
    Patch AutoModelForCausalLM.from_pretrained to drop dtype args and optionally
    install runtime router masks for masked checkpoints when running HF evaluations.
    """
    orig_from_pretrained = AutoModelForCausalLM.from_pretrained

    def _hf_from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
        kwargs.pop("dtype", None)
        model = orig_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if apply_pruned_masks and not use_server:
            try:
                path = pathlib.Path(pretrained_model_name_or_path)
                pruned_path = path / "pruned_experts.json"
                if pruned_path.exists():
                    # Only apply runtime masks for known masked layouts:
                    # - legacy hybrid models
                    # - pruned search outputs under pruned_models_searched
                    if "hybrid_models" not in str(path) and "pruned_models_searched" not in str(path):
                        logger.info(
                            "Found pruned_experts.json but model path %s is not hybrid_models or pruned_models_searched; skipping runtime mask hook.",
                            path,
                        )
                        return model
                    with open(pruned_path, "r") as f:
                        pruned = json.load(f)
                    apply_runtime_router_masks(model, pruned)
                    verify_runtime_pruning(model, pruned)
                    logger.info(
                        "Applied runtime router masks (hook installed) from %s", pruned_path
                    )
                else:
                    logger.debug("No pruned_experts.json found at %s; skipping mask hook", pruned_path)
            except Exception as e:
                logger.warning(f"Failed to apply runtime router masks: {e}")
        return model

    AutoModelForCausalLM.from_pretrained = _hf_from_pretrained
    try:
        yield
    finally:
        AutoModelForCausalLM.from_pretrained = orig_from_pretrained


@contextmanager
def patch_evalplus_max_new_tokens(max_new_tokens: int):
    """
    Clamp evalplus decoder max_new_tokens to our eval_max_length without
    modifying the vendored evalplus code.
    """
    try:
        import evalplus.provider as _provider
        import evalplus.codegen as _codegen
    except Exception:
        yield
        return

    orig_make_model = getattr(_provider, "make_model", None)
    codegen_make_model = getattr(_codegen, "make_model", None)
    if orig_make_model is None and codegen_make_model is None:
        yield
        return

    def _wrapped_make_model(*args, **kwargs):
        base_fn = orig_make_model or codegen_make_model
        model = base_fn(*args, **kwargs)
        try:
            model.max_new_tokens = max_new_tokens
            logger.info("Patched evalplus max_new_tokens=%s", max_new_tokens)
        except Exception:
            logger.warning("Failed to set evalplus max_new_tokens; using default.")
        return model

    if orig_make_model is not None:
        _provider.make_model = _wrapped_make_model
    if codegen_make_model is not None:
        _codegen.make_model = _wrapped_make_model
    try:
        yield
    finally:
        if orig_make_model is not None:
            _provider.make_model = orig_make_model
        if codegen_make_model is not None:
            _codegen.make_model = codegen_make_model


def _patch_helm_release_date_parsing() -> None:
    """
    HELM expects `ModelMetadata.release_date: Optional[datetime.date]` when loading
    `helm/config/model_metadata.yaml`. Depending on the active YAML loader semantics,
    `YYYY-MM-DD` scalars can come through as plain strings, which trips dacite type
    validation. Patch HELM at runtime (without modifying the HELM submodule) to accept
    both ISO date strings and `datetime.date`.
    """
    try:
        import dacite
        import yaml
        from datetime import date, datetime
        from typing import Any

        from helm.benchmark import config_registry as cr
        from helm.benchmark import model_metadata_registry as mmr

        if getattr(mmr, "_reap_patched_release_date", False):
            return

        def _parse_date(value: Any) -> date:
            if isinstance(value, datetime):
                return value.date()
            if isinstance(value, date):
                return value
            if isinstance(value, str):
                try:
                    return date.fromisoformat(value)
                except ValueError:
                    return datetime.fromisoformat(value).date()
            raise TypeError(
                f"Expected ISO date string or datetime/date, got {type(value).__name__}: {value!r}"
            )

        def _register_model_metadata_from_path(path: str) -> None:
            with open(path, "r") as f:
                raw = yaml.safe_load(f)
            model_metadata_list = dacite.from_dict(
                mmr.ModelMetadataList,
                raw,
                config=dacite.Config(type_hooks={date: _parse_date}),
            )
            for model_metadata in model_metadata_list.models:
                mmr.register_model_metadata(model_metadata)

        # Patch both:
        # - the function in the defining module, and
        # - the already-imported reference in config_registry (imported via `from ... import ...`).
        mmr.register_model_metadata_from_path = _register_model_metadata_from_path  # type: ignore[assignment]
        cr.register_model_metadata_from_path = _register_model_metadata_from_path  # type: ignore[assignment]
        mmr._reap_patched_release_date = True
        logger.info("Patched HELM release_date parsing to accept ISO date strings.")
    except Exception as e:
        logger.warning("Failed to patch HELM release_date parsing: %s", e)


def prepare_wildbench_config(port: int, served_model_name: str) -> pathlib.Path:
    """
    Ensure a WildBench HELM config directory exists for the given port.
    If a port-specific directory is missing, seed it from the .example
    template and replace the port placeholder.
    """
    config_root = pathlib.Path(__file__).parent.parent.parent / "config"
    target = config_root / f"wildbench_prod_env_{port}"
    template_dir = config_root / "wildbench_prod_env_XXXX.example"
    if not template_dir.exists():
        raise FileNotFoundError(
            f"WildBench template directory {template_dir} not found; cannot prepare config."
        )

    if not target.exists():
        target.mkdir(parents=True, exist_ok=True)
        for src_file in template_dir.iterdir():
            if src_file.is_dir():
                continue
            dest_name = src_file.name.replace(".example", "")
            dest_file = target / dest_name
            try:
                text = src_file.read_text()
                text = text.replace("XXXX", str(port))
                if dest_name == "model_deployments.yaml":
                    # Prefer localhost to avoid ambiguous 0.0.0.0 client URL.
                    text = text.replace(f"http://0.0.0.0:{port}", f"http://127.0.0.1:{port}")
                dest_file.write_text(text)
            except UnicodeDecodeError:
                shutil.copy(src_file, dest_file)

    # Always write explicit configs for:
    # - the evaluated model (vLLM server on this port), and
    # - the WildBench judge deployment (override HELM's built-in GPT-4o deployment to route to the user's provider).
    deployments_file = target / "model_deployments.yaml"
    deployment_engine = served_model_name.split("/", 1)[-1].replace("/", "-")
    deployment_name = f"vllm/{deployment_engine}"
    model_deployments: list[dict] = [
        {
            "name": deployment_name,
            "model_name": served_model_name,
            "tokenizer_name": served_model_name,
            "client_spec": {
                "class_name": "helm.clients.vllm_client.VLLMChatClient",
                "args": {"base_url": f"http://127.0.0.1:{port}/v1/"},
            },
        }
    ]

    # Override the built-in WildBench judge deployment. WildBenchAnnotator hardcodes
    # `model_deployment="openai/gpt-4o-2024-05-13"`, so we redefine that deployment here
    # to call an OpenAI-compatible endpoint (e.g. NVIDIA Integrate / HF Router) with a
    # user-provided `openai_model_name`.
    try:
        credentials_path = target / "credentials.conf"
        judge_base_url = None
        judge_model_name = None
        if credentials_path.exists():
            from pyhocon import ConfigFactory

            creds = ConfigFactory.parse_string(credentials_path.read_text())
            judge_base_url = creds.get("openaiBaseUrl")
            judge_model_name = creds.get("wildbenchJudgeOpenAIModelName")
        if isinstance(judge_model_name, str) and judge_model_name.strip():
            judge_args: dict = {"openai_model_name": judge_model_name.strip()}
            if isinstance(judge_base_url, str) and judge_base_url.strip():
                # If present, set base_url explicitly (so configs are self-contained).
                judge_args["base_url"] = judge_base_url.strip()
            model_deployments.append(
                {
                    "name": "openai/gpt-4o-2024-05-13",
                    "model_name": "openai/gpt-4o-2024-05-13",
                    "tokenizer_name": "openai/o200k_base",
                    "client_spec": {
                        "class_name": "helm.clients.openai_client.OpenAIClient",
                        "args": judge_args,
                    },
                }
            )
        else:
            logger.warning(
                "WildBench judge override not written: set `wildbenchJudgeOpenAIModelName` in %s",
                credentials_path,
            )
    except Exception as e:
        logger.warning("Failed to configure WildBench judge deployment override: %s", e)

    deployments_payload = {"model_deployments": model_deployments}
    deployments_file.write_text(
        yaml.safe_dump(deployments_payload, default_flow_style=False, sort_keys=False)
    )

    # Ensure tokenization works for the evaluated model deployment by registering a tokenizer config.
    # (VLLMChatClient inherits OpenAIClient and requires a tokenizer instance.)
    tokenizer_configs_payload = {
        "tokenizer_configs": [
            {
                "name": served_model_name,
                "tokenizer_spec": {
                    "class_name": "helm.tokenizers.huggingface_tokenizer.HuggingFaceTokenizer",
                    "args": {
                        "pretrained_model_name_or_path": served_model_name,
                        "trust_remote_code": True,
                    },
                },
            }
        ]
    }
    (target / "tokenizer_configs.yaml").write_text(
        yaml.safe_dump(tokenizer_configs_payload, default_flow_style=False, sort_keys=False)
    )

    # Write model metadata so `model=<served_model_name>` works cleanly and to avoid stale template entries.
    creator_org = served_model_name.split("/", 1)[0] if "/" in served_model_name else "local"
    model_metadata_payload = {
        "models": [
            {
                "name": served_model_name,
                "display_name": served_model_name,
                "description": "Local evaluation model served via vLLM.",
                "creator_organization_name": creator_org,
                "access": "open",
                "release_date": None,
                "tags": ["TEXT_MODEL_TAG", "LIMITED_FUNCTIONALITY_TEXT_MODEL_TAG"],
            }
        ]
    }
    (target / "model_metadata.yaml").write_text(
        yaml.safe_dump(model_metadata_payload, default_flow_style=False, sort_keys=False)
    )

    logger.info("Prepared WildBench config at %s from template %s", target, template_dir)
    return target


def maybe_materialize_structural_nonuniform(model_path: str, apply_pruned_masks: bool):
    """
    If we are in structural non-uniform mode but the checkpoint was saved without the
    custom non-uniform class, rebuild a structural checkpoint from pruned_experts.json.
    Returns the path to a materialized checkpoint (or None) without mutating the original
    masked checkpoint. Materialized checkpoints are cached in a sibling directory with a
    '-struct' suffix.
    """
    if apply_pruned_masks:
        return None
    path = pathlib.Path(model_path)
    # Only allow structural non-uniform materialization for search outputs.
    # Plain pruned_models should stay in the base Olmoe class to remain vLLM-compatible.
    if "pruned_models_searched" not in str(path):
        return None
    pruned_json = path / "pruned_experts.json"
    if not pruned_json.exists() or not path.is_dir():
        return None

    try:
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    except Exception as e:
        logger.warning("Failed to load config from %s: %s", path, e)
        return None

    model_type = getattr(config, "model_type", "")
    # Already structural: skip.
    if getattr(config, "num_experts_per_layer", None) is not None:
        return None
    if model_type not in {"olmoe", "ernie4_5_moe"}:
        return None

    with open(pruned_json, "r") as f:
        pruned_experts = json.load(f)

    num_layers = getattr(config, "num_hidden_layers", None)
    if num_layers is None:
        logger.warning("Config at %s missing num_hidden_layers; cannot materialize structural non-uniform checkpoint.", path)
        return None
    num_layers = int(num_layers)

    def _is_ernie_moe_layer(cfg, layer_idx: int) -> bool:
        try:
            end_idx = int(cfg.moe_layer_end_index)
            if end_idx == -1:
                end_idx = int(cfg.num_hidden_layers) - 1
            return (
                ((layer_idx + 1) % int(cfg.moe_layer_interval) == 0)
                and layer_idx >= int(cfg.moe_layer_start_index)
                and layer_idx <= end_idx
            )
        except Exception:
            return False

    if model_type == "olmoe":
        num_experts = getattr(config, "num_experts", None)
        if num_experts is None:
            logger.warning(
                "Config at %s missing num_experts; cannot materialize structural non-uniform checkpoint.", path
            )
            return None
        num_experts = int(num_experts)
        retained_per_layer = []
        for idx in range(num_layers):
            pruned = pruned_experts.get(str(idx), [])
            retained_per_layer.append(num_experts - len(pruned))
    else:
        base_num_experts = getattr(config, "moe_num_experts", None)
        if base_num_experts is None:
            logger.warning(
                "Config at %s missing moe_num_experts; cannot materialize structural non-uniform checkpoint.", path
            )
            return None
        base_num_experts = int(base_num_experts)
        retained_per_layer = [base_num_experts] * num_layers
        for idx in range(num_layers):
            if not _is_ernie_moe_layer(config, idx):
                continue
            pruned = pruned_experts.get(str(idx), [])
            retained_per_layer[idx] = base_num_experts - len(pruned)

    logger.info(
        "Materializing structural non-uniform checkpoint from %s using pruned_experts.json",
        path,
    )
    base_model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
    if model_type == "olmoe":
        # Late import to avoid circular dependency with non_uniform pruning pipeline
        from reap.non_uni_prune import _build_nonuniform_config, _structural_prune_olmoe
        from reap.models.non_uniform.olmoe.modeling_olmoe_nonuniform import (
            NonUniformOlmoeForCausalLM,
        )

        nu_config = _build_nonuniform_config(config, [num_experts] * num_layers)
        nu_model = NonUniformOlmoeForCausalLM(nu_config)
        nu_model.load_state_dict(base_model.state_dict(), strict=False)
        _structural_prune_olmoe(nu_model, pruned_experts, retained_per_layer)
        nu_model.config.num_experts_per_layer = retained_per_layer
        nu_model.config.num_experts_per_tok = min(
            getattr(nu_model.config, "num_experts_per_tok", nu_model.config.num_experts),
            min(retained_per_layer),
        )
        copy_files = ["__init__.py", "configuration_olmoe_nonuniform.py", "modeling_olmoe_nonuniform.py"]
        code_dir = "olmoe"
    else:
        from reap.non_uni_prune import _build_nonuniform_ernie_config, _structural_prune_ernie4_5_moe
        from reap.models.non_uniform.ernie4_5_moe.modeling_ernie4_5_moe_nonuniform import (
            NonUniformErnie4_5_MoeForCausalLM,
        )

        nu_config = _build_nonuniform_ernie_config(config, [base_num_experts] * num_layers)
        nu_model = NonUniformErnie4_5_MoeForCausalLM(nu_config)
        nu_model.load_state_dict(base_model.state_dict(), strict=False)
        _structural_prune_ernie4_5_moe(nu_model, pruned_experts, retained_per_layer)
        nu_model.config.num_experts_per_layer = retained_per_layer
        moe_retained = [kept for idx, kept in enumerate(retained_per_layer) if _is_ernie_moe_layer(config, idx)]
        if moe_retained:
            nu_model.config.moe_k = min(int(getattr(nu_model.config, "moe_k", 1)), int(min(moe_retained)))
        copy_files = [
            "__init__.py",
            "configuration_ernie4_5_moe_nonuniform.py",
            "modeling_ernie4_5_moe_nonuniform.py",
        ]
        code_dir = "ernie4_5_moe"

    # Persist a structural checkpoint under the given directory to avoid regenerating
    target_dir = path / "structural_nonuniform"
    if target_dir.exists():
        logger.info("Structural checkpoint already exists at %s; reusing.", target_dir)
        return target_dir.__str__()

    target_dir.mkdir(parents=True, exist_ok=True)
    # Ensure custom modeling files are present for HF loaders
    src_dir = pathlib.Path(__file__).resolve().parent / "models" / "non_uniform" / code_dir
    for fname in copy_files:
        shutil.copy(src_dir / fname, target_dir / fname)

    nu_model.save_pretrained(target_dir)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tokenizer.save_pretrained(target_dir)
    shutil.copy(pruned_json, target_dir / pruned_json.name)
    logger.info("Structural checkpoint saved to %s for evaluation", target_dir)
    return target_dir.__str__()


def get_original_model_name(model_name: str) -> Tuple[str, bool]:
    original_model_name_map = {
        "Mixtral-8x7B-Instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Qwen3-30B-A3B": "Qwen/Qwen3-30B-A3B",
        "Llama-4-Scout-17B-16E-Instruct": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "ERNIE-4.5-21B-A3B-PT": "baidu/ERNIE-4.5-21B-A3B-PT",
        "DeepSeek-V2-Lite-Chat": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "Qwen3-Coder-30B-A3B-Instruct": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "Qwen3-30B-A3B-Instruct-2507": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "gpt-oss-20b": "openai/gpt-oss-20b",
        "gpt-oss-120b": "openai/gpt-oss-120b",
        "GLM-4.5-Air": "zai-org/GLM-4.5-Air",
        "Qwen3-Coder-480B-A35B-Instruct-FP8": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
        # HELM model names are case-sensitive and use the canonical lowercase HF id.
        "OLMoE-1B-7B-0125-Instruct": "allenai/olmoe-1b-7b-0125-instruct",
    }

    original_model = None
    for key, value in original_model_name_map.items():
        if key in model_name:
            original_model = value
            break
    uncompressed_model = False
    if original_model is None:
        # it's an uncompressed model or bad path
        if model_name in original_model_name_map.values():
            original_model = model_name
            uncompressed_model = True
        else:
            logger.warning(
                f"Could not find original model for {model_name}, using model_name as original_model"
            )
            original_model = model_name
    return original_model, uncompressed_model


def wait_for_server(base_url, timeout=1200, check_interval=5):
    """Wait for the server to be ready by checking the health endpoint."""
    health_url = f"{base_url}/health"
    start_time = time.time()

    logger.info(f"Waiting for server to be ready at {health_url}")

    while time.time() - start_time < timeout:
        try:
            response = requests.get(health_url, timeout=10)
            if response.status_code == 200:
                logger.info("Server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass

        logger.info(f"Server not ready yet, waiting {check_interval} seconds...")
        time.sleep(check_interval)

    raise TimeoutError(f"Server did not become ready within {timeout} seconds")


def start_server(model_name, model_args, eval_args, seed, log_file, port, max_model_len: int):
    """Starts a VLLM server for the specified model."""

    num_gpus = torch.cuda.device_count()
    logger.info("Running on %d GPUs", num_gpus)

    gen_cfg = _normalize_generation_config(eval_args)
    override_generation_config = {
        "temperature": gen_cfg["temperature"],
        "top_p": gen_cfg["top_p"],
        "top_k": gen_cfg["top_k"],
        "min_p": gen_cfg["min_p"],
    }
    logger.info(
        "Using %s generation_config=%s",
        "greedy" if eval_args.greedy else "sampling",
        _format_generation_config_for_log(override_generation_config),
    )
    override_generation_config_str = json.dumps(override_generation_config)

    hf_overrides = {}
    max_num_seqs = 32
    gpu_memory_utilization = 0.90
    if model_args.num_experts_per_tok_override is not None:
        logger.info(
            f"Overriding number of experts per token to {model_args.num_experts_per_tok_override}"
        )
        key = "num_experts_per_tok"
        if "ernie" in model_name.lower():
            key = "moe_k"
        hf_overrides = {key: model_args.num_experts_per_tok_override}
    hf_overrides_str = json.dumps(hf_overrides)
    max_num_batched_tokens = max_model_len

    original_model_name, _ = get_original_model_name(model_name)

    # TODO: once  limit_mm_per_prompt={"image": 1 if use_image else 0}, is stable, use
    # in place of patching vllm.model_executor.models.registry for Llama4.

    server_command = [
        "vllm",
        "serve",
        model_name,
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--tensor-parallel-size",
        str(num_gpus),
        # str(1),
        "--seed",
        str(seed),
        "--port",
        str(port),
        # "--enable-expert-parallel",
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--override-generation-config",
        override_generation_config_str,
        "--trust-remote-code",
        "--hf-overrides",
        hf_overrides_str,
        "--served-model-name",  # for HELM eval.
        original_model_name,
        model_name,
    ]

    logger.info(f"Starting VLLM OpenAI API server for {model_name} on port {port}")
    logger.info(
        f"Using {num_gpus} GPUs with {gpu_memory_utilization} memory utilization"
    )
    logger.info(f"Command: {' '.join(server_command)}")

    # Start the server process with log redirection
    with open(log_file, "w") as log_file:
        process = subprocess.Popen(
            server_command, stdout=log_file, stderr=subprocess.STDOUT
        )

    # base_url = f"http://0.0.0.0:{port}"
    base_url = f"http://127.0.0.1:{port}"  # instead of 0.0.0.0

    wait_for_server(base_url)

    return base_url, process


def _get_model_context_len(model_name: str) -> int | None:
    try:
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        return None
    for attr in ("max_position_embeddings", "max_sequence_length", "model_max_length", "max_length"):
        val = getattr(cfg, attr, None)
        if val:
            try:
                return int(val)
            except Exception:
                continue
    return None

def run_evaluate(model_args, results_dir, eval_args, seed):
    def _as_bool(val):
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ("1", "true", "yes", "y", "t")
        return bool(val)

    model_name = model_args.model_name
    if isinstance(model_name, pathlib.Path):
        model_name = model_name.__str__()
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    _seed_everything(seed)
    use_server = _as_bool(eval_args.use_server)
    apply_pruned_masks = _as_bool(eval_args.apply_pruned_masks)
    run_lm_eval = _as_bool(eval_args.run_lm_eval)
    run_evalplus = _as_bool(eval_args.run_evalplus)
    run_livecodebench = _as_bool(eval_args.run_livecodebench)
    run_math = _as_bool(eval_args.run_math)
    run_wildbench = _as_bool(eval_args.run_wildbench)
    server_endpoint = None
    base_url = None
    process = None
    logger.info("Evaluation backend: %s", "vLLM server" if use_server else "HF local (no server)")
    backend_dir_name = "eval_vllm" if use_server else "eval_hf"
    if results_dir is None:
        model_short_name = model_name.split("/")[-1]
        if model_args.num_experts_per_tok_override is not None:
            model_short_name += (
                f"-num_experts_per_tok_{model_args.num_experts_per_tok_override}"
            )
        results_dir = pathlib.Path.cwd() / "artifacts" / backend_dir_name / model_short_name
    if isinstance(results_dir, str):
        results_dir = pathlib.Path(results_dir)
    if results_dir.name == "eval":
        results_dir = results_dir.with_name(backend_dir_name)
    if not eval_args.greedy:
        results_dir = results_dir.parent / f"{results_dir.name}_sampling"
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {results_dir}")
    num_gpus = torch.cuda.device_count()
    hf_parallelize = (not use_server) and num_gpus > 1
    if hf_parallelize:
        logger.info("HF eval: enabling model parallel across %d GPUs", num_gpus)
    model_name = patched_model_map(model_name)
    # For vLLM runs, always attempt to materialize a structural checkpoint so the
    # server can import the custom non-uniform modeling code via trust_remote_code.
    maybe_structural_model = maybe_materialize_structural_nonuniform(
        model_name, apply_pruned_masks if not use_server else False
    )
    if maybe_structural_model:
        model_name = maybe_structural_model
        logger.info("Using materialized structural non-uniform checkpoint for evaluation: %s", model_name)
    print(f"Using model name {model_name} for evaluation:")
    # eval_max_length controls max new tokens; context cap is eval_model_max_length or observer default.
    max_new_tokens = eval_args.eval_max_length
    if max_new_tokens is None or max_new_tokens <= 0:
        raise ValueError("eval_max_length must be set to a positive integer for evaluations.")
    model_ctx = eval_args.eval_model_max_length
    if model_ctx is None:
        # fallback to model config only
        model_ctx = _get_model_context_len(model_name)
        if model_ctx is None:
            raise ValueError("eval_model_max_length is not set and model context length could not be determined.")
    logger.info("Using eval_max_length=%s (max new tokens), model context=%s", max_new_tokens, model_ctx)
    gen_cfg = _normalize_generation_config(eval_args)
    logger.info("Normalized generation_config for evaluations: %s", _format_generation_config_for_log(gen_cfg))

    if use_server:
        server_endpoint, process = start_server(
            model_name,
            model_args,
            eval_args,
            seed,
            log_file=eval_args.server_log_file_name,
            port=eval_args.vllm_port,
            max_model_len=model_ctx,
        )
        base_url = f"{server_endpoint}/v1"
    lm_eval_apply_chat_template = eval_args.lm_eval_apply_chat_template
    if use_server and lm_eval_apply_chat_template:
        logger.warning(
            "lm-eval chat templating is not supported with the vLLM server backend; disabling apply_chat_template."
        )
        lm_eval_apply_chat_template = False

    hf_patch_context = (
        patch_hf_from_pretrained(apply_pruned_masks, use_server)
        if not use_server
        else nullcontext()
    )

    with hf_patch_context:
        if run_lm_eval:
            results_file_base_name = results_dir / "lm_eval_results"
            lm_model_args = None
            if use_server:
                lm_model_args = {
                    "pretrained": model_name,
                    "tensor_parallel_size": num_gpus,
                    "gpu_memory_utilization": 0.85,
                    "num_concurrent": 32,
                    "timeout": 1200,
                    "max_retries": 10,
                    "trust_remote_code": True,
                    "max_length": model_ctx,
                }
            else:
                # HF backend: keep only HF-friendly keys; drop vLLM/server-specific args.
                lm_model_args = {
                    "pretrained": model_name,
                    "trust_remote_code": True,
                    "max_length": model_ctx,
                }
                if hf_parallelize:
                    lm_model_args["parallelize"] = True
            if "baidu" in model_name.lower():
                logger.warning("Using slow tokenizer for Ernie-4.5")
                lm_model_args["use_fast_tokenizer"] = False
            if use_server:
                lm_model_args["base_url"] = f"{base_url}/completions"
                lm_model_args["tokenized_requests"] = False
            logger.info(f"Running lm-eval on tasks {eval_args.lm_eval_tasks}")
            is_ernie = "ernie" in model_name.lower()
            logger.warning(f"Is Ernie: {is_ernie}, using batch size 1")

            lm_eval_kwargs = {
                "model_args": lm_model_args,
                "tasks": eval_args.lm_eval_tasks,
                "num_fewshot": eval_args.lm_eval_num_fewshot,
                "random_seed": seed,
                "numpy_random_seed": seed,
                "torch_random_seed": seed,
                "apply_chat_template": lm_eval_apply_chat_template,
                "fewshot_as_multiturn": eval_args.lm_eval_fewshot_as_multiturn,
            }
            # Respect eval_max_length for lm-eval if the API supports it.
            simple_eval_sig = inspect.signature(evaluator.simple_evaluate)
            if "max_gen_toks" in simple_eval_sig.parameters:
                lm_eval_kwargs["max_gen_toks"] = max_new_tokens
            elif "max_new_tokens" in simple_eval_sig.parameters:
                lm_eval_kwargs["max_new_tokens"] = max_new_tokens
            elif "max_tokens" in simple_eval_sig.parameters:
                lm_eval_kwargs["max_tokens"] = max_new_tokens
            gen_kwargs = {
                "temperature": float(gen_cfg["temperature"]),
                "top_p": float(gen_cfg["top_p"]),
            }
            if "gen_kwargs" in simple_eval_sig.parameters:
                lm_eval_kwargs["gen_kwargs"] = gen_kwargs
            elif "generation_kwargs" in simple_eval_sig.parameters:
                lm_eval_kwargs["generation_kwargs"] = gen_kwargs

            if use_server:
                results = evaluator.simple_evaluate(
                    model="local-completions",
                    batch_size=eval_args.parallel_tasks if not is_ernie else 1,
                    **lm_eval_kwargs,
                )
            else:
                results = evaluator.simple_evaluate(
                    model="hf",
                    batch_size="auto",
                    **lm_eval_kwargs,
                )
            try:
                with open(f"{results_file_base_name}_table.txt", "w") as f:
                    print(make_table(results))
                    print(make_table(results), file=f)
                    if "groups" in results:
                        print(make_table(results, "groups"))
                        print(make_table(results, "groups"), file=f)
                with open(f"{results_file_base_name}.json", "w") as f:
                    json.dump(results, f)
            except Exception as e:
                pass
            logger.info(f"Finished evaluating lm-eval")

        try:
            if run_evalplus:
                with patch_evalplus_max_new_tokens(max_new_tokens):
                    enable_thinking = True
                    if "qwen" in model_name.lower() or "glm-4.5" in model_name.lower():
                        logger.info("Disabling thinking for Qwen/GLM models")
                        enable_thinking = False
                    attn_implementation = "flash_attention_2"
                    if not is_flash_attn_2_available():
                        attn_implementation = "sdpa"
                        logger.info("FlashAttention2 not available; using attn_implementation=%s", attn_implementation)
                    for task in eval_args.evalplus_tasks:
                        logger.info(f"Running evalplus on task {task}")
                        output_file = results_dir / f"{task}.json"
                        # evalplus fork
                        if use_server:
                            evalplus_evaluator(
                                model=model_name,
                                root=results_dir / "evalplus_results",
                                dataset=task,
                                backend="openai",
                                attn_implementation=attn_implementation,
                                bs=eval_args.evalplus_batch_size,
                                greedy=eval_args.greedy,
                                output_file=output_file,
                                trust_remote_code=True,
                                base_url=base_url,
                                temperature=gen_cfg["temperature"],
                                enable_thinking=enable_thinking,
                                parallel_tasks=eval_args.parallel_tasks,
                            )
                        else:
                            evalplus_evaluator(
                                model=model_name,
                                root=results_dir / "evalplus_results",
                                dataset=task,
                                backend="hf",
                                attn_implementation=attn_implementation,
                                bs=eval_args.evalplus_batch_size,
                                greedy=eval_args.greedy,
                                output_file=output_file,
                                trust_remote_code=True,
                                temperature=gen_cfg["temperature"],
                                enable_thinking=enable_thinking,
                                device_map="auto" if hf_parallelize else None,
                            )
        except Exception as e:
            logger.error(f"An error occurred during evalplus: {e}")
            raise e
            pass
        try:
            if run_livecodebench:
                from datetime import datetime
                from lcb_runner.runner.main import main as lcb_main
                from lcb_runner.runner.main import get_args_dict
                from lcb_runner.lm_styles import LanguageModelStore, LanguageModel, LMStyle

                original_model, uncompressed_model = get_original_model_name(model_name)

                if original_model not in LanguageModelStore:
                    LanguageModelStore[original_model] = LanguageModel(
                        original_model,
                        original_model.split("/")[-1],
                        LMStyle.ReapBase,
                        datetime.utcnow(),
                        link=None,
                    )

                if not use_server:
                    raise ValueError(
                        "LiveCodeBench evaluation requires use_server=True (HF evaluation path removed)."
                    )
                if not isinstance(base_url, str) or not base_url:
                    raise ValueError("LiveCodeBench requires a running vLLM server; base_url is missing.")

                lcb_args = get_args_dict(
                    model=original_model,
                    n=1,
                    codegen_n=1,
                    temperature=float(gen_cfg["temperature"]),
                    top_p=float(gen_cfg["top_p"]),
                    output_path=results_dir,
                    enable_thinking=False,
                    base_url=base_url,
                    start_date="2025-01-01",
                    end_date="2025-07-31",
                    evaluate=True,
                    timeout=120,
                    local_model_path=model_name if not uncompressed_model else None,
                    max_tokens=max_new_tokens,
                )
                logger.info(f"Running LiveCodeBench with args: {lcb_args}")
                lcb_main(lcb_args)
                logger.info(f"Finished evaluating LiveCodeBench")
        except Exception as e:
            logger.error(f"An error occurred during livecodebench: {e}")
            pass
        if run_math:
            try:
                from evalscope.constants import EvalType
                from evalscope.run import run_task, TaskConfig

                eval_type = EvalType.SERVICE if use_server else EvalType.CHECKPOINT
                api_url = base_url if use_server else None
                model_args = {}
                generation_config = {
                    "do_sample": not eval_args.greedy,
                    "temperature": float(gen_cfg["temperature"]),
                    "top_p": float(gen_cfg["top_p"]),
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                if use_server:
                    # vLLM OpenAI path expects max_tokens; include max_new_tokens as a fallback.
                    generation_config["max_tokens"] = max_new_tokens
                    generation_config["max_new_tokens"] = max_new_tokens
                else:
                    model_args["model_max_length"] = model_ctx
                    generation_config["max_length"] = model_ctx
                    generation_config["max_new_tokens"] = max_new_tokens

                task_config = TaskConfig(
                    model=model_name,
                    model_args=model_args,
                    generation_config=generation_config,
                    datasets=eval_args.math_datasets,
                    api_url=api_url,
                    api_key="EMPTY",
                    timeout=3600,
                    work_dir=results_dir / "evalscope_results",
                    dataset_args={
                        "gsm8k": {"few_shot_num": 0},
                    },
                    eval_batch_size=8,
                    # eval_batch_size=16,
                    # eval_batch_size=32,
                    eval_type=eval_type,
                )
                logger.info(f"Running evalscope math with config: {task_config}")
                with ensure_evalscope_deterministic():
                    run_task(task_config)
                # if use_server:
                #     run_task(task_config)
                # else:
                #     # HF path: patch evalscope to keep deterministic generation (temperature=0)
                #     with ensure_evalscope_deterministic():
                #         run_task(task_config)
                logger.info(f"Finished evaluating evalscope math benchmarks")
            except Exception as e:
                logger.error(f"An error occurred during math evaluation: {e}")
                pass
        try:
            if run_wildbench:
                from helm.benchmark.run import helm_run, create_helm_run_args
                from helm.common.hierarchical_logger import setup_default_logging

                original_model, uncompressed_model = get_original_model_name(model_name)

                suite = "test"
                _patch_helm_release_date_parsing()
                # HELM's built-in WildBench RunSpec sets max_tokens=16384; override it with
                # our evaluation max new tokens (consistent with the other eval paths here).
                wildbench_max_tokens = int(max_new_tokens)
                logger.info(
                    "WildBench max_tokens override: %s (eval_max_length=%s, model_ctx=%s)",
                    wildbench_max_tokens,
                    max_new_tokens,
                    model_ctx,
                )
                if use_server:
                    # vLLM server path: copy env configs and point HELM to the server port
                    config_src = prepare_wildbench_config(eval_args.vllm_port, original_model)
                    local_path = f"{results_dir}/wildbench_prod_env_{eval_args.vllm_port}"
                    shutil.copytree(config_src, local_path, dirs_exist_ok=True)

                    run_entries = [
                        f"wildbench:subset=v2,num_output_tokens={wildbench_max_tokens},model={original_model}"
                    ]
                    helm_args = create_helm_run_args(
                        suite=suite,
                        local_path=local_path,
                        run_entries=run_entries,
                        output_path=f"{results_dir}/wildbench",
                        cache_instances=True,
                        disable_cache=False,
                        num_threads=eval_args.wildbench_num_threads
                        if eval_args.wildbench_num_threads is not None
                        else 4,
                    )
                else:
                    # HF path: register local HF model with HELM
                    helm_model_name = f"huggingface/{pathlib.Path(model_name).name}"
                    run_entries = [
                        f"wildbench:subset=v2,num_output_tokens={wildbench_max_tokens},model={helm_model_name}"
                    ]
                    helm_args = create_helm_run_args(
                        suite=suite,
                        run_entries=run_entries,
                        output_path=f"{results_dir}/wildbench",
                        cache_instances=True,
                        disable_cache=False,
                        enable_local_huggingface_models=[model_name],
                        num_threads=eval_args.wildbench_num_threads
                        if eval_args.wildbench_num_threads is not None
                        else 4,
                    )
                logger.info(f"Running WildBench with args: {helm_args}")
                setup_default_logging()
                helm_run(helm_args)
                logger.info("Finished evaluating WildBench")
        except Exception as e:
            logger.error(f"An error occurred during wildbench: {e}")
            pass

    if use_server and process is not None:
        process.terminate()


if __name__ == "__main__":
    parser = HfArgumentParser((ReapArgs, ModelArgs, EvalArgs))
    reap_args, model_args, eval_args = parser.parse_args_into_dataclasses()
    run_evaluate(
        model_args=model_args,
        results_dir=eval_args.results_dir,
        eval_args=eval_args,
        seed=reap_args.seed,
    )
