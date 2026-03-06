from dataclasses import dataclass, field
import dotenv
import os

dotenv.load_dotenv()

DEFAULT_INIT_PATTERNS = (
    "uniform",
    "random",
    "up",
    "down",
    "mid",
    "mid-neg",
    "front-heavy",
    "back-heavy",
    "block-early",
    "block-late",
    "alt",
    "deep-flat",
)


@dataclass
class ReapArgs:
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility."})  # 11, 99
    debug: bool = field(
        default=False, metadata={"help": "Enable debug mode for more verbose output."}
    )
    profile: bool = field(
        default=True, metadata={"help": "Enable profiling prior to run to avoid OOM."}
    )
    run_observer_only: bool = field(
        default=False,
        metadata={"help": "Whether to only run the observer to collect activation data."},
    )
    do_eval: bool = field(
        default=True,
        metadata={"help": "Whether to run evaluation after merging experts."},
    )
    plot_clusters: bool = field(
        default=True,
        metadata={
            "help": "Whether to plot clusters after clustering experts. "
        }
    )
    smoke_test: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to run a smoke test on the merged model to ensure it works "
                "as expected prior to saving"
            )
        },
    )


@dataclass
class ModelArgs:
    model_name: str = field(
        default="Qwen/Qwen3-30B-A3B",
        metadata={
            "help": "Name of the model to use.",
            # "choices": [
            #     "mistralai/Mixtral-8x7B-Instruct-v0.1",
            #     "Qwen/Qwen3-30B-A3B",
            #     "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            #     "baidu/ERNIE-4.5-21B-A3B-PT",
            #     "deepseek-ai/DeepSeek-V2-Lite-Chat",
            #     "Qwen/Qwen3-Coder-30B-A3B-Instruct",
            #     "Qwen/Qwen3-30B-A3B-Instruct-2507",
            #     "openai/gpt-oss-20b",
            #     "openai/gpt-oss-120b",
            #     "zai-org/GLM-4.5-Air"
            # ],
        },
    )
    num_experts_per_tok_override: int | None = field(
        default=None,
        metadata={
            "help": (
                "Override the number of experts per token. If None, uses the model's "
                "default number of experts per token."
            )
        },
    )


@dataclass
class DatasetArgs:
    dataset_name: str = field(
        default="theblackcat102/evol-codealpaca-v1",
        metadata={
            "help": "Name of the dataset to use.",
            "choices": [
                "m-a-p/CodeFeedback-Filtered-Instruction",
                "ise-uiuc/Magicoder-Evol-Instruct-110K",
                "allenai/c4",
                "theblackcat102/evol-codealpaca-v1",
                "euclaise/WritingPrompts_curated",
                "allenai/tulu-3-sft-personas-math",
                "combined"
            ],
        },
    )
    dataset_config_name: str = field(
        default="all", metadata={"help": "Configuration name of the dataset."}
    )
    split: str = field(default="train", metadata={"help": "Dataset split to use."})
    shuffle: bool = field(
        default=True, metadata={"help": "Whether to shuffle the dataset."}
    )
    # for SFT only
    dataset_test_split: str = field(default="test", metadata={"help": "Dataset split to use for evaluation."})


@dataclass
class ObserverArgs:
    samples_per_category: int = 1024
    split_by_category: bool = False
    select_only_categories: list[str] | str | None = field(
        default=None,
        metadata={
            "help": (
                "List of categories to select for observation. If None, all categories "
                "are selected."
            )
        },
    )
    model_max_length: int | None = 4096
    return_vllm_tokens_prompt: bool = False
    truncate: bool = False
    overwrite_observations: bool = field(
        default=False,
        metadata={"help": "Whether to overwrite existing observer data files."},
    )
    distance_measure: str = field(
        default="angular",
        metadata={
            "help": "Distance function to use for clustering.",
            "choices": ["angular", "euclidean", "jsd", "cka", "cosine"],
        },
    )
    output_file_name: str = field(
        default=f"observations_1024_cosine.pt",
        metadata={"help": "Name of the output file for observer data."},
    )
    record_pruning_metrics_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only record pruning metrics during observation to reduce "
                "memory usage and wall-clock time."
            )
        },
    )
    renormalize_router_weights: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to renormalize topk router weights to sum to 1 if the model.config.norm_topk_prob is True."
            )
        }, 
    )

@dataclass
class ClusterArgs:
    cluster_description: str | None = field(
        default=None,
        metadata={
            "help": (
                "Description of the clustering run, used for naming output dir. "
                "If None, use: "
                "f'{cluster_args.expert_sim}_{obs_args.distance_measure}_{num_clusters}_"
                "{cluster_args.linkage_method}_freq-penalty-{cluster_args.frequency_penalty}_"
                "softmax-{cluster_args.softmax_temperature}'"
            )
        },
    )
    expert_sim: str = field(
        default="ttm",
        metadata={
            "help": "Expert similiarty method.",
            "choices": [
                "ttm",
                "dynamic_ttm",
                "characteristic_activation",
                "routed_characteristic_activation",
                "router_logits",
                "online_characteristic_activation_dist"
            ],
        },
    )
    compression_ratio: float | None = field(
        default=0.5,
        metadata={
            "help": (
                "Compression ratio for clustering experts. If None, num_clusters must "
                "be set."
            )
        },
    )
    num_clusters: int | None = field(
        default=None,
        metadata={
            "help": (
                "Number of clusters to place experts into per layer. If None, "
                "num_clusters is calculated as int(num_experts * compression_ratio)."
            )
        },
    )
    cluster_method: str = field(
        default="agglomerative",
        metadata={
            "help": "Clustering method to use.",
            "choices": ["agglomerative", "kmeans", "spectral", "mc_smoe"],
        },
    )
    linkage_method: str = field(
        default="average",
        metadata={
            "help": "Linkage method for agglomerative clustering.",
            "choices": ["ward", "complete", "average", "single"],
        },
    )
    frequency_penalty: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to apply frequency penalty to expert similarity scores. "
                "If True, the frequency of each expert is used to scale the similarity"
            )
        },
    )
    softmax_temperature: float | None = field(
        default=None,
        metadata={
            "help": (
                "Temperature for softmax scaling of expert probabilities to calculate "
                "distance penalty vector. If 0 or None, expert probabilites are max "
                "normalized."
            )
        },
    )
    multi_layer: int | None = field(
        default=None,
        metadata={
            "help": (
                "Number of layers to merge at once. If None, merges all layers "
                "separately."
            )
        },
    )
    max_cluster_size: int | None = field(
        default=None,
        metadata={
            "help": (
                "If not None, maximum number of experts per cluster. Only agglomerative"
                " cluster method supported"
            )
        }
    )
    singleton_super_experts: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to keep super experts in a singleton when clustering"
            )
        }
    )
    singleton_outlier_experts: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to keep outlier experts in a singleton when clustering"
            )
        }
    )



@dataclass
class MergeArgs:
    overwrite_merged_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to overwrite existing merged model files."
        },
    )
    merged_model_dir_name: str | None = field(
        default=None,
        metadata={
            "help": (
                "Name of the merged model. If None, uses a concatenation of releveant hyperparameters:"
                "'merge_args.merge_method-merge_args.dom_as_base-merge_args.select_top_k-permute_merge_args.permute'"
            )
        },
    )
    merge_method: str = field(
        default="frequency_weighted_average",
        metadata={
            "help": "Method to use for merging experts.",
            "choices": [
                "frequency_weighted_average",
                "average",
                "ties",
                "multislerp",
                "sce",
                "karcher",
                "submoe"
            ],
        },
    )
    skip_first: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to skip the first layer when merging experts. "
            )
        }
    )
    skip_last: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to skip the last layer when merging experts. "
            )
        }
    )
    dom_as_base: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use the most frequent expert as the base model for "
                "multislerp."
            )
        },
    )
    select_top_k: float = field(
        default=0.1,
        metadata={
            "help": (
                "Top-k percentage of weights to keep in non-dom experts for TIES."
            )
        }
    )
    permute: str | None = field(
        default=None,
        metadata={
            "help": (
                "Permutation to apply prior to merge"
            ),
            "choices": [
                None,
                "direct",
                "wm",
            ]
        },
    )
    save_as_tied_params: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to save the merged model as tied parameters. "
                "If False, saves merged experts as copies."
            )
        },
    )
    save_as_merged_model: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to save the merged model as a custom merged model. "
                "If False, saves merged experts as separate experts."
            )
        },
    )
    hybrid_prune_layers: list[int] = field(
        default_factory=list,
        metadata={
            "help": (
                "Layer indices to prune instead of merge. Uses the same per-layer "
                "removal budget as merging unless overridden by hybrid_prune_counts."
            )
        },
    )
    hybrid_prune_counts: list[int] | None = field(
        default=None,
        metadata={
            "help": (
                "Optional list of experts to prune per layer, aligned with hybrid_prune_layers. "
                "If None, each listed layer prunes the default removal budget derived from compression_ratio/num_clusters."
            )
        },
    )
    hybrid_dry_run: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, print the hybrid prune/merge plan and exit before modifying the model."
            )
        },
    )
    hybrid_router_probe_text: str | None = field(
        default=None,
        metadata={
            "help": (
                "Optional: when set with hybrid_dry_run, run a single forward pass on this text "
                "and report router selections before/after zero-out simulation for the pruned layers."
            )
        },
    )
    hybrid_router_probe_max_length: int = field(
        default=16,
        metadata={
            "help": (
                "Max tokenized length for the hybrid router probe text. "
                "Keep small to avoid heavy probes."
            )
        },
    )

@dataclass
class KdArgs:
    pass


@dataclass
class EvalArgs:
    use_server: bool = field(
        default=True,
        metadata={
            "help": "Whether to use a vllm server for evaluation. If False, uses hf backends."
        },
    )
    greedy: bool = field(
        default=True,
        metadata={
            "help": "Whether to use greedy decoding for evaluation. If False, uses sampling."
        },
    )
    temperature: float = field(
        default=0.7,
        metadata={
            "help": "Temperature for sampling during evaluation. Ignored if greedy=True."
        },
    )
    top_p: float = field(
        default=0.8,
        metadata={"help": "Top-p value for nucleus sampling during evaluation. Ignored if greedy=True."},
    )
    top_k: int = field(
        default=20,
        metadata={
            "help": "Top-k value for top-k sampling during evaluation. Ignored if greedy=True."
        },
    )
    min_p: float = field(
        default=0.00,
        metadata={
            "help": "Minimum probability for sampling during evaluation. Ignored if greedy=True."
        },
    )
    results_dir: str | None = field(
        default=None,
        metadata={
            "help": (
                "Directory to save evaluation results. If None, results are saved "
                "in artifacts/model_name directory."
            )
        },
    )
    run_lm_eval: bool = field(
        default=True,
        metadata={"help": "Whether to run evaluation on the merged model."},
    )
    run_evalplus: bool = field(
        default=True,
        metadata={"help": "Whether to run evaluation using evalplus."},
    )
    run_livecodebench: bool = field(
        default=True,
        metadata={"help": "Whether to run evaluation using livecodebench."},
    )
    run_wildbench: bool = field(
        default=False,
        metadata={"help": "Whether to run evaluation using wildbench."},
    )
    run_math: bool = field(
        default=False,
        metadata={"help": "Whether to run evaluation using math tasks."},
    )
    math_datasets: list[str] = field(
        default_factory=lambda: ["gsm8k", "math_500"],
        # default_factory=lambda: ["math_500"],
        metadata={"help": "List of math datasets to run via evalscope."},
    )
    eval_max_length: int | None = field(
        default=2048,
        metadata={
            "help": (
                "Maximum new tokens to generate during evaluation (lm-eval, evalplus, math, vLLM server). "
                "This controls the completion length; does not auto-clamp to model config."
            )
        },
    )
    eval_model_max_length: int | None = field(
        default=4096,
        metadata={
            "help": (
                "Optional context length cap for evaluation (used as max_length/model_max_length and vLLM max_model_len). "
                "If None, falls back to ObserverArgs.model_max_length."
            )
        },
    )

    lm_eval_tasks: list[str] = field(
        default_factory=lambda: [
            "winogrande",
            "arc_challenge",
            "arc_easy",
            "boolq",
            "hellaswag",
            "mmlu",
            "openbookqa",
            "rte",
        ],
        metadata={
            "help": "List of tasks to evaluate on using lm-eval.",
        },
    )
    evalplus_tasks: list[str] = field(
        default_factory=lambda: [
            "mbpp",
            "humaneval",
        ],
        metadata={
            "help": "List of tasks to evaluate on using evalplus.",
        },
    )
    evalplus_batch_size: int | None = field(
        default=64,
        metadata={
            "help": (
                "Batch size to use for evalplus codegen. If None, uses evalplus default "
                "(min(n_samples, 32)); set explicitly to control GPU memory."
            )
        },
    )
    lm_eval_apply_chat_template: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to apply the tokenizer chat template when running lm-eval. "
                "Enable for instruction/chat models."
            )
        },
    )
    lm_eval_num_fewshot: int = field(
        default=0,
        metadata={
            "help": "Number of few-shot examples to use for lm-eval tasks.",
        },
    )
    lm_eval_fewshot_as_multiturn: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, format few-shot examples as multi-turn conversations when applying chat templates."
            )
        },
    )
    server_log_file_name: str = field(
        default="server.log"
        if os.environ.get("SERVER_LOG_FILE_NAME") is None
        else os.environ["SERVER_LOG_FILE_NAME"],
        metadata={
            "help": "Name of the log file for the evaluation server.",
        },
    )
    vllm_port: int = field(
        default=8000,
        metadata={
            "help": "Port number for vLLM serve"
        },
    )
    force_hf_online: bool = field(
        default=True,
        metadata={
            "help": (
                "If True, unset HF offline env vars (HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE/"
                "HF_DATASETS_OFFLINE) for the vLLM server subprocess so it can download missing files."
            )
        },
    )
    parallel_tasks: int = field(
        default=32,
        metadata={
            "help": "Number of parallel tasks to run during evalplus evaluation."
        },
    )
    livecodebench_batch_size: int | None = field(
        default=8,
        metadata={
            "help": (
                "Deprecated: used only by the removed HF LiveCodeBench shim. "
                "LiveCodeBench now requires use_server=True."
            )
        },
    )
    apply_pruned_masks: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, apply runtime router masks from pruned_experts.json when loading HF models during eval."
            )
        },
    )
    wildbench_num_threads: int | None = field(
        default=4,
        metadata={
            "help": "Override HELM num_threads for WildBench. If None, use HELM default (4)."
        },
    )

@dataclass
class PruneArgs:
    overwrite_pruned_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to overwrite existing pruned model files."
        },
    )
    prune_method: str = field(
        default="frequency",
        metadata={
            "help": "Method to use for pruning experts.",
            "choices": [
                "frequency",
                "ean_ca",
                "ean_sum",
                'ean_mean',
                "weighted_frequency_sum",
                "weighted_expert_frequency_sum",
                "weighted_ean_sum",
                "weighted_ean_sum_l2",
                "reap",
                "reap_l2",
                "max_activations"
            ]
        },
    )
    n_experts_to_prune: int | None = field(
        default=None,
        metadata={
            "help": (
                "Number of experts to keep after pruning. If None, use "
                "--compression-ratio."
            )
        },
    )
    perserve_super_experts: bool = field(
        default=False,
        metadata={
            "help": (
                r"Whether to perserve super experts when pruning. Excludes last 25% of layers"
            )
        }
    )
    perserve_outliers: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to perserve outlier experts when pruning, includes all layers"
            )
        }
    )
    zero_out: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, do not remove experts; zero their weights and set router rows "
                "to a negligible value. Keeps num_experts/slots unchanged."
            )
        },
    )

@dataclass
class QuantizationArgs:
    quantization_method: str = field(
        default="awq",
        metadata={
            "help": "Method to use for quantization.",
            "choices": ["awq","gptq"]
        },
    )
    scheme: str = field(
        default="W4A16",
        metadata={
            "help": "Quantization scheme to use.",
            "choices": ["W4A16",]
        },
    )
    group_size: int = field(
        default=128,
        metadata={
            "help": "Group / block size for quantization.",
        },
    )


@dataclass
class SearchArgs:
    output_dir: str | None = field(
        default=None,
        metadata={
            "help": (
                "Optional explicit output directory for searched/pruned model. "
                "If set, overrides default pruned_models_searched/<run_name>."
            )
        },
    )
    int_sparsity: int = field(
        default=1,
        metadata={
            "help": (
                "Integer sparsity: experts to prune per layer in the uniform baseline "
                "(global budget = int_sparsity * num_layers)."
            )
        },
    )
    population_size: int = field(
        default=8,
        metadata={"help": "Population size for evolutionary search."},
    )
    generations: int = field(
        default=4,
        metadata={"help": "Number of generations to run evolutionary search."},
    )
    topk: int = field(
        default=4,
        metadata={
            "help": "Number of top-performing candidates to keep each generation."
        },
    )
    mutation_times: int = field(
        default=1,
        metadata={
            "help": "Number of valid level-switch mutations to apply per offspring."
        },
    )
    mutation_max_attempts: int = field(
        default=32,
        metadata={
            "help": (
                "Maximum attempts to find valid mutations for a single offspring "
                "before giving up."
            )
        },
    )
    mutation_max_delta: int = field(
        default=2,
        metadata={
            "help": "Maximum experts to shift in a single level-switch mutation."
        },
    )
    mutation_even_experts: bool = field(
        default=False,
        metadata={
            "help": (
                "Restrict mutations to even deltas and seed plans so per-layer remaining "
                "experts stay divisible by 2."
            )
        },
    )
    init_patterns: list[str] = field(
        default_factory=lambda: list(DEFAULT_INIT_PATTERNS),
        metadata={
            "help": (
                "Initial sparsity distribution patterns to seed the first generation "
                "before filling remaining slots with random."
            )
        },
    )
    search_dataset_name: str | None = field(
        default=None,
        metadata={
            "help": (
                "Dataset name for evolutionary search scoring. "
                "Defaults to observer dataset if None. "
                "Comma-separated list is allowed to mix datasets."
            )
        },
    )
    search_dataset_weights: str | None = field(
        default=None,
        metadata={
            "help": (
                "Optional comma-separated weights aligned with search_dataset_name. "
                "Weights split the total search samples across datasets "
                "(1.0 = base proportion)."
            )
        },
    )
    search_dataset_config_name: str | None = field(
        default=None,
        metadata={
            "help": "Optional dataset config name for the search dataset."
        },
    )
    search_dataset_split: str = field(
        default="train",
        metadata={"help": "Dataset split to use for search scoring."},
    )
    search_samples_per_category: int = field(
        default=64,
        metadata={
            "help": (
                "Total number of samples for search evaluation. "
                "When split_by_category is True, this is divided across categories."
            )
        },
    )
    search_pack_samples: bool = field(
        default=True,
        metadata={
            "help": (
                "Pack multiple samples into one sequence when preparing the search "
                "dataset (mirrors observer behavior)."
            )
        },
    )
    search_microbatch_size: int = field(
        default=1,
        metadata={
            "help": (
                "Microbatch size for scoring during search. "
                "0 = auto-pick the largest that fits, 1 = disable batching."
            )
        },
    )
    search_microbatch_max: int = field(
        default=0,
        metadata={
            "help": (
                "Upper bound for auto microbatch selection. 0 = no explicit cap "
                "(uses a conservative internal limit)."
            )
        },
    )
    search_resume_from: str | None = field(
        default=None,
        metadata={
            "help": (
                "Path to a search checkpoint to resume from. If set, search state "
                "is restored and generations continue from the saved point."
            )
        },
    )
    search_checkpoint_every: int = field(
        default=10,
        metadata={
            "help": (
                "Save a search checkpoint every N generations. 0 disables "
                "checkpointing."
            )
        },
    )
    log_eval_example: bool = field(
        default=False,
        metadata={
            "help": (
                "Log one decoded search sample (calibration) with the reference "
                "answer and the model's response for each individual."
            )
        },
    )
    example_max_new_tokens: int = field(
        default=128,
        metadata={
            "help": "Max new tokens to generate when logging the example response."
        },
    )
    esp_p_max_new_tokens: int = field(
        default=512,
        metadata={
            "help": (
                "Max new tokens the baseline model may generate for esp-p / kl-p "
                "fitness before scoring."
            )
        },
    )
    spec_dec_chunk_size: int = field(
        default=1,
        metadata={
            "help": (
                "Chunk size for speculative decoding fitness. Draft this many tokens "
                "before validating with the baseline model."
            )
        },
    )
    spec_dec_assistant_device: str | None = field(
        default=None,
        metadata={
            "help": (
                "Optional device for the assistant (pruned) model when using spec-dec "
                "(e.g., 'cuda:0')."
            )
        },
    )
    spec_dec_baseline_device: str | None = field(
        default=None,
        metadata={
            "help": (
                "Optional device for the baseline (target) model when using spec-dec "
                "(e.g., 'cuda:1')."
            )
        },
    )
    structural_nonuniform: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, materialize non-uniform pruning structurally for supported models (OLMoE). "
                "Otherwise keep mask-based pruning."
            )
        },
    )
    fitness: str = field(
        default="nll-full",
        metadata={
            "help": "Fitness metric for search scoring.",
            "choices": [
                "nll-full",
                "nll-assistant",
                "kl-baseline",
                "kl-pq-b",
                "kl-p",
                "spec-dec",
                "esp-dataset",
                "sp-dataset",
                "esp-p",
            ],
        },
    )

    def __post_init__(self) -> None:
        if self.init_patterns is None:
            self.init_patterns = list(DEFAULT_INIT_PATTERNS)


@dataclass
class FSDPArgs:
    compute_environment: str = field(
        default="LOCAL_MACHINE", metadata={"help": "Compute environment type."}
    )
    debug: bool = field(default=False, metadata={"help": "Enable debug mode."})
    distributed_type: str = field(
        default="FSDP",
        metadata={
            "help": "Distributed training type.",
            "choices": [
                "NO",
                "MULTI_CPU",
                "MULTI_GPU",
                "MULTI_NPU",
                "FSDP",
                "DEEPSPEED",
            ],
        },
    )
    downcast_bf16: str = field(
        default="no",
        metadata={
            "help": "Whether to downcast bf16 operations.",
            "choices": ["no", "yes"],
        },
    )
    fsdp_auto_wrap_policy: str = field(
        default="TRANSFORMER_BASED_WRAP",
        metadata={
            "help": "FSDP auto wrap policy.",
            "choices": ["NO_WRAP", "SIZE_BASED_WRAP", "TRANSFORMER_BASED_WRAP"],
        },
    )
    fsdp_backward_prefetch_policy: str = field(
        default="BACKWARD_PRE",
        metadata={
            "help": "FSDP backward prefetch policy.",
            "choices": ["NO_PREFETCH", "BACKWARD_PRE", "BACKWARD_POST"],
        },
    )
    fsdp_forward_prefetch: bool = field(
        default=False, metadata={"help": "Enable FSDP forward prefetch."}
    )
    fsdp_cpu_ram_efficient_loading: bool = field(
        default=True, metadata={"help": "Enable CPU RAM efficient loading for FSDP."}
    )
    fsdp_offload_params: bool = field(
        default=False, metadata={"help": "Offload parameters to CPU with FSDP."}
    )
    fsdp_sharding_strategy: str = field(
        default="FULL_SHARD",
        metadata={
            "help": "FSDP sharding strategy.",
            "choices": ["NO_SHARD", "SHARD_GRAD_OP", "FULL_SHARD", "HYBRID_SHARD"],
        },
    )
    fsdp_state_dict_type: str = field(
        default="SHARDED_STATE_DICT",
        metadata={
            "help": "FSDP state dict type.",
            "choices": ["FULL_STATE_DICT", "LOCAL_STATE_DICT", "SHARDED_STATE_DICT"],
        },
    )
    fsdp_sync_module_states: bool = field(
        default=True, metadata={"help": "Synchronize module states in FSDP."}
    )
    fsdp_transformer_layer_cls_to_wrap: str = field(
        default="BertLayer",
        metadata={"help": "Transformer layer class name to wrap with FSDP."},
    )
    fsdp_use_orig_params: bool = field(
        default=True, metadata={"help": "Use original parameters in FSDP."}
    )
    machine_rank: int = field(
        default=0, metadata={"help": "Rank of the machine in multi-machine setup."}
    )
    main_training_function: str = field(
        default="main", metadata={"help": "Name of the main training function."}
    )
    mixed_precision: str = field(
        default="bf16",
        metadata={
            "help": "Mixed precision mode for training.",
            "choices": ["no", "fp16", "bf16"],
        },
    )
    num_machines: int = field(
        default=1, metadata={"help": "Number of machines for distributed training."}
    )
    num_processes: int = field(
        default=2, metadata={"help": "Number of processes for distributed training."}
    )
    rdzv_backend: str = field(
        default="static",
        metadata={"help": "Rendezvous backend for distributed training."},
    )
    same_network: bool = field(
        default=True, metadata={"help": "Whether all machines are on the same network."}
    )
    use_cpu: bool = field(
        default=False, metadata={"help": "Force use of CPU instead of GPU."}
    )
