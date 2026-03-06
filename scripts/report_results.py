import argparse
import sys
from pathlib import Path

from report_evals import process_eval_directory


def generate_report(model_directory_str: str, decimals: int = 2):
    """Generates evaluation results report for a model."""
    model_dir = Path(model_directory_str)
    if not model_dir.is_dir():
        print(
            f"Error: Model directory not found at {model_dir}",
            file=sys.stderr,
        )
        return

    output_full_csv_path = model_dir / "results_full.csv"
    output_summary_csv_path = model_dir / "results_summary.csv"
    output_full_even_csv_path = model_dir / "results_full_even.csv"
    output_summary_even_csv_path = model_dir / "results_summary_even.csv"

    metric_header = [
        "HumanEval (pass@1)",
        "HumanEval+ (pass@1)",
        "MBPP",
        "MBPP+",
        "evalplus",
        "arc_c (acc norm)",
        "arc_e (acc norm)",
        "boolq",
        "hellaswag (acc norm)",
        "mmlu",
        "openbookqa (acc norm)",
        "rte",
        "winogrande",
        "mc_average",
        "livecodebench_pass@1",
        "Wildbench_creative_writing_score_rescaled",
        "gsm8k",
        "MATH-500",
        "AIME-25",
        "math_average",
    ]
    livecode_idx = metric_header.index("livecodebench_pass@1")
    evalplus_idx = metric_header.index("evalplus")
    metric_header_with_code_avg = (
        metric_header[: livecode_idx + 1]
        + ["Code Avg"]
        + metric_header[livecode_idx + 1 :]
    )
    full_header = [
        "Model",
        "calib_dataset",
        "search_dataset",
        "compression_technique",
        "generations",
        "sample_size",
        "compression_method",
        "fitness",
        "recovery_method",
        "compression_ratio",
        "seed",
        "perserve experts",
        *metric_header_with_code_avg,
        "subdir",
    ]

    results_to_print = []

    model_name = model_dir.name

    if decimals < 0:
        raise ValueError("--decimals must be >= 0")

    def _fmt_float(val):
        if val == "N/A" or val is None:
            return "N/A"
        try:
            f = float(val)
        except (TypeError, ValueError):
            return str(val)
        return f"{f:.{decimals}f}"

    def _fmt_ratio(val):
        if val == "N/A" or val is None:
            return "N/A"
        try:
            f = float(val)
        except (TypeError, ValueError):
            return "N/A"
        if f == 0:
            return "0"
        return f"{f:.{decimals}f}"

    def _to_float(val):
        if val == "N/A" or val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def _avg_values(values):
        nums = [v for v in (_to_float(v) for v in values) if v is not None]
        if not nums:
            return "N/A"
        return sum(nums) / len(nums)

    def _split_csv(value):
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return []

    def _normalize_list(value):
        if value is None or value == "N/A":
            return []
        if isinstance(value, (list, tuple)):
            items = []
            for item in value:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    items.append(text)
            return items
        return _split_csv(str(value))

    def _short_dataset_single(name):
        if not name or name == "N/A":
            return "N/A"
        try:
            return str(name).split("/")[-1]
        except Exception:
            return "N/A"

    def _format_weight(weight):
        try:
            f = float(weight)
        except (TypeError, ValueError):
            return str(weight)
        if f.is_integer():
            return str(int(f))
        return f"{f:g}"

    def _format_search_dataset(name, weights=None):
        names = _normalize_list(name)
        if not names:
            return "N/A"
        short_names = []
        for item in names:
            short = _short_dataset_single(item)
            if short != "N/A":
                short_names.append(short)
        if not short_names:
            return "N/A"
        if len(short_names) == 1:
            return short_names[0]
        weight_list = _normalize_list(weights)
        if weight_list and len(weight_list) == len(short_names):
            pairs = [
                f"{dataset}({_format_weight(weight)})"
                for dataset, weight in zip(short_names, weight_list)
            ]
            return "+".join(pairs)
        return "+".join(short_names)

    def _with_code_avg(metric_values):
        code_avg_value = _fmt_float(
            _avg_values([metric_values[evalplus_idx], metric_values[livecode_idx]])
        )
        return (
            metric_values[: livecode_idx + 1]
            + [code_avg_value]
            + metric_values[livecode_idx + 1 :]
        )

    for calib_dataset_dir in model_dir.iterdir():
        if not calib_dataset_dir.is_dir():
            continue
        calib_dataset = calib_dataset_dir.name

        # --- Full model results (explicit results_dir like .../full_olmoe) ---
        for full_dir in calib_dataset_dir.iterdir():
            if not full_dir.is_dir() or not full_dir.name.startswith("full_"):
                continue
            try:
                result_row = process_eval_directory(full_dir, calib_dataset_dir)
                if result_row:
                    (
                        output_name,
                        compression_method_full,
                        fitness,
                        compression_ratio,
                        seed,
                        _,  # old perserve_super_experts, now unused
                        *metrics,
                    ) = result_row

                    compression_method = compression_method_full.split("-")[0]
                    subdir_path = Path(model_name) / calib_dataset / output_name
                    metric_values = _with_code_avg([_fmt_float(m) for m in metrics])
                    full_row = [
                        model_name,
                        calib_dataset,
                        "N/A",
                        "full",
                        "N/A",
                        "N/A",
                        compression_method,
                        fitness,
                        "N/A",  # recovery_method
                        _fmt_ratio(compression_ratio),
                        seed,
                        "N/A",  # perserve experts
                        *metric_values,
                        str(subdir_path),
                    ]
                    results_to_print.append(full_row)
            except Exception as e:
                print(
                    f"Warning: Error processing directory {full_dir}: {e}",
                    file=sys.stderr,
                )

        for tech_dir in calib_dataset_dir.iterdir():
            if not tech_dir.is_dir() or tech_dir.name not in [
                "pruned_models",
                "merged_models",
                "non_uniform_merged_models",
                "pruned_models_searched",
            ]:
                continue

            compression_technique = (
                "pruning" if "pruned" in tech_dir.name else "merging"
            )
            if tech_dir.name == "pruned_models_searched":
                compression_technique = "searched"

            for eval_parent_dir in tech_dir.iterdir():
                if not eval_parent_dir.is_dir():
                    continue

                eval_dir = None
                candidate_names = ["eval", "eval_hf", "eval_vllm"]
                if compression_technique == "merging":
                    # Merged models have an extra directory layer, e.g., m_smoe-0.50/m_smoe/eval*
                    for subdir in eval_parent_dir.iterdir():
                        if not subdir.is_dir():
                            continue
                        for candidate in candidate_names:
                            if (subdir / candidate).is_dir():
                                eval_dir = subdir / candidate
                                break
                        if eval_dir:
                            break
                else:
                    # Pruned models have eval directly inside, e.g., l1_unstructured-0.50/eval*
                    for candidate in candidate_names:
                        if (eval_parent_dir / candidate).is_dir():
                            eval_dir = eval_parent_dir / candidate
                            break

                if not eval_dir or not eval_dir.is_dir():
                    continue

                try:
                    result_row = process_eval_directory(eval_dir, tech_dir)
                    if result_row:
                        (
                            output_name,
                            compression_method_full,
                            fitness,
                            compression_ratio,
                            seed,
                            _,  # old perserve_super_experts, now unused
                            *metrics,
                        ) = result_row

                        compression_method = compression_method_full.split("-")[0]
                        search_dataset = "N/A"
                        generations = "N/A"
                        sample_size = "N/A"
                        if tech_dir.name == "pruned_models_searched":
                            meta_path = eval_parent_dir / "search_metadata.json"
                            if meta_path.exists():
                                try:
                                    import json

                                    meta = json.loads(meta_path.read_text())
                                    fitness = (
                                        meta.get("search_params", {}).get("fitness")
                                        or meta.get("fitness")
                                        or fitness
                                    )
                                    compression_ratio = meta.get("prune_ratio", compression_ratio)
                                    seed = meta.get("seed", seed)
                                    generations = (
                                        meta.get("search_params", {}).get("generations")
                                        or meta.get("search_params", {}).get("generation")
                                        or "N/A"
                                    )
                                    sample_size = (
                                        meta.get("search_params", {}).get("search_samples_per_category")
                                        or meta.get("search_samples_per_category")
                                        or "N/A"
                                    )
                                    search_params = meta.get("search_params") or {}
                                    search_dataset = _format_search_dataset(
                                        search_params.get("search_dataset_name")
                                        or meta.get("search_dataset_name"),
                                        search_params.get("search_dataset_weights")
                                        or meta.get("search_dataset_weights"),
                                    )
                                except Exception:
                                    pass

                        perserve_experts = "N/A"
                        if (
                            "perserve_outlier" in output_name
                            or "outlier_expert_singletons" in output_name
                        ):
                            perserve_experts = "outlier"
                        elif (
                            "perserve_super" in output_name
                            or "super_expert_singletons" in output_name
                        ):
                            perserve_experts = "super"

                        subdir_path = (
                            Path(model_name)
                            / calib_dataset
                            / tech_dir.name
                            / output_name
                        )

                        metric_values = _with_code_avg([_fmt_float(m) for m in metrics])
                        full_row = [
                            model_name,
                            calib_dataset,
                            search_dataset,
                            compression_technique,
                            generations,
                            sample_size,
                            compression_method,
                            fitness,
                            "N/A",  # recovery_method
                            _fmt_ratio(compression_ratio),
                            seed,
                            perserve_experts,
                            *metric_values,
                            str(subdir_path),
                        ]
                        results_to_print.append(full_row)
                except Exception as e:
                    print(
                        f"Warning: Error processing directory {eval_dir}: {e}",
                        file=sys.stderr,
                    )

    def _sort_key(row):
        # Column indices for readability
        idx = {name: i for i, name in enumerate(full_header)}
        technique = row[idx["compression_technique"]]
        search_dataset = row[idx["search_dataset"]]
        fitness = row[idx["fitness"]]
        compression_method = row[idx["compression_method"]]
        ratio = row[idx["compression_ratio"]]

        technique_order = {"full": 0, "searched": 1, "pruning": 2, "merging": 3}
        t_ord = technique_order.get(str(technique), 99)
        is_full = 0 if str(technique) == "full" else 1

        def _na_last(val):
            s = str(val)
            return (1, "") if s == "N/A" else (0, s)

        ds_key = _na_last(search_dataset)
        fit_key = _na_last(fitness)

        # Prefer smaller compression ratio first within equal groups (best-effort).
        try:
            ratio_key = float(ratio)
        except Exception:
            ratio_key = 1e9

        return (is_full, ds_key, fit_key, t_ord, str(compression_method), ratio_key, row[-1])

    results_to_print.sort(key=_sort_key)

    summary_id_columns = [
        "Model",
        "calib_dataset",
        "search_dataset",
        "compression_technique",
        "compression_method",
        "recovery_method",
        "seed",
        "perserve experts",
        "generations",
        "sample_size",
        "fitness",
        "compression_ratio",
    ]
    summary_header = summary_id_columns + [
        "Eval+",
        "LiveCode",
        "Code Avg",
        "WildBench",
        "GSM8K",
        "MATH-500",
        "Math Avg",
        "MC Avg",
        "subdir",
    ]
    full_idx = {name: i for i, name in enumerate(full_header)}
    tech_dirs = {
        "pruned_models",
        "merged_models",
        "non_uniform_merged_models",
        "pruned_models_searched",
    }

    def _extract_run_dir_name(row):
        subdir = row[full_idx["subdir"]]
        try:
            parts = Path(subdir).parts
        except Exception:
            return ""
        for idx, part in enumerate(parts):
            if part in tech_dirs and idx + 1 < len(parts):
                return parts[idx + 1]
        return ""

    results_even = []
    for row in results_to_print:
        technique = row[full_idx["compression_technique"]]
        if technique in {"full", "merging", "pruning"}:
            results_even.append(row)
            continue
        if technique == "searched":
            run_dir = _extract_run_dir_name(row)
            if run_dir and "even" in run_dir.lower():
                results_even.append(row)

    def _summary_row(row):
        evalplus_value = row[full_idx["evalplus"]]
        livecode_value = row[full_idx["livecodebench_pass@1"]]
        code_avg_value = _fmt_float(_avg_values([evalplus_value, livecode_value]))
        wildbench_value = row[full_idx["Wildbench_creative_writing_score_rescaled"]]
        gsm8k_value = row[full_idx["gsm8k"]]
        math500_value = row[full_idx["MATH-500"]]
        math_avg_value = _fmt_float(_avg_values([gsm8k_value, math500_value]))
        mc_avg_value = row[full_idx["mc_average"]]
        subdir_value = row[full_idx["subdir"]]
        return [row[full_idx[name]] for name in summary_id_columns] + [
            evalplus_value,
            livecode_value,
            code_avg_value,
            wildbench_value,
            gsm8k_value,
            math500_value,
            math_avg_value,
            mc_avg_value,
            subdir_value,
        ]

    with open(output_full_csv_path, "w", newline="") as f:
        f.write(",".join(full_header) + "\n")
        for result in results_to_print:
            f.write(",".join(map(str, result)) + "\n")

    with open(output_full_even_csv_path, "w", newline="") as f:
        f.write(",".join(full_header) + "\n")
        for result in results_even:
            f.write(",".join(map(str, result)) + "\n")

    with open(output_summary_csv_path, "w", newline="") as f:
        f.write(",".join(summary_header) + "\n")
        for result in results_to_print:
            f.write(",".join(map(str, _summary_row(result))) + "\n")

    with open(output_summary_even_csv_path, "w", newline="") as f:
        f.write(",".join(summary_header) + "\n")
        for result in results_even:
            f.write(",".join(map(str, _summary_row(result))) + "\n")

    print(f"Results full saved to {output_full_csv_path}")
    print(f"Results summary saved to {output_summary_csv_path}")
    print(f"Results full even saved to {output_full_even_csv_path}")
    print(f"Results summary even saved to {output_summary_even_csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation results report for a model."
    )
    parser.add_argument(
        "model_directory",
        type=str,
        help="The root directory for a model (e.g., 'artifacts/Qwen3-30B-A3B').",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Number of decimal places to format numeric values in the CSV (default: 3).",
    )
    args = parser.parse_args()
    generate_report(args.model_directory, decimals=args.decimals)


if __name__ == "__main__":
    main()
