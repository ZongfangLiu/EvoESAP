import argparse
import json
import os
import re
import sys
from pathlib import Path


def custom_sort_key(item):
    """
    Custom sort key to sort by name prefix alphabetically and suffix numerically descending.
    Example: 'name-0.50' comes before 'name-0.25'.
    """
    try:
        # Splits the name by the last hyphen to handle names with multiple hyphens
        name, value = item[0].rsplit("-", 1)
        return name, -float(value)
    except (ValueError, IndexError):
        # Fallback for names that don't match the expected 'name-value' pattern
        return item[0], 0


def get_pass_at_k(results_data):
    """Safely extracts pass@1 metrics from results data."""
    pass_at_k = results_data.get("pass_at_k", {})
    base_pass = pass_at_k.get("base", {}).get("pass@1", "N/A")
    plus_pass = pass_at_k.get("plus", {}).get("pass@1", "N/A")
    return base_pass, plus_pass


def calculate_average(values):
    """Calculates the average of a list of values, ignoring 'N/A'."""
    numeric_values = [float(v) for v in values if v != "N/A" and v is not None]
    if not numeric_values:
        return "N/A"
    return sum(numeric_values) / len(numeric_values)


def process_eval_directory(eval_dir, parent_path):
    """
    Parses evaluation files from a single evaluation results directory.
    """
    # --- Determine output name ---
    eval_dir = Path(eval_dir)
    parent_path = Path(parent_path)
    try:
        relative_path = eval_dir.relative_to(parent_path)
    except ValueError:
        # Fall back to best-effort naming if parent_path isn't an ancestor.
        relative_path = eval_dir

    # Typical layouts:
    # - merged: .../<merge-run>/<method>/eval_{hf|vllm}
    # - pruned: .../<prune-run>/eval_{hf|vllm}
    # - full:  .../full_<name>/ (results files live at the root)
    output_name = "/".join(relative_path.parts[:-1]) if relative_path.parts else "N/A"
    if not output_name:
        output_name = relative_path.parts[0] if relative_path.parts else eval_dir.name

    # --- Skip specified directories ---
    skip_dirs = ["sft", "sft-lora-r_8", "sft-lora-r_16"]
    if any(part in skip_dirs for part in relative_path.parts):
        return None

    # --- Parse output_name for compression_ratio, seed and perserve_super_experts ---
    compression_ratio = "N/A"
    compression_method = "N/A"
    fitness = "N/A"

    # Check if this is a merged model path structure
    is_merged_model = len(relative_path.parts) > 2 and relative_path.parts[-1] in ("eval", "eval_hf", "eval_vllm")

    if is_merged_model:
        # For merged: .../m_smoe-0.50/m_smoe/eval or .../hc_smoe-seed_11_0.50/hc_smoe/eval
        # compression_ratio from parent dir name
        # compression_method from method dir name
        parent_dir_name = relative_path.parts[-3]
        method_dir_name = relative_path.parts[-2]
        
        # Use regex to find the compression ratio, which is the last float number in the dir name
        match = re.search(r"(\d+\.\d+)$", parent_dir_name)
        if match:
            try:
                compression_ratio = float(match.group(1))
            except (ValueError, IndexError):
                pass # keep default if parsing fails

        compression_method = method_dir_name
    else:
        # For full: .../full_<name>/
        if output_name.startswith("full_") or eval_dir.name.startswith("full_"):
            # Full model: no compression.
            compression_ratio = 0.0
            compression_method = "full"
        # For searched pruning: .../pruned_models_searched/fit_<fitness>-<prune_method>-seed_<seed>-ratio_<ratio>-.../eval*
        elif output_name.startswith("fit_") and "ratio_" in output_name:
            match = re.search(r"(?:^|-)ratio_(\d+(?:\.\d+)?)", output_name)
            if match:
                try:
                    compression_ratio = float(match.group(1))
                except (ValueError, IndexError):
                    pass
            # Parse prune method from: fit_<fitness>-<prune_method>-seed_<seed>...
            pre_seed = output_name.split("-seed_", 1)[0]
            if pre_seed.startswith("fit_"):
                tmp = pre_seed[len("fit_") :]
                if "-" in tmp:
                    fit, prune_method = tmp.rsplit("-", 1)
                    compression_method = prune_method
                    fitness = fit
        else:
            # For pruned: .../frequency-0.50/eval
            try:
                # Splits the name by the last hyphen to handle names with multiple hyphens
                method, value_part = output_name.rsplit("-", 1)
                compression_ratio = float(value_part)
                compression_method = method
            except (ValueError, IndexError):
                pass  # keep default if parsing fails

    seed = 42  # default
    perserve_super_experts = False  # default
    
    # For merged models, seed is in parent_dir_name, otherwise in output_name
    search_string_for_seed = parent_dir_name if is_merged_model else output_name
    parts = search_string_for_seed.replace("_", "-").split("-")

    for i, part in enumerate(parts):
        if part == "seed" and i + 1 < len(parts):
            try:
                seed = int(parts[i+1])
            except (ValueError, IndexError):
                pass  # keep default if parsing fails
        elif part == "perserve" and i + 1 < len(parts) and parts[i+1] == "super":
            perserve_super_experts = True

    # --- File paths ---
    humaneval_path = eval_dir / "humaneval.json"
    mbpp_path = eval_dir / "mbpp.json"
    lm_eval_path = eval_dir / "lm_eval_results.json"
    livecodebench_path = eval_dir / "Scenario.codegeneration_1_0.2_eval.json"

    # check degenerate outputs

    # --- Process humaneval.json and mbpp.json ---
    code_eval_values = []
    for file_path in [humaneval_path, mbpp_path]:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                base, plus = get_pass_at_k(data)
                code_eval_values.extend([base, plus])
        except (FileNotFoundError, json.JSONDecodeError):
            if (
                file_path == humaneval_path
                and (
                    eval_dir
                    / "evalplus_results"
                    / "humaneval"
                    / "degenerate_outputs.txt"
                ).exists()
            ) or (
                file_path == mbpp_path
                and (
                    eval_dir / "evalplus_results" / "mbpp" / "degenerate_outputs.txt"
                ).exists()
            ):
                code_eval_values.extend([0, 0])
            else:
                code_eval_values.extend(["N/A", "N/A"])

    coding_avg = calculate_average(code_eval_values)

    # --- Process lm_eval_results.json ---
    lm_eval_values = []
    try:
        with open(lm_eval_path, "r") as f:
            lm_data = json.load(f)
        results = lm_data.get("results", {})
        metrics_to_extract = [
            ("arc_challenge", "acc_norm,none"),
            ("arc_easy", "acc_norm,none"),
            ("boolq", "acc,none"),
            ("hellaswag", "acc_norm,none"),
            ("mmlu", "acc,none"),
            ("openbookqa", "acc_norm,none"),
            ("rte", "acc,none"),
            ("winogrande", "acc,none"),
        ]
        for task, metric in metrics_to_extract:
            try:
                value = results[task][metric]
                lm_eval_values.append(value)
            except KeyError:
                lm_eval_values.append("N/A")
    except (FileNotFoundError, json.JSONDecodeError):
        lm_eval_values.extend(["N/A"] * 8)

    mc_avg = calculate_average(lm_eval_values)

    # --- Process livecodebench ---
    def _find_livecodebench_eval_file(eval_dir: Path) -> Path | None:
        candidates = list(eval_dir.glob("Scenario.codegeneration_*_eval.json"))
        if not candidates:
            return None

        # Prefer the common n=1 file(s), then greedy temp=0.0, else newest by mtime.
        n1 = [p for p in candidates if re.match(r"^Scenario\\.codegeneration_1_", p.name)]
        for p in n1:
            if re.match(r"^Scenario\\.codegeneration_1_0\\.0(?:_cot)?_eval\\.json$", p.name):
                return p
        chosen_pool = n1 if n1 else candidates
        return max(chosen_pool, key=lambda p: p.stat().st_mtime)

    livecodebench_pass_at_1 = "N/A"
    try:
        livecodebench_path = _find_livecodebench_eval_file(eval_dir)
        if livecodebench_path is None:
            raise FileNotFoundError("No LiveCodeBench eval file found")
        with open(livecodebench_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            metrics = data[0]
            livecodebench_pass_at_1 = metrics.get("pass@1", "N/A")
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass  # livecodebench_pass_at_1 is already "N/A"

    # --- Process wildbench ---
    wildbench_score = "N/A"
    wildbench_dir = eval_dir / "wildbench" / "runs" / "test"
    try:
        if wildbench_dir.is_dir():
            # The subdir name is variable, so we search for stats.json
            stats_files = list(wildbench_dir.glob("**/stats.json"))
            if stats_files:
                with open(stats_files[0], "r") as f:
                    data = json.load(f)
                for item in data:
                    if (
                        isinstance(item, dict)
                        and item.get("name", {}).get("name")
                        == "wildbench_score_rescaled"
                        and "perturbation" not in item.get("name", {})
                    ):
                        wildbench_score = item.get("sum", "N/A")
                        break
    except (FileNotFoundError, json.JSONDecodeError, IndexError):
        pass  # wildbench_score is already "N/A"

    # --- Process math results ---
    math_scores = []
    math_benchmarks = ["gsm8k", "math_500", "aime25"]
    evalscope_glob_path = eval_dir / "evalscope_results" / "*" / "reports" / "*"
    
    found_reports_path = None
    # Use glob to find the reports directory, assuming one per eval_dir
    possible_paths = list(eval_dir.glob("evalscope_results/*/reports/*"))
    if possible_paths:
        # Let's assume the first one found is the correct one.
        # This might need refinement if multiple report directories exist.
        found_reports_path = possible_paths[0]

    if found_reports_path and found_reports_path.is_dir():
        for benchmark in math_benchmarks:
            math_file_path = found_reports_path / f"{benchmark}.json"
            try:
                with open(math_file_path, "r") as f:
                    data = json.load(f)
                score = data.get("score", "N/A")
                math_scores.append(score)
            except (FileNotFoundError, json.JSONDecodeError):
                math_scores.append("N/A")
    else:
        math_scores.extend(["N/A"] * len(math_benchmarks))

    math_avg = calculate_average(math_scores)

    # --- Combine all data ---
    all_values = (
        [
            output_name,
            compression_method,
            fitness,
            compression_ratio if isinstance(compression_ratio, float) else "N/A",
            seed,
            perserve_super_experts,
        ]
        + code_eval_values
        + [coding_avg]
        + lm_eval_values
        + [mc_avg]
        + [livecodebench_pass_at_1]
        + [wildbench_score]
        + math_scores
        + [math_avg]
    )
    return all_values


def find_and_process_evals(parent_dir):
    """
    Finds and processes all 'eval' directories recursively.
    """
    parent_path = Path(parent_dir)
    if not parent_path.is_dir():
        print(f"Error: Directory not found at {parent_dir}", file=sys.stderr)
        return

    results = []
    for eval_dir in parent_path.rglob("**/"):
        if eval_dir.is_dir() and eval_dir.name in ("eval", "eval_hf", "eval_vllm"):
            try:
                result_row = process_eval_directory(eval_dir, parent_path)
                if result_row:
                    results.append(result_row)
            except Exception as e:
                print(
                    f"Warning: Error processing directory {eval_dir}: {e}",
                    file=sys.stderr,
                )

    # Sort results
    results.sort(key=custom_sort_key)

    # Print CSV header
    header = [
        "sub_dir",
        "compression_method",
        "fitness",
        "compression_ratio",
        "seed",
        "perserve_super_experts",
        "humaneval_base",
        "humaneval_plus",
        "mbpp_base",
        "mbpp_plus",
        "coding_avg",
        "arc_challenge",
        "arc_easy",
        "boolq",
        "hellaswag",
        "mmlu",
        "openbookqa",
        "rte",
        "winogrande",
        "mc_avg",
        "livecodebench_pass@1",
        "wildbench_score",
        "gsm8k",
        "math_500",
        "aime25",
        "math_avg",
    ]
    print(",".join(header))

    # Print sorted results
    for result in results:
        print(",".join(map(str, result)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recursively find and report metrics from evaluation directories."
    )
    parser.add_argument(
        "parent_directory",
        help="The parent directory to start the recursive search from.",
    )
    args = parser.parse_args()

    find_and_process_evals(args.parent_directory)
