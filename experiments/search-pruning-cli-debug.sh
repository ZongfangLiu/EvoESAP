#!/bin/bash
set -e
set -o pipefail

export CUDA_VISIBLE_DEVICES=${1}
FIRST_DEVICE=$(echo "$1" | cut -d',' -f1)
port=$((8200 + FIRST_DEVICE))
debug_port=${DEBUG_PORT:-$((5678 + FIRST_DEVICE))}
model_name=${2:-"Qwen/Qwen3-30B-A3B"}
pruning_method=${3:-"reap"}
seed=${4:-42}
int_sparsity=${5:-1}
dataset_name=${6:-"theblackcat102/evol-codealpaca-v1"}
search_dataset_name=${7:-"$dataset_name"}
search_dataset_weights="${SEARCH_DATASET_WEIGHTS:-""}"
spec_dec_assistant_device="${SPEC_DEC_ASSISTANT_DEVICE:-""}"
spec_dec_baseline_device="${SPEC_DEC_BASELINE_DEVICE:-""}"
population_size=${8:-8}
generations=${9:-4}
topk=${10:-4}
mutation_max_delta=${11:-2}
mutation_times=${12:-1}
mutation_max_attempts=${13:-32}
search_samples_per_category=${14:-128}
fitness=${15:-"nll-full"}

# qa
run_lm_eval=${16:-true}
# coding
run_evalplus=${17:-true}
run_livecodebench=${18:-true}
# math
run_math=${19:-false}
# wildbench
run_wildbench=${20:-false}
singleton_super_experts=${21:-"false"}
singleton_outlier_experts=${22:-"false"}
lm_eval_apply_chat_template=${23:-false}
lm_eval_num_fewshot=${24:-0}
lm_eval_fewshot_as_multiturn=${25:-false}
use_server=${26:-true}
structural_nonuniform=${27:-true}
# control flow toggles
run_search=${28:-true}
run_final_eval=${29:-true}
# optional override for pruned dir (pos 30 or trailing arg)
pruned_dir_override=${30:-""}
# number of experts (used for deriving ratio in dir names)
num_experts=${31:-64}
# resume/checkpoint controls
# search_resume_from is a generation number; 0 = start from scratch
search_resume_from=${32:-0}
search_checkpoint_every=${33:-10}
init_patterns="${INIT_PATTERNS:-}"
mutation_even_experts="${MUTATION_EVEN_EXPERTS:-false}"

if (( $# > 33 )); then
    for arg in "${@:34}"; do
        if [[ "${arg}" == init_patterns=* || "${arg}" == --init-patterns=* ]]; then
            init_patterns="${arg#*=}"
        elif [[ "${arg}" == mutation_even_experts=* || "${arg}" == --mutation-even-experts=* ]]; then
            mutation_even_experts="${arg#*=}"
        elif [[ "${arg}" == search_dataset_weights=* || "${arg}" == --search-dataset-weights=* ]]; then
            search_dataset_weights="${arg#*=}"
        elif [[ "${arg}" == spec_dec_assistant_device=* || "${arg}" == --spec-dec-assistant-device=* ]]; then
            spec_dec_assistant_device="${arg#*=}"
        elif [[ "${arg}" == spec_dec_baseline_device=* || "${arg}" == --spec-dec-baseline-device=* ]]; then
            spec_dec_baseline_device="${arg#*=}"
        elif [[ -z "${pruned_dir_override}" ]]; then
            pruned_dir_override="${arg}"
        fi
    done
fi

pruned_dir_override_norm="${pruned_dir_override,,}"
if [[ "${pruned_dir_override_norm}" == "none" || "${pruned_dir_override_norm}" == "null" ]]; then
    pruned_dir_override=""
fi
init_patterns_norm="${init_patterns,,}"
if [[ "${init_patterns_norm}" == "none" || "${init_patterns_norm}" == "null" ]]; then
    init_patterns=""
fi

# derive ratio for naming; fail fast on zero experts
ratio=$(python - <<PY
int_sparsity = float("${int_sparsity}")
num_experts = float("${num_experts}")
if num_experts == 0:
    raise SystemExit("number of experts must be non-zero")
print(f"{int_sparsity / num_experts:.3f}")
PY
) || exit 1

# derive mask application: use masks in non-structural mode, skip for structural outputs
if [[ "${structural_nonuniform}" == "true" ]]; then
    apply_pruned_masks=false
else
    apply_pruned_masks=true
fi

search_bits="int${int_sparsity}-pop${population_size}-gen${generations}-top${topk}-mut${mutation_max_delta}x${mutation_times}-att${mutation_max_attempts}-samples${search_samples_per_category}"
if [[ "${mutation_even_experts}" == "true" ]]; then
    search_bits="${search_bits}-even"
fi
search_bits_pattern="${search_bits/-gen${generations}/-gen*}"
calib_clean=""
if [[ "${search_dataset_name}" == *","* ]]; then
    IFS=',' read -r -a calib_datasets <<< "${search_dataset_name}"
    weights_list=()
    if [[ -n "${search_dataset_weights}" ]]; then
        IFS=',' read -r -a weights_list <<< "${search_dataset_weights}"
        if [[ ${#weights_list[@]} -ne ${#calib_datasets[@]} ]]; then
            echo "Error: search_dataset_weights count does not match search_dataset_name list" >&2
            exit 1
        fi
    fi
    calib_parts=()
    for i in "${!calib_datasets[@]}"; do
        ds="${calib_datasets[$i]}"
        ds="${ds//[[:space:]]/}"
        ds_base=$(echo "${ds}" | awk -F/ '{print $NF}')
        ds_base_lower=$(echo "${ds_base}" | tr 'A-Z' 'a-z')
        if [[ "${ds_base_lower}" == "c4" ]]; then
            short="c4"
        elif [[ "${ds_base_lower}" == tulu-3* ]]; then
            short="tulu"
        elif [[ "${ds_base_lower}" == *evol-codealpaca* ]]; then
            short="evol"
        else
            short="${ds_base}"
        fi
        weight="1"
        if [[ -n "${search_dataset_weights}" ]]; then
            weight="${weights_list[$i]}"
            weight="${weight//[[:space:]]/}"
            if [[ -z "${weight}" ]]; then
                weight="1"
            fi
        fi
        calib_parts+=("${short}" "${weight}")
    done
    calib_clean=$(IFS=_; echo "${calib_parts[*]}")
else
    calib_base=$(echo "${search_dataset_name}" | awk -F/ '{print $NF}')
    if [[ "${calib_base}" == "tulu-3-sft-personas-math" ]]; then
        calib_base="tulu-3"
    fi
    calib_clean="${calib_base}"
fi
calib_clean=$(echo "${calib_clean}" | sed 's/[^A-Za-z0-9._-]/_/g')
init_patterns_clean="${init_patterns//,/ }"
init_patterns_args=()
uniform_only=false
if [[ -n "${init_patterns_clean}" ]]; then
    read -r -a init_patterns_list <<< "${init_patterns_clean}"
    if [[ ${#init_patterns_list[@]} -gt 0 ]]; then
        init_patterns_args=(--init-patterns "${init_patterns_list[@]}")
        uniform_only=true
        for pattern in "${init_patterns_list[@]}"; do
            if [[ "${pattern}" != "uniform" ]]; then
                uniform_only=false
                break
            fi
        done
    fi
fi
uni_suffix=$([[ "${uniform_only}" == "true" ]] && echo "-uni" || echo "")
exact_dir_name="fit_${fitness}-${pruning_method}-seed_${seed}-ratio_${ratio}${search_bits:+-${search_bits}}-calib_${calib_clean}${uni_suffix}"
short_model_name=$(echo $model_name | cut -d'/' -f2)
short_dataset_name=$(echo $dataset_name | cut -d'/' -f2)
results_root="artifacts/${short_model_name}/${short_dataset_name}"
searched_dir="${results_root}/pruned_models_searched"
pruned_out_dir="${searched_dir}/${exact_dir_name}"

num_samples=1024
output_file_name="observations_${num_samples}_cosine-seed_${seed}.pt"

mkdir -p "${pruned_out_dir}"
search_log_file_name="${pruned_out_dir}/search-pruning-cli-${FIRST_DEVICE}.log"
eval_log_file_name="${pruned_out_dir}/eval-cli-${FIRST_DEVICE}.log"

echo "Running non-uniform pruning with model: $model_name on devices: $CUDA_VISIBLE_DEVICES"
echo "Integer sparsity per layer: $int_sparsity (global budget = int_sparsity * num_layers)"
echo "Number of experts: $num_experts (ratio: ${ratio})"
echo "Search dataset: $search_dataset_name (total samples: $search_samples_per_category)"
if [[ -n "${search_dataset_weights}" ]]; then
    echo "Search dataset weights: $search_dataset_weights"
fi
echo "Population: $population_size, generations: $generations, topk: $topk, mutation max delta: $mutation_max_delta, mutation times: $mutation_times, mutation max attempts: $mutation_max_attempts"
echo "Fitness metric: ${fitness}"
echo "Search log: ${search_log_file_name}"
echo "Eval log: ${eval_log_file_name}"
echo "Debugpy listening on 0.0.0.0:${debug_port} (wait-for-client)"

if [[ "${run_search}" == "true" ]]; then
    extra_search_args=()
    if [[ "${search_resume_from}" != "0" ]]; then
        if ! [[ "${search_resume_from}" =~ ^[0-9]+$ ]]; then
            echo "Error: search_resume_from must be a non-negative integer (0 = from scratch)." >&2
            exit 1
        fi
    fi
    if [[ -n "${search_dataset_weights}" ]]; then
        extra_search_args+=(--search-dataset-weights "${search_dataset_weights}")
    fi
    if [[ -n "${spec_dec_assistant_device}" ]]; then
        extra_search_args+=(--spec-dec-assistant-device "${spec_dec_assistant_device}")
    fi
    if [[ -n "${spec_dec_baseline_device}" ]]; then
        extra_search_args+=(--spec-dec-baseline-device "${spec_dec_baseline_device}")
    fi
    if (( search_resume_from > 0 )); then
        resume_target=${search_resume_from}
        resume_dir_pattern="fit_${fitness}-${pruning_method}-seed_${seed}-ratio_${ratio}${search_bits_pattern:+-${search_bits_pattern}}-calib_${calib_clean}${uni_suffix}"
        shopt -s nullglob
        run_dirs=( ${searched_dir}/${resume_dir_pattern} )
        shopt -u nullglob
        candidates=()
        for run_dir in "${run_dirs[@]}"; do
            ckpt_dir="${run_dir}/search_checkpoints"
            [[ -d "${ckpt_dir}" ]] || continue
            shopt -s nullglob
            for ckpt in "${ckpt_dir}/search_state_gen_"*.pt; do
                [[ -f "${ckpt}" ]] || continue
                gen="${ckpt##*search_state_gen_}"
                gen="${gen%.pt}"
                gen_num=$((10#${gen}))
                if (( gen_num <= resume_target )); then
                    candidates+=("${gen_num}:${ckpt}")
                fi
            done
            shopt -u nullglob
        done
        if [[ ${#candidates[@]} -eq 0 ]]; then
            echo "Error: no checkpoint <= ${resume_target} found for pattern '${resume_dir_pattern}' in ${searched_dir}" >&2
            exit 1
        fi
        resume_choice=$(printf '%s\n' "${candidates[@]}" | sort -t: -k1,1n | tail -n 1)
        resume_gen="${resume_choice%%:*}"
        resume_checkpoint="${resume_choice#*:}"
        echo "Resuming from checkpoint ${resume_checkpoint} (gen=${resume_gen}, requested=${resume_target})"
        extra_search_args+=(--search-resume-from "${resume_checkpoint}")
    fi
    python -m debugpy --listen 0.0.0.0:${debug_port} --wait-for-client src/reap/non_uni_prune.py \
        --model-name $model_name \
        --dataset-name $dataset_name \
        --prune-method $pruning_method \
        --profile false \
        --vllm_port $port \
        --server-log-file-name $search_log_file_name \
        --do-eval false \
        --distance_measure cosine \
        --seed $seed \
        --output_file_name ${output_file_name} \
        --samples_per_category ${num_samples} \
        --record_pruning_metrics_only false \
        --singleton_super_experts ${singleton_super_experts} \
        --singleton_outlier_experts ${singleton_outlier_experts} \
        --int-sparsity ${int_sparsity} \
        --search-dataset-name ${search_dataset_name} \
        --search-samples-per-category ${search_samples_per_category} \
        --population-size ${population_size} \
        --generations ${generations} \
        --topk ${topk} \
        --mutation-max-delta ${mutation_max_delta} \
        --mutation-times ${mutation_times} \
        --mutation-max-attempts ${mutation_max_attempts} \
        --mutation-even-experts ${mutation_even_experts} \
        --structural-nonuniform ${structural_nonuniform} \
        --fitness ${fitness} \
        --output-dir "${pruned_out_dir}" \
        --search-checkpoint-every ${search_checkpoint_every} \
        "${extra_search_args[@]}" \
        "${init_patterns_args[@]}" | tee "${search_log_file_name}"
fi

# choose pruned model directory by explicit name only (no mtime fallback)

# Auto-detect pruned dir from parameter-derived pattern (ratio wildcard)
if [[ -n "${pruned_dir_override}" ]]; then
    model_dir="${pruned_dir_override}"
    if [[ ! -d "${model_dir}" ]]; then
        echo "Error: pruned_dir_override '${model_dir}' does not exist" >&2
        exit 1
    fi
else
    model_dir="${pruned_out_dir}"
    if [[ ! -d "${model_dir}" ]]; then
        echo "Error: expected pruned model directory '${model_dir}' not found. Set pruned_dir_override to override." >&2
        exit 1
    fi
fi

if [[ "${run_final_eval}" != "true" ]]; then
    echo "Skipping evaluation. Model directory: ${model_dir}"
    exit 0
fi

echo "Evaluating model: ${model_dir}"
bash experiments/eval.sh \
    $model_dir \
    $seed \
    $port \
    $eval_log_file_name \
    ${run_lm_eval} \
    ${run_evalplus} \
    ${run_livecodebench} \
    ${run_math} \
    ${run_wildbench} \
    ${lm_eval_apply_chat_template} \
    ${lm_eval_num_fewshot} \
    ${lm_eval_fewshot_as_multiturn} \
    "" \
    ${use_server} \
    ${apply_pruned_masks}
echo "Finished evaluating model: ${model_dir}"
