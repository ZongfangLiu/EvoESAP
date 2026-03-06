#!/bin/bash
set -e
set -o pipefail

gpu_arg="${1:-}"
if [[ -n "${gpu_arg}" && "${gpu_arg}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    gpus="${gpu_arg}"
    shift
else
    gpus="${CUDA_VISIBLE_DEVICES:-0}"
fi
gpus="$(echo "${gpus}" | tr -d '[:space:]')"
export CUDA_VISIBLE_DEVICES="${gpus}"
FIRST_DEVICE="${gpus%%,*}"
if ! [[ "${FIRST_DEVICE}" =~ ^[0-9]+$ ]]; then
    FIRST_DEVICE=0
fi
port=$((8300 + FIRST_DEVICE))

args=("$@")
model_name=${args[0]:-"allenai/OLMoE-1B-7B-0125-Instruct"}
metric_arg=${args[1]:-"frequency"}

allocation_metric=""
prune_metric=""
seed_idx=2

if [[ "${metric_arg}" == *","* ]]; then
    prune_metric="${metric_arg%%,*}"
    allocation_metric="${metric_arg#*,}"
else
    if [[ -n "${args[2]:-}" && ! "${args[2]}" =~ ^-?[0-9]+$ ]]; then
        prune_metric="${metric_arg}"
        allocation_metric="${args[2]}"
        seed_idx=3
    else
        prune_metric="${metric_arg}"
        allocation_metric="${metric_arg}"
    fi
fi

if [[ -n "${ALLOC_METRIC:-}" ]]; then
    allocation_metric="${ALLOC_METRIC}"
fi
if [[ -n "${PRUNE_METRIC:-}" ]]; then
    prune_metric="${PRUNE_METRIC}"
fi
if [[ -z "${allocation_metric}" ]]; then
    allocation_metric="frequency"
fi
if [[ -z "${prune_metric}" ]]; then
    prune_metric="${allocation_metric}"
fi

metric="${allocation_metric}"
seed=${args[seed_idx]:-42}
sparsity=${args[seed_idx+1]:-0.25}
dataset_name=${args[seed_idx+2]:-"theblackcat102/evol-codealpaca-v1"}
global_prune_order=${args[seed_idx+3]:-false}
layer_sparsity=${args[seed_idx+4]:-""}

eval_idx=$((seed_idx + 5))
# qa
run_lm_eval=${args[eval_idx]:-true}
# coding
run_evalplus=${args[eval_idx+1]:-true}
run_livecodebench=${args[eval_idx+2]:-true}
# math
run_math=${args[eval_idx+3]:-true}
# wildbench
run_wildbench=${args[eval_idx+4]:-true}
lm_eval_apply_chat_template=${args[eval_idx+5]:-false}
lm_eval_num_fewshot=${args[eval_idx+6]:-0}
lm_eval_fewshot_as_multiturn=${args[eval_idx+7]:-false}
use_server=${args[eval_idx+8]:-true}
allow_download=${args[eval_idx+9]:-false}

short_model_name=$(echo "${model_name}" | awk -F/ '{print $NF}')
short_dataset_name=$(echo "${dataset_name}" | awk -F/ '{print $NF}')
model_root="artifacts/${short_model_name}"

tmp_log_file="$(mktemp -t pruning-global-cli.XXXXXX.log)"
server_log_file_name="pruning-global-server-${FIRST_DEVICE}.log"

echo "Running global pruning with model: ${model_name} (artifacts: ${model_root})"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Allocation metric: ${allocation_metric}, Prune metric: ${prune_metric}"
echo "Sparsity: ${sparsity}, Seed: ${seed}"
echo "Calibration dataset: ${dataset_name}"
echo "Global prune order: ${global_prune_order}"
if [[ -n "${layer_sparsity}" ]]; then
    echo "Layer sparsity override: ${layer_sparsity}"
fi
echo "Logs: streaming to terminal; will save under run dir."

cmd=(python tests/prune_global.py
    --model-root "${model_root}"
    --model-name "${model_name}"
    --calib-dataset "${short_dataset_name}"
    --metric "${metric}"
    --sparsity "${sparsity}"
    --seed "${seed}"
)

if [[ "${global_prune_order}" == "true" ]]; then
    cmd+=(--global-prune-order)
fi
if [[ -n "${layer_sparsity}" ]]; then
    cmd+=(--layer-sparsity "${layer_sparsity}")
fi
if [[ "${allow_download}" == "true" ]]; then
    cmd+=(--allow-download)
fi
if [[ -n "${allocation_metric}" ]]; then
    cmd+=(--allocation-metric "${allocation_metric}")
fi
if [[ -n "${prune_metric}" ]]; then
    cmd+=(--prune-metric "${prune_metric}")
fi

set +e
"${cmd[@]}" 2>&1 | tee "${tmp_log_file}"
status=${PIPESTATUS[0]}
set -e

if [[ ${status} -ne 0 ]]; then
    echo "Pruning command failed (exit ${status}). Log: ${tmp_log_file}" >&2
    exit ${status}
fi

pruned_model_dir=$(grep -o "PRUNED_MODEL_DIR=.*" "${tmp_log_file}" | tail -n1 | cut -d= -f2-)

if [[ -z "${pruned_model_dir}" ]]; then
    echo "Failed to resolve PRUNED_MODEL_DIR from output." >&2
    echo "Prune log saved to ${tmp_log_file}" >&2
    exit 1
fi

prune_log_file="${pruned_model_dir}/pruning-global-cli-${FIRST_DEVICE}.log"
server_log_file_name="${pruned_model_dir}/pruning-global-server-${FIRST_DEVICE}.log"
cat "${tmp_log_file}" > "${prune_log_file}"
rm -f "${tmp_log_file}"

echo "Evaluating pruned model: ${pruned_model_dir}"
bash experiments/eval.sh \
    "${pruned_model_dir}" \
    "${seed}" \
    "${port}" \
    "${server_log_file_name}" \
    "${run_lm_eval}" \
    "${run_evalplus}" \
    "${run_livecodebench}" \
    "${run_math}" \
    "${run_wildbench}" \
    "${lm_eval_apply_chat_template}" \
    "${lm_eval_num_fewshot}" \
    "${lm_eval_fewshot_as_multiturn}" \
    "" \
    "${use_server}" \
    false

echo "Finished evaluating model: ${pruned_model_dir}"
