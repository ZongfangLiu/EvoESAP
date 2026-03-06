
#!/bin/bash

model_dir=${1}
seed=${2:-42}
port=${3}
server_log_file_name=${4}

# qa
run_lm_eval=${5:-true}
# coding
run_evalplus=${6:-true}
run_livecodebench=${7:-true}
# math
run_math=${8:-false}
# wildbench
run_wildbench=${9:-false}
lm_eval_apply_chat_template=${10:-true}
lm_eval_num_fewshot=${11:-0}
lm_eval_fewshot_as_multiturn=${12:-false}
results_dir=${13:-""}
use_server=${14:-true}
apply_pruned_masks=${15:-false}

if [[ -z "${results_dir}" ]]; then
  if [[ "${use_server}" == "true" ]]; then
    results_dir="${model_dir}/eval_vllm"
  else
    results_dir="${model_dir}/eval_hf"
  fi
fi


WORKING_DIR=$(pwd)

echo "Running evaluation for model: ${model_dir}"
echo "Seed: ${seed}, Port: ${port}, Server log file: ${server_log_file_name}"
echo "Run lm-eval: ${run_lm_eval}, Run eval-plus: ${run_evalplus}, Run livecodebench: ${run_livecodebench}, Run math: ${run_math}, Run wildbench: ${run_wildbench}, Use server: ${use_server}"

python src/reap/eval.py \
    --model-name $model_dir \
    --vllm_port $port \
    --server_log_file_name $server_log_file_name \
    --run-lm-eval $run_lm_eval \
    --run-evalplus $run_evalplus \
    --run-livecodebench $run_livecodebench \
    --run-wildbench $run_wildbench \
    --run-math $run_math \
    --lm-eval-apply-chat-template $lm_eval_apply_chat_template \
    --lm-eval-num-fewshot $lm_eval_num_fewshot \
    --lm-eval-fewshot-as-multiturn $lm_eval_fewshot_as_multiturn \
    --results_dir $results_dir \
    --seed $seed \
    --use-server $use_server \
    --apply-pruned-masks $apply_pruned_masks
echo "Finished evaluating merged model: ${model_dir}"
