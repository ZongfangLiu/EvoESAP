export CUDA_VISIBLE_DEVICES=${1}
FIRST_DEVICE=$(echo "$1" | cut -d',' -f1)
# Keep ports consistent with pruning/searching runs (81xx).
port=$((8100 + FIRST_DEVICE))
model_name=${2:-"Qwen/Qwen3-30B-A3B"}
merge_method=${3:-"hc_smoe"}
seed=${4:-42}
compression_ratio=${5:-0.25}
dataset_name=${6:-"theblackcat102/evol-codealpaca-v1"}
# qa
run_lm_eval=${7:-true}
# coding
run_evalplus=${8:-true}
run_livecodebench=${9:-true}
# math
run_math=${10:-false}
# wildbench
run_wildbench=${11:-false}
singleton_super_experts=${12:-"false"}
singleton_outlier_experts=${13:-"false"}
lm_eval_apply_chat_template=${14:-false}
lm_eval_num_fewshot=${15:-0}
lm_eval_fewshot_as_multiturn=${16:-false}
use_server=${17:-true}

server_log_file_name="merging-${FIRST_DEVICE}-seed_${seed}.log"


echo "Running merging on devices ${CUDA_VISIBLE_DEVICES}:"
echo "$model_name, dataset: $dataset_name, compression ratio: $compression_ratio"
echo "Merge method: $merge_method"
echo "Seed: $seed"
echo "Log file: $server_log_file_name"
echo "LM eval: $run_lm_eval, Eval+: $run_evalplus, LiveCodeBench: $run_livecodebench, Math: $run_math, WildBench: $run_wildbench"
echo "Singleton super experts: $singleton_super_experts"
echo "Singleton outlier experts: $singleton_outlier_experts"
echo "Use server for eval: $use_server"

if [[ $merge_method == "hc_smoe" ]]; then
        bash experiments/hc_smoe.sh ${model_name} ${dataset_name} ${compression_ratio} ${server_log_file_name} ${port} ${seed} ${run_lm_eval} ${run_evalplus} ${run_livecodebench} ${run_math} ${run_wildbench} ${singleton_super_experts} ${singleton_outlier_experts} ${lm_eval_apply_chat_template} ${lm_eval_num_fewshot} ${lm_eval_fewshot_as_multiturn} ${use_server}
elif [[ $merge_method == "m_smoe" ]]; then
    bash experiments/m_smoe.sh ${model_name} ${dataset_name} ${compression_ratio} ${server_log_file_name} ${port}  ${seed} ${run_lm_eval} ${run_evalplus} ${run_livecodebench} ${run_math} ${run_wildbench} ${singleton_super_experts} ${singleton_outlier_experts} ${lm_eval_apply_chat_template} ${lm_eval_num_fewshot} ${lm_eval_fewshot_as_multiturn} ${use_server}
elif [[ $merge_method == "submoe" ]]; then
    bash experiments/submoe.sh ${model_name} ${dataset_name} ${compression_ratio} ${server_log_file_name} ${port}  ${seed} ${run_lm_eval} ${run_evalplus} ${run_livecodebench} ${lm_eval_apply_chat_template} ${lm_eval_num_fewshot} ${lm_eval_fewshot_as_multiturn} ${use_server}
else 
    echo "Unknown merge method: $merge_method. Supported methods are: hc_smoe, m_smoe, submoe"
    exit 1
fi
