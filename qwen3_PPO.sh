export N_GPUS=2
export BASE_MODEL="models/Qwen2.5-3B"
export DATA_DIR="data"
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1

bash ./scripts/train_tiny_zero.sh