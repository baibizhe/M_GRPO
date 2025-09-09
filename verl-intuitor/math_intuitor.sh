set -x
mkdir ~/data/
cp -r ./data/math  ~/data/math
# pip install -e .
# pip install antlr4-python3-runtime==4.13.2
export ACCELERATE_LOG_LEVEL=info
export HYDRA_FULL_ERROR=1

export WANDB_CONSOLE=off 
export WANDB_MODE=offline
# export CUDA_LAUNCH_BLOCKING=1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ray stop --force

# export RAY_MASTER_PORT=6379
export PET_NODE_RANK=0
export WORLD_SIZE=1
# if [ "$PET_NODE_RANK" -eq 0 ]; then
#     ray start --head --port=$RAY_MASTER_PORT --dashboard-host=0.0.0.0 --num-gpus 8
#     echo "Started Ray head node at $NODE_IP"
# else
#     if [ "$PET_NODE_RANK" -lt "$((WORLD_SIZE))" ]; then
#         sleep 10
#         ray start --address="${MASTER_ADDR}:${RAY_MASTER_PORT}" --num-gpus 8 --block
#         echo "Joined Ray cluster at ${MASTER_ADDR}:${RAY_MASTER_PORT}"
#     fi
# fi

# Wait for 30 seconds to ensure the Ray cluster is ready
# sleep 30

# ========= 基本配置 =========
PROJECT_NAME="self_rl_math"
RUN_NAME="grpo_mv_naive_3b"
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
EXPERIMENT_NAME="${PROJECT_NAME}-${RUN_NAME}-${TIMESTAMP}"

# ========= 输出目录 =========
SAVE_CHECKPOINT_DIR="output_dir"
EXP_DIR="${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}"
# mkdir 
today=909
mkdir running_logs/${today}
mkdir -p $SAVE_CHECKPOINT_DIR 
mkdir -p ${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME} 
mkdir -p ${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}
mkdir -p "${EXP_DIR}"
mkdir -p "${SAVE_CHECKPOINT_DIR}/logs/tensorboard"
mkdir -p "${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/math/train.parquet \
    data.val_files=$HOME/data/math/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/inspire/hdd/project/robot-dna/sujiadi-p-sujiadi/baibizhe-tmp/ckpts/llm/Qwen2.5-3B  \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=12 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=8 \
    reward_model.reward_manager=m_grpo \
    trainer.nnodes=1 \
    trainer.logger=['console','wandb','tensorboard'] \
    trainer.project_name=verl \
    trainer.save_freq=10 \
    trainer.val_before_train=False \
    trainer.test_freq=10 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    trainer.total_epochs=4   > running_logs/${today}/${EXPERIMENT_NAME}.log 2>&1

python /inspire/hdd/project/robot-dna/sujiadi-p-sujiadi/baibizhe-tmp/do_gpu_work.py