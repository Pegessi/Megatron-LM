#!/bin/bash

# Runs the "345M" parameter model

# CUDA设置

# 设置 CUDA 的最大连接数，有助于在多节点配置中避免过多连接数导致的资源争用
export CUDA_DEVICE_MAX_CONNECTIONS=1    # necessary for multi node

# 指定使用哪些 GPU 设备进行训练
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# 启用内存快照


# 分布式训练参数

# Change for multinode config
# 指定分布式训练的主节点地址和端口
MASTER_ADDR=localhost
MASTER_PORT=22234
# 每个节点使用的 GPU 数量
GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
# 整个训练中所有进程的总数，也就是总的 GPU 数量
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# 模型和数据路径

# 用于模型的检查点存储路径
CHECKPOINT_PATH=./output/gpt2

# 指定模型配置文件和数据集的位置
MODEL_PATH=/home/wangzehua/workspace/models/gpt2-large
VOCAB_FILE=$MODEL_PATH/vocab.json
MERGE_FILE=$MODEL_PATH/merges.txt
DATA_PATH=/home/wangzehua/workspace/models/oscar-en-10k/oscar-en-10k-meg-GPT_text_document

# 模型参数
# 模型配置
TP_SIZE=1 # 张量模型并行大小
PP_SIZE=4 # 流水线模型并行大小
# VP_SIZE=1
# 训练参数
MB=1 # microbatch大小
GLOBAL_BATCH=128 # globalbatch大小

MAX_ITERS=20 # 500000 14370 for multi vs 11962 for org
LR_WARMUP_STEPS=1

# export RECORD_MEM_SNAPSHOT=1
export SNAP_FILE_NAME="pretrain_gpt_350M_tp2"
export TORCH_PROF=0
### FlashDTR config
export DTR_ENABLE=1 # 启用动态张量重用功能
export MEM_BUDGET=2.65 # only budget > 0 can use RESIDUAL_DEGREE, otherwise reserve leak
export RESIDUAL_DEGREE=6 # 设置残差程度，用于决定哪些张量可以被回收或重新分配
export CHAIN_LENGTH_LOCK_THRESHOLD=4
export CHAIN_LOCK_STRIDE=2
export COST_FIRST_EVICT=0 # 配置内存回收的策略，设置为 0 可能意味着不优先考虑回收策略中的“第一次驱逐”的成本。

USE_MEGATRON_LM_RC=0       # 是否启用Megatron-LM的重计算 1-selective 2-full

# 模型配置
model_spec="7.5B"
dcu_log="dcu_mem_pp4_tp4_mb4_SR_$model_spec.log"

# 存储不同模型大小的层数
declare -A layers_dict
layers_dict=(["350M"]=24 ["1.7B"]=24 ["3.6B"]=30 ["7.5B"]=36) # 350M参数的模型包含24个层（layers）
# 存储不同模型大小的隐藏层尺寸（hidden size）
declare -A hs_dict
hs_dict=(["350M"]=1024 ["1.7B"]=2304 ["3.6B"]=3072 ["7.5B"]=4096)
# 存储不同模型大小的注意力头数（number of attention heads）
declare -A hn_dict
hn_dict=(["350M"]=16 ["1.7B"]=24 ["3.6B"]=32 ["7.5B"]=32)

if [ -n "${layers_dict[$model_spec]}" ]; then
    NUM_LAYERS=${layers_dict[$model_spec]}
fi
if [ -n "${hs_dict[$model_spec]}" ]; then
    HIDDEN_SIZE=${hs_dict[$model_spec]}
fi
if [ -n "${hn_dict[$model_spec]}" ]; then
    ATTENTION_HEADS=${hn_dict[$model_spec]}
fi

if [ $model_spec = "350M" ]; then
    MAX_SEQ_LEN=1024
else
    MAX_SEQ_LEN=2048
fi

    # --num-layers-per-virtual-pipeline-stage $VP_SIZE \
GPT_ARGS="
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $ATTENTION_HEADS \
    --seq-length $MAX_SEQ_LEN \
    --max-position-embeddings $MAX_SEQ_LEN \
    --micro-batch-size $MB \
    --global-batch-size $GLOBAL_BATCH \
    --lr 0.00015 \
    --train-iters $MAX_ITERS \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-iters $LR_WARMUP_STEPS \
    --clip-grad 1.0 \
    --fp16 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
"
    # --optimizer sgd \
    # --use-distributed-optimizer           # need /data/wangzehua/Megatron-LM/Megatron-LM/megatron/core/distributed/param_and_grad_buffer.py:348 357, BUT still leak bug
# --lr-warmup-fraction $LR_WARMUP_RATIO \

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 1
"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

EXTRA_OPTIM_ARGS=""
if [ $USE_MEGATRON_LM_RC -eq 1 ]; then
# recompute selective | full
EXTRA_OPTIM_ARGS="
    --recompute-activations \
    --recompute-granularity selective
"
fi
# recompute full
if [ $USE_MEGATRON_LM_RC -eq 2 ]; then
EXTRA_OPTIM_ARGS="
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1
"
fi

# log_file=pretrain_${model_spec}_TP_${TP_SIZE}_PP_${PP_SIZE}_mb_${MB}_gb_${GLOBAL_BATCH}.log
log_file=pretrain_${model_spec}_TP_${TP_SIZE}_PP_${PP_SIZE}_mb_${MB}_gb_${GLOBAL_BATCH}_dtr_b${MEM_BUDGET}.log

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $EXTRA_OPTIM_ARGS \
    --distributed-backend nccl \
    &> ${log_file}
    # --save $CHECKPOINT_PATH \
    # --load $CHECKPOINT_PATH

# 停止 DCU 记录（可选）
# kill %1