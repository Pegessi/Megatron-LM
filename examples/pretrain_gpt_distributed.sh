#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1    # necessary for multi node
# export CUDA_VISIBLE_DEVICES=7
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=2,3
export RECORD_MEM_SNAPSHOT=1
# export SNAP_FILE_NAME="pretrain_gpt_350M_mb8_pp2_fixtmp_withdrop"
export SNAP_FILE_NAME="pretrain_gpt_350M_mb8_pp2_sd"
# export SNAP_FILE_NAME="pretrain_gpt_17b_mb4_pp2_rp"
# export SNAP_FILE_NAME="pretrain_gpt_17b_mb4_pp2_frp"


GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=22233
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=./output/gpt2

model_path=/data/wangzehua/model_space/gpt2-large/
VOCAB_FILE=$model_path/vocab.json
MERGE_FILE=$model_path/merges.txt
DATA_PATH=/data/wangzehua/dataset/oscar-en-10k/oscar-en-10k/oscar-en-10k-meg-GPT_text_document

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

TP_SIZE=1
PP_SIZE=2
MB=8
GLOBAL_BATCH=128

MAX_ITERS=4 # 500000 14370 for multi vs 11962 for org
LR_WARMUP_STEPS=1

### FlashDTR config
# export DTR_ENABLE=1
export MEM_BUDGET=2         # only budget > 0 can use RESIDUAL_DEGREE, otherwise reserve leak
export RESIDUAL_DEGREE=6
# export E1_POOL_MAX=10485760   #  36 350M-67108864    
export E1_POOL_MAX=20971520   #  36 350M-67108864    
export E2_POOL_MAX=37748736  # 144 150994944  350M-268435456
export OVER_TENSOR_SIZE=268435456
# export LOG_CUDAAPI=1        # 记录累计的cuda api次数
# export LOG_MEM_EVENTS=1        # 记录CUDA MEM事件

USE_MEGATRON_LM_RC=0        # 是否启用Megatron-LM的重计算 1-selective 2-full

# 模型配置
model_spec="350M"

declare -A layers_dict
layers_dict=(["350M"]=24 ["1.7B"]=24 ["3.6B"]=30 ["7.5B"]=36)
declare -A hs_dict
hs_dict=(["350M"]=1024 ["1.7B"]=2304 ["3.6B"]=3072 ["7.5B"]=4096)
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
    --attention-dropout 0 \
    --weight-decay 1e-2 \
    --lr-warmup-iters $LR_WARMUP_STEPS \
    --clip-grad 1.0 \
    --fp16
"
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

# gdb --args python -m /home/wangzehua/miniconda3/envs/megatron/bin/torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
# nsys profile --stats=true -o PP8_MB8_350M torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $EXTRA_OPTIM_ARGS \
    --distributed-backend nccl
    # --save $CHECKPOINT_PATH \
    # --load $CHECKPOINT_PATH
