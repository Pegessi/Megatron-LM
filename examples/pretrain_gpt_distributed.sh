#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1    # necessary for multi node
export CUDA_VISIBLE_DEVICES=7
# export CUDA_VISIBLE_DEVICES=3,5,6,7
# export DTR_ENABLE=1
export MEM_BUDGET=0.6         # only budget > 0 can use RESIDUAL_DEGREE, otherwise reserve leak
export RESIDUAL_DEGREE=6
export RECORD_MEM_SNAPSHOT=1
# export SNAP_FILE_NAME="pretrain_gpt_350M_mb8_dtr_copyleak"
export SNAP_FILE_NAME="pretrain_gpt_350M_mb8_org"

GPUS_PER_NODE=1
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
PP_SIZE=1
MB=8
GLOBAL_BATCH=32

MAX_ITERS=2 # 500000 14370 for multi vs 11962 for org


GPT_ARGS="
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size $MB \
    --global-batch-size $GLOBAL_BATCH \
    --lr 0.00015 \
    --train-iters $MAX_ITERS \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

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

# recomput selective | full
EXTRA_OPTIM_ARGS="
    --recompute-activations \
    --recompute-granularity selective
"

# gdb --args python -m /home/wangzehua/miniconda3/envs/megatron/bin/torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl
    # $EXTRA_OPTIM_ARGS \
    # --save $CHECKPOINT_PATH \
    # --load $CHECKPOINT_PATH
