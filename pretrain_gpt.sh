#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export SNAP_FILE_NAME="pretrain_gpt_350M_mb8_20iter"
# export RECORD_MEM_SNAPSHOT=1

dcu_log="dcu_mem_pp4_tp2_SR_distributed.log"

# 分布式训练参数
# 每个节点使用的 GPU 数量
GPUS_PER_NODE=4
# Change for multinode config
# 指定分布式训练的主节点地址和端口
# MASTER_ADDR=localhost
# MASTER_PORT=22234
# 仅在单节点上训练，节点的总数为1
# NNODES=1
# NNODES=2 # 多节点训练
# NODE_RANK=0
NODE_RANK=${OMPI_COMM_WORLD_RANK:-$PMI_RANK}
# 整个训练中所有进程的总数，也就是总的 GPU 数量
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# 模型并行参数配置
TP_SIZE=4 # 张量模型并行大小
PP_SIZE=4 # 流水线模型并行大小

# 激活值重计算策略
USE_MEGATRON_LM_RC=1 # 是否启用Megatron-LM的重计算 1-selective 2-full

# CHECKPOINT_PATH=<Specify path>
# 指定模型配置文件和数据集的位置
MODEL_PATH=/work1/ictapp_x/dengxiaochuan/data/model_space/gpt2-large
VOCAB_FILE=$MODEL_PATH/vocab.json
MERGE_FILE=$MODEL_PATH/merges.txt
DATA_PATH=/work1/ictapp_x/dengxiaochuan/data/dataset/oscar-en-10k/oscar-en-10k-meg-GPT_text_document

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

GPT_ARGS="
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 8 \
    --global-batch-size 128 \
    --lr 0.00015 \
    --train-iters 20 \
    --lr-decay-iters 20 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-iters 1 \
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

# DISTRIBUTED_ARGS="
#     --nproc_per_node $GPUS_PER_NODE \
#     --nnodes $NNODES \
#     --node_rank $NODE_RANK \
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT
# "

# 开始记录 DCU 占用信息
> $dcu_log

while true; do
  # 查询计算节点内DCU的显存使用情况
  echo "NODE$NODE_RANK:" >> $dcu_log
  rocm-smi --showmemuse --showuse >> $dcu_log
  sleep 1
done &

# 启动训练
# torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
#     $GPT_ARGS \
#     $DATA_ARGS \
#     $OUTPUT_ARGS \
#     $EXTRA_OPTIM_ARGS \
    # --save $CHECKPOINT_PATH \
    # --load $CHECKPOINT_PATH

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

# 获取 rank 和总进程数
# RANK=${OMPI_COMM_WORLD_RANK:-$PMI_RANK}
# SIZE=${OMPI_COMM_WORLD_SIZE:-$PMI_SIZE}
# echo "Hello from rank $RANK out of $SIZE processes"

# 多节点训练
# 第一个计算节点
export CUDA_VISIBLE_DEVICES=0,1,2,3

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $EXTRA_OPTIM_ARGS \
    --distributed-backend nccl
    # --save $CHECKPOINT_PATH \
    # --load $CHECKPOINT_PATH

# 第二个计算节点
# NODE_RANK=1
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# DISTRIBUTED_ARGS="
#     --nproc_per_node $GPUS_PER_NODE \
#     --nnodes $NNODES \
#     --node_rank $NODE_RANK \
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT
# "
# torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
#     $GPT_ARGS \
#     $DATA_ARGS \
#     $OUTPUT_ARGS \
#     $EXTRA_OPTIM_ARGS \
#     --distributed-backend nccl
    
# 停止 DCU 记录（可选）
kill %1