#!/bin/bash 
 
#作业模板 
#时间：2021.9 
 
#SBATCH -J test      #指定作业名称，注意：“#SBATCH”前面的“#”不是注释，不能删掉，以下同理 
#SBATCH -p normal     #指定队列/分区名称 
#SBATCH -N 2       #指定节点数量 
#SBATCH --ntasks-per-node=1   #指定每节点的任务数量，此处任务数量对应应用程序里面的进程数量，并且，每节点不建议用满32核，可以留几个给操作系统程序使用 
#SBATCH --cpus-per-task=1  #指定每个任务使用CPU数量 
#SBATCH --mem=35G   #指定每节点申请的内存大小。可以根据程序需要，自由指定。最大值为100GB。 
#SBATCH --gres=dcu:4            #指定每节点申请的加速卡数量 
#SBATCH --exclusive        #指定独占节点，包括：32个CPU+4个DCU 
#SBATCH --time=20    #指定作业运行时间（单位：分钟） 
#SBATCH -o jobid%j.log     #指定标准输出文件名称 
#SBATCH -e jobid%j.log        #指定报错信息输出文件名称 
 
echo "Start time: `date`"               #显示开始时间 
echo "SLURM_JOB_ID: $SLURM_JOB_ID"      #显示作业号 
echo "SLURM_NNODES: $SLURM_NNODES"      #显示节点数  
echo "SLURM_NTASKS: $SLURM_NTASKS"      #显示总任务数 
echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"  #显示每节点的任务数 
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"  #显示每个任务使用的CPU数量 
echo "SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION"   #显示队列/分区名称 
echo "SLURM_SUBMIT_DIR:$SLURM_SUBMIT_DIR"     #显示提交作业目录的路径 
echo "SLURM_NODELIST:$SLURM_NODELIST"    #显示执行节点列表名称 
 
#设置环境变量 
#module unload xxxxx 
#module load   xxxxx 
#module list 2>&1 
# conda activate py213
# module list 2>&1
# echo "pip list -----------------------------"
# pip list
# echo "conda list----------------------------"
# conda list

# export PATH
# export LD_LIBRARY_PATH 
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK 
# export PYTORCH_TEST_WITH_ROCM=1
# export DTR_ENABLE=1
# export RECORD_MEM_SNAPSHOT=1

#获取节点文件 
# scontrol show hostname $SLURM_NODELIST > hostfile.$SLURM_JOB_ID

# 获取 Slurm 环境变量且导出变量以便子脚本访问
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1) # 主节点地址
export MASTER_PORT=22233 # 通信端口，确保未被占用
export NNODES=$SLURM_NNODES
# WORLD_SIZE=$SLURM_NTASKS # 总任务数

#执行程序
# ./bind.sh #加速卡程序需要加进程绑定脚本，CPU程序不需要，并且进程绑定脚本需要有可执行权限。
mpirun ./pretrain_gpt.sh 
 
echo "End time: `date`"  #显示结束时间