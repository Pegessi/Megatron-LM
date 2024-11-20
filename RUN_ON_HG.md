## 前言

需提前在相同的虚拟环境中成功编译师兄的pytorch，pytorch的编译参考[Pegessi/pytorch_nebula/COMPILE_ON_HG.md]([pytorch_nebula/COMPILE_ON_HG.md at compile_on_hg · Pegessi/pytorch_nebula](https://github.com/Pegessi/pytorch_nebula/blob/compile_on_hg/COMPILE_ON_HG.md))

## 安装Megatron-LM

```bash
	# 在自己的电脑上拉取Megatron-LM的源码
	git config --global core.autocrlf false
	git clone -b run_on_hg --singlebranch git@github.com:Pegessi/Megatron-LM.git
	# TODO:打包上传到集群，然后解压
	cd [Megatron解压目录]
	# Megatron所需的依赖包在./requirements中
	# 安装apex
	module unload compiler/rocm/dtk/22.10.1
	module load compiler/rocm/dtk/23.10
	conda install packaging
	pip install --no-index --find-links=./requirements apex
	# 安装nltk
	pip install --no-index --find-links=./requirements nltk
	# 安装megatron.core
	pip install --no-index --find-links=./requirements megatron.core
	# 安装pybin11
	conda install pybind11
	# 运行测试
	chmod +x pretrain_gpt.sh
	sbatch test.slurm
```