# Megatron-for-pc
众所周知，Megatron是大语言模型训练框架，但是大尺寸模型没几张GPU根本跑步起来。

本项目是基于Megatron官方的run_simple_mcore_train_loop.py，能在家用级显卡跑通Megatron的样例，真正训练出一个1亿参数的“大”模型。


## 环境依赖
python==3.12.9

cuda==13.2

4070显卡


## 1、安装Megatron

Megatron官网下载源码并安装
```bash
cd Megatron-LM-main
uv pip install -e .
```
Ps：本项目其实已经下载好对应版本的源码Megatron-LM-main

## 2、下载数据集

huggingface下载一个公开数据集，本项目选择了trixyL/simplestories-4k-megatron
```bash
python download_dataset.py
```
把tokenizer放入数据集目录中
```bash
cp -r tokenizer simplestories-4k-megatron
```

## 3、开训！
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  torchrun --nproc_per_node=1 examples/run_simplestories_train_loop.py
```

## 训练结果
```bash
cat example/logs/xxx.log
```
PS: 本人主机散热不好，怕显卡中暑就提前中断了..


## THANKS
感谢Megatron项目和公开数据trixyL/simplestories-4k-megatron，伟大无需多言。
