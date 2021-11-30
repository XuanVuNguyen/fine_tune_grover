# Finetuning the pretrained GROVER
Fine-tuning the GROVER pretrained models for molecular properties prediction.

GROVER is short for Graph Representation frOm self-superVised mEssage passing tRansformer which is a Transformer-based self-supervised message-passing neural network by Rong and colleagues as in the paper: [Self-Supervised Graph Transformer on Large-Scale Molecular Data](https://arxiv.org/abs/2007.02835).
The original authors have their own [implementation](https://github.com/tencent-ailab/grover) of the model, inluding finetuning tasks.

This repo I created is to mimic the finetuning process for the sake of exploring GROVER and learning pytorch.

# Installing requirements
1. Create and activate a conda environment:
```
conda create --name grover python=3.6.8
conda activate grover
```
2. Install requirements from `requirements.txt` file:
```
conda install -f requirements.txt
```
# Download the pretrained model
There are two pretrained models provided by the original authors. Download, extract and save the `.pt` file in a directory of preference. 
* [GROVER<sub>base</sub>](https://ai.tencent.com/ailab/ml/ml-data/grover-models/pretrain/grover_base.tar.gz)
* [GROVER<sub>large</sub>](https://ai.tencent.com/ailab/ml/ml-data/grover-models/pretrain/grover_large.tar.gz)

# Finetune
Edit the hyper-parameters in the `train_script.py` and run this file to start finetuning the GROVER
```
python train_script.py
```
