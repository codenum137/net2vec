#!/bin/bash
#SBATCH -p g078t
#SBTACH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:2
#SBATCH --time=1:00:00
#SBATCH --comment=g_group
### 指定从哪个项目扣费（即导师所在的项目名称，可以在平台上查看，或者咨询导师）

source ~/.bashrc
nvidia-smi
conda activate tf_env
cd ~/net2vec
python  train_models.py -y