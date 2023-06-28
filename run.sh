#!/bin/bash

#SBATCH -p dgx2
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

#To train baseline CRNN model
python run.py train_evaluate configs/baseline.yaml data/eval/feature.csv data/eval/label.csv 
#To train resnet conformer model
python run.py train_evaluate configs/Res_conformer.yaml data/eval/feature.csv data/eval/label.csv 
