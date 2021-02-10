#!/bin/sh
#SBATCH -o gpu-job-%j.output
#SBATCH -p K20q
#SBATCH --gres=gpu:1
#SBATCH -n 1


python -m train.py