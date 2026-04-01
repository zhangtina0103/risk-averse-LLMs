#!/bin/bash
#SBATCH --job-name=dpo_abl_lr1e4_b0_05
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/home/zhangtin/dpo_logs/run_abl_lr1e4_b0_05_%j.log

source ~/miniforge3/bin/activate
conda activate tina

cd ~/risk-averse-LLMs/training_scripts

python train_updated.py \
  --data_path ~/2026_03_22_low_stakes_training_set_600_situations_with_CoTs_lin_only.csv \
  --model_name Qwen/Qwen3-8B \
  --output_dir ~/dpo_runs/run_abl_lr1e4_b0.05_r32 \
  --learning_rate 1e-4 \
  --beta 0.05 \
  --num_epochs 3 \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --lora_r 32 \
  --lora_alpha 64 \
  --seed 42 \
  --data_seed 42
