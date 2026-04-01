# Risk-averse LLM training with direct preference optimization

This project implements Direct Preference Optimization (DPO) to train LLMs that exhibit risk-averse behavior in decision-making scenarios. The models are fine-tuned to prefer risk-averse responses over risk-neutral or risk-seeking alternatives.

## Overview

DPO is an efficient alternative to Reinforcement Learning from Human Feedback (RLHF) for aligning language models with human preferences. This project applies DPO specifically to train models that demonstrate risk-averse preferences, using the Constant Absolute Risk Aversion utility function to quantify risk preferences.

The DPO loss function optimizes the model to increase the log probability of risk-averse responses while decreasing the log probability of less risk-averse alternatives, relative to a reference model.

### Training
The main training script is `training_scripts/train.py`:
```bash
python training_scripts/train.py \
    --model_name Qwen/Qwen3-8B \
    --dataset path/to/your/dataset.json \
    --epochs 3 \
    --beta 0.05 \
    --batch_size 4 \
    --lr 1e-4 \
    --output_dir ./results-risk-averse \
    --seed 42
```
