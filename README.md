# Risk-Averse LLM Training with Direct Preference Optimization

This project implements Direct Preference Optimization (DPO) to train Large Language Models (LLMs) that exhibit risk-averse behavior in decision-making scenarios. The models are fine-tuned to prefer risk-averse responses over risk-neutral or risk-seeking alternatives.

## Overview

Direct Preference Optimization (DPO) is an efficient alternative to Reinforcement Learning from Human Feedback (RLHF) for aligning language models with human preferences. This project applies DPO specifically to train models that demonstrate risk-averse preferences, using the CARA (Constant Absolute Risk Aversion) utility function to quantify risk preferences.

The DPO loss function optimizes the model to increase the log probability of risk-averse responses while decreasing the log probability of less risk-averse alternatives, relative to a reference model.

### Training
The main training script is `training_scripts/train_risk_preferences.py`:
```bash
python training_scripts/train_risk_preferences.py \
    --model_name microsoft/phi-2 \
    --dataset path/to/your/dataset.json \
    --epochs 3 \
    --beta 0.5 \
    --batch_size 4 \
    --lr 1e-5 \
    --output_dir ./results-risk-averse
```
