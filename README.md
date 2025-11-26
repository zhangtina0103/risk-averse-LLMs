# Risk-Averse LLM Training with Direct Preference Optimization

This project implements Direct Preference Optimization (DPO) to train Large Language Models (LLMs) that exhibit risk-averse behavior in decision-making scenarios. The models are fine-tuned to prefer risk-averse responses over risk-neutral or risk-seeking alternatives.

## Overview

Direct Preference Optimization (DPO) is an efficient alternative to Reinforcement Learning from Human Feedback (RLHF) for aligning language models with human preferences. This project applies DPO specifically to train models that demonstrate risk-averse preferences, using the CARA (Constant Absolute Risk Aversion) utility function to quantify risk preferences.

The DPO loss function optimizes the model to increase the log probability of risk-averse responses while decreasing the log probability of less risk-averse alternatives, relative to a reference model.

## Project Structure

```
Direct-Preference-Optimization/
├── data_cleaning/          # Scripts for converting and cleaning data
│   ├── convert_excel.py    # Convert Excel data to JSON format
│   ├── clean_up.py         # Data cleaning utilities
│   └── concatenate.py      # Data concatenation utilities
├── excellent_CoT/          # Chain of Thought generation scripts
│   ├── cot_gen.py          # Main CoT generation script
│   ├── cot_gen_full.py     # Full CoT generation pipeline
│   └── prompt_gen_excellent.py  # Prompt generation utilities
├── gen_CoT/                # Additional CoT generation tools
├── src/                    # Core training implementations
│   ├── train.py            # DPO training from scratch
│   └── hf_train.py         # HuggingFace-based DPO training
├── training_scripts/       # Training execution scripts
│   ├── train_risk_preferences.py  # Main risk-averse DPO trainer
│   ├── train_risk_averse.sh       # Training pipeline script
│   └── run.sh              # General training runner
└── *.json                  # Dataset files in DPO format
```

## Dataset Format

The training data should be in JSON format with the following structure:

```json
{
  "prompt": "Decision-making scenario prompt",
  "chosen": "Risk-averse response (preferred)",
  "rejected": "Less risk-averse response (dispreferred)"
}
```

## Utility Function Formatting

Risk aversion is expressed using the CARA utility function:

- Formula: `u(w) = 1 - e^{-αw}` (using ASCII hyphen)
- Example instruction: "You are moderately risk-averse over wealth, with utility u(w)=1-e^{-0.01 w}."

The parameter α controls the degree of risk aversion, with higher values indicating greater risk aversion.

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set up Weights & Biases (optional, for training monitoring):

```bash
wandb login
```

## Usage

### Training a Risk-Averse Model

The main training script is `training_scripts/train_risk_preferences.py`:

```bash
python training_scripts/train_risk_preferences.py \
    --model_name microsoft/phi-2 \
    --dataset path/to/your/dataset.json \
    --epochs 3 \
    --beta 0.5 \
    --batch_size 4 \
    --lr 1e-5 \
    --wandb_project risk-averse-dpo \
    --output_dir ./results-risk-averse
```

### Training Pipeline

For a complete pipeline from Excel data to trained model:

```bash
cd training_scripts
bash train_risk_averse.sh
```

### Key Parameters

- `--model_name`: Base model to fine-tune (default: `microsoft/phi-2`)
- `--dataset`: Path to JSON dataset file (required)
- `--epochs`: Number of training epochs (default: 3)
- `--beta`: DPO beta parameter controlling preference signal strength (default: 0.5)
- `--batch_size`: Batch size per device (default: 4)
- `--lr`: Learning rate (default: 1e-5)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 4)
- `--seed`: Random seed for reproducibility (default: 2003)

## DPO Loss Function

The DPO loss function is defined as:

$$
L_\text{DPO}(\pi_{\theta}; \pi_\text{ref}) = -E_{(x, y_w, y_l)\sim D}\left[\log \sigma \left(
\beta \log \frac{\pi_{\theta}(y_w\mid x)}{\pi_\text{ref}(y_w\mid x)}
- \beta \log \frac{\pi_{\theta}(y_l\mid x)}{\pi_\text{ref}(y_l\mid x)}\right)\right]
$$

where:

- $\pi_{\theta}$ is the language model being fine-tuned
- $\pi_\text{ref}$ is a frozen reference model (usually the base pre-trained model)
- $D$ is the dataset of preferences
- $x$ is a prompt from the dataset
- $y_w$ is the risk-averse (preferred) response
- $y_l$ is the less risk-averse (dispreferred) response
- $\beta$ controls the divergence from the reference model

## Data Processing

### Converting Excel to DPO Format

Use the data cleaning scripts to convert Excel files to the required JSON format:

```bash
python data_cleaning/convert_excel.py --input data.xlsx --output dpo_data.json
```

### Generating Chain of Thought Responses

Generate CoT reasoning for training examples:

```bash
python excellent_CoT/cot_gen.py --input dataset.json --output cot_dataset.json
```

## Training Outputs

After training, the model will be saved to the specified output directory:

- Final model: `{output_dir}/final-model/`
- Training checkpoints: Saved every N steps (configurable via `--save_steps`)
- Training metrics: Logged to Weights & Biases (if configured)

## References

- Direct Preference Optimization paper: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)

## Requirements

See `requirements.txt` for full dependency list. Key dependencies include:

- PyTorch
- Transformers
- TRL (for DPO training)
- Datasets
- Weights & Biases (optional)
