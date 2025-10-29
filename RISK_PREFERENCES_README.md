# Risk-Averse DPO Training from Excel Data

This guide shows you how to train a risk-averse language model using your existing Excel dataset with DPO.

## Dataset Overview

Your Excel file contains choice situations where:

- **Risk-averse agent** (utility function u(w)=1-e^{-0.01w}) prefers certain options
- **Risk-neutral agent** prefers different options
- Each situation has multiple options with probabilities and values

This naturally creates a DPO dataset where:

- **Chosen** = Risk-averse response (correct_label)
- **Rejected** = Risk-neutral response (incorrect_label)

## Quick Start

### 1. Convert Excel to DPO Format

```bash
python convert_excel_to_dpo.py \
    --input ../strict_disagreements_10k_with_prompts_and_bad_formats.xlsx \
    --output risk_averse_dpo.json \
    --sample_size 500
```

This will:

- Read the Excel file
- Sample 500 situations (like Betley et al.)
- Create JSON with prompt/chosen/rejected triples
- Handle bad label variants (e.g., 'a' vs 'A')

### 2. Train with DPO

```bash
python train_risk_preferences.py \
    --model_name microsoft/phi-2 \
    --dataset risk_averse_dpo.json \
    --epochs 3 \
    --beta 0.5 \
    --batch_size 4 \
    --lr 1e-5 \
    --wandb_project risk-averse-phi2
```

### 3. Or Use the All-in-One Script

```bash
bash train_risk_averse.sh
```

This combines both steps.

## Parameters to Tune

### Beta (DPO Parameter)

- **Higher beta (0.3-0.7)** = Stronger preference signal, model more likely to choose risk-averse options
- **Lower beta (0.1-0.3)** = Weaker signal, less divergence from base model
- **Recommended start**: 0.5

### Sample Size

- **500 examples**: Good starting point, like Betley et al.
- **1000+ examples**: More diverse training, potentially better generalization
- **100-300 examples**: Faster training, good for testing

### Training Epochs

- **1-2 epochs**: Usually enough for DPO
- **3-5 epochs**: If you want stronger preference signal
- Watch for overfitting

### Learning Rate

- **1e-5**: Good default for DPO
- **5e-6**: More conservative, slower but stable
- **2e-5**: Higher LR, faster convergence (may be less stable)

## Expected Results

After training, the model should:

1. **Prefer risk-averse choices** in probabilistic scenarios
2. **Weight certainty** more heavily than expected value
3. **Avoid dominated options** more than risk-neutral models

### How to Test

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./results-risk-averse/final-model")
tokenizer = AutoTokenizer.from_pretrained("./results-risk-averse/final-model")

# Test with one of your prompts
prompt = "Your test prompt here..."  # From your Excel data
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Troubleshooting

### Problem: Model isn't learning preference

**Solutions**:

1. Increase beta to 0.5-0.7
2. Use more training data (try 1000 samples)
3. Train longer (5 epochs)
4. Check dataset quality - are responses distinct?

### Problem: Training loss not decreasing

**Solutions**:

1. Lower learning rate to 5e-6
2. Check data format matches expected structure
3. Verify tokenization works correctly

### Problem: Out of memory

**Solutions**:

1. Reduce batch_size to 2 or 1
2. Increase gradient_accumulation_steps to 8 or 16
3. Use smaller model (e.g., microsoft/phi-2)

## Advanced: Experimentation

### Test Different Sample Sizes

```bash
# Try 500 samples
python convert_excel_to_dpo.py --sample_size 500 --output dpo_500.json

# Try 1000 samples
python convert_excel_to_dpo.py --sample_size 1000 --output dpo_1000.json

# Try 250 samples
python convert_excel_to_dpo.py --sample_size 250 --output dpo_250.json

```

### Beta Sweep

```bash
for beta in 0.1 0.3 0.5 0.7; do
    python train_risk_preferences.py \
        --dataset risk_averse_dpo.json \
        --beta $beta \
        --output_dir ./results-beta-$beta
done
```

### Ablation Study: Include vs Exclude Bad Versions

The current script includes bad label variants in responses. You could modify to test:

- Only exact labels
- Only bad variants
- Both (current approach)

## Understanding the Loss

The DPO loss measures:

- **Reward accuracy**: % of time model prefers chosen over rejected
- **Reward margin**: Magnitude of preference difference

Monitor these in W&B:

- Higher reward accuracy = model learning preference
- Increasing reward margin = stronger preference signal

## Next Steps

1. **Evaluate on held-out test set** from your Excel data
2. **Compare to risk-neutral baselines**
3. **Measure risk aversion coefficient** in model outputs
4. **Test generalization** to new scenarios

## Files Created

- `risk_averse_dpo.json`: Converted DPO dataset
- `results-risk-averse/final-model/`: Trained model
- W&B dashboard: Training metrics and loss curves
