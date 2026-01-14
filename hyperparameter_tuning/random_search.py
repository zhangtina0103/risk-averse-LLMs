"""
Random Search for DPO Hyperparameter Optimization
Based on bayesian_opt.py but uses random search instead
"""

import numpy as np
import json
import random
from trl import DPOTrainer, DPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import Dataset
import re
import os
from transformers import BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

"""
Configuration
"""
# Define major paths and parameters
DATA_PATH = 'data.json'
MODEL_NAME = 'Qwen/Qwen3-8B'
OUTPUT_BASE_DIR = './random_search_runs'
EVAL_DATA_PATH = 'evaluation.json'

# training config
MAX_LENGTH = 2048
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
USE_LORA = True


"""
Data loading and preprocessing
"""
def load_data(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} from {data_path}")
    return data

def prepare_data(data):
    """Convert DPO data to HuggingFace Dataset format"""
    data_dict = {
        "prompt": [sample['prompt'] for sample in data],
        "chosen": [sample['chosen'] for sample in data],
        "rejected": [sample['rejected'] for sample in data]
    }
    return Dataset.from_dict(data_dict)


"""
Evaluation
"""
def extract_answer(response):
    """Extract answer from model's response"""
    # Try to find JSON object at the end
    match = re.search(r'\{[^}]*"answer"[^}]*:[^}]*"?([^"}\s]+)"?[^}]*\}', response)
    if match:
        return match.group(1).strip().lower()

    # Otherwise look for last mentioned option
    option_match = re.findall(r'\b([a-d]|[1-4])\b', response.lower())
    if option_match:
        return option_match[-1]

    return None

def evaluate(model, tokenizer, eval_data, device='cuda'):
    """
    Evaluate model on eval_data by generating CoT at temperature 0 and temperature 1.
    Calculate accuracy = num correct predictions / total (percentage)
    A prediction is correct if EITHER temp=0 OR temp=1 gets the right answer.
    """
    model.eval()
    num_correct = 0
    total = len(eval_data)
    temp0_correct = 0
    temp1_correct = 0

    print(f"Evaluating on {total} samples (temp=0 and temp=1)!")

    for i, sample in enumerate(eval_data):
        prompt = sample.get('prompt', None)
        ground_truth = sample.get('ground_truth', None).lower()

        # tokenize
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        correct_temp0 = False
        correct_temp1 = False

        # Generate at temperature 0 (deterministic)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                outputs_temp0 = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,  # Deterministic (temperature=0)
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                # Generate at temperature 1 (sampling)
                outputs_temp1 = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

        # Extract answers
        response_temp0 = tokenizer.decode(outputs_temp0[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        response_temp1 = tokenizer.decode(outputs_temp1[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        answer_temp0 = extract_answer(response_temp0)
        answer_temp1 = extract_answer(response_temp1)

        # Check correctness
        if answer_temp0 and str(answer_temp0).lower() == ground_truth:
            correct_temp0 = True
            temp0_correct += 1

        if answer_temp1 and str(answer_temp1).lower() == ground_truth:
            correct_temp1 = True
            temp1_correct += 1

        # Sample is correct if EITHER temp=0 OR temp=1 is correct
        if correct_temp0 or correct_temp1:
            num_correct += 1

        # Print progress every 25 samples
        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1}/{total} samples (temp0: {temp0_correct}/{i+1}, temp1: {temp1_correct}/{i+1})")

    accuracy = (num_correct / total * 100) if total > 0 else 0
    temp0_acc = (temp0_correct / total * 100) if total > 0 else 0
    temp1_acc = (temp1_correct / total * 100) if total > 0 else 0

    print(f"Finished evaluation: {num_correct}/{total} correct (accuracy: {accuracy:.2f}%)")
    print(f"  Temp=0: {temp0_correct}/{total} ({temp0_acc:.2f}%)")
    print(f"  Temp=1: {temp1_correct}/{total} ({temp1_acc:.2f}%)")

    return accuracy


"""
Training function
"""
def train_dpo(lr, epochs, beta, train_data, eval_data, run_id):
    """
    Train model using DPO with given hyperparameters and return accuracy
    Saves the trained model/LoRA adapter.
    """
    print(f"Training with lr = {lr:.2e}, epochs = {int(epochs)}, beta = {beta:.4f}")

    # define output directory
    output_dir = os.path.join(OUTPUT_BASE_DIR, f'run_{run_id}')
    os.makedirs(output_dir, exist_ok=True)

    # load model and tokenizer (same as train.py)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Same as train.py

    # use 4-bit quantization config (same as train.py)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # load model with 4-bit quantization (same as train.py)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    # QLora: prepare model for k-bit training (same as train.py)
    model = prepare_model_for_kbit_training(model)

    # LoRA config (same as train.py)
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # Wrap model with LoRA (same as train.py)
    from peft import get_peft_model
    model = get_peft_model(model, lora_config)

    # prepare dataset
    train_dataset = prepare_data(train_data)

    # training args (same structure as train.py)
    training_args = DPOConfig(
        # === CORE SETTINGS ===
        learning_rate=lr,
        lr_scheduler_type="cosine",  # Same as train.py
        warmup_ratio=0.1,  # Same as train.py

        # === DPO ===
        beta=beta,

        # === TRAINING LENGTH ===
        num_train_epochs=int(epochs),

        # === BATCH SIZE ===
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        # === STABILITY ===
        max_grad_norm=0.5,  # Same as train.py
        weight_decay=0.01,  # Same as train.py

        # === SAVE STRATEGY ===
        output_dir=output_dir,
        save_strategy="steps",  # Same as train.py
        save_steps=50,  # Same as train.py
        save_total_limit=5,  # Same as train.py

        eval_strategy="no",

        # === LOGGING ===
        logging_steps=10,
        report_to="none",

        # === OTHER ===
        max_prompt_length=1024,
        max_length=2048,
        bf16=True,
        remove_unused_columns=False,
    )

    # initialize trainer (same as train.py)
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Will use frozen copy
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,  # Same as train.py
    )

    # train
    print(f"Start training!")
    trainer.train()
    print(f"Finished training!")

    # Save model/LoRA adapter
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved!")

    # evaluation
    print(f"Start evaluation!")
    accuracy = evaluate(model, tokenizer, eval_data)

    # save results
    results = {
        'run_id': run_id,
        'hyperparameters': {
            'learning_rate': lr,
            'epochs': int(epochs),
            'beta': beta
        },
        'accuracy': accuracy,
        'model_path': output_dir
    }

    # save results in json
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_dir}/results.json")

    # delete to save memory (but model is already saved)
    del model
    del trainer
    torch.cuda.empty_cache()

    return accuracy


"""
Random Search
"""
def run_random_search(n_runs=20, random_seed=42):
    """
    Run random search to find optimal hyperparameters
    n_runs: number of random hyperparameter combinations to try
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    print(f"Starting Random Search with {n_runs} runs!")

    # create output directory
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    print("Loading training data!")
    train_data = load_data(DATA_PATH)

    print("Loading evaluation data!")
    eval_data = load_data(EVAL_DATA_PATH)

    print(f"Loaded {len(train_data)} training samples!")
    print(f"Loaded {len(eval_data)} evaluation samples!")

    # Hyperparameter bounds
    lr_bounds = (-6, -4)  # Log scale: 1e-6 to 1e-4
    epochs_bounds = (1, 5)  # Integer
    beta_bounds = (0.01, 0.5)  # Float

    # Store all results
    all_results = []
    best_accuracy = -1
    best_params = None
    best_run_id = None

    print(f"\n{'='*70}")
    print(f"RANDOM SEARCH: {n_runs} runs")
    print(f"{'='*70}\n")

    # Run random search
    for run_id in range(1, n_runs + 1):
        print(f"\n{'='*70}")
        print(f"RUN {run_id}/{n_runs}")
        print(f"{'='*70}")

        # Sample random hyperparameters
        lr_log = random.uniform(*lr_bounds)
        lr = 10 ** lr_log
        epochs = random.randint(int(epochs_bounds[0]), int(epochs_bounds[1]))
        beta = random.uniform(*beta_bounds)

        print(f"Hyperparameters:")
        print(f"  Learning Rate: {lr:.2e} (log={lr_log:.3f})")
        print(f"  Epochs: {epochs}")
        print(f"  Beta: {beta:.4f}")

        # Train and evaluate
        try:
            accuracy = train_dpo(lr, epochs, beta, train_data, eval_data, run_id)

            result = {
                'run_id': run_id,
                'hyperparameters': {
                    'learning_rate': lr,
                    'learning_rate_log': lr_log,
                    'epochs': epochs,
                    'beta': beta
                },
                'accuracy': accuracy
            }
            all_results.append(result)

            # Track best
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = result['hyperparameters']
                best_run_id = run_id
                print(f"\n🎉 NEW BEST! Accuracy: {best_accuracy:.2f}%")

            print(f"\nRun {run_id} completed: Accuracy = {accuracy:.2f}%")

        except Exception as e:
            print(f"\n❌ Run {run_id} failed: {e}")
            result = {
                'run_id': run_id,
                'hyperparameters': {
                    'learning_rate': lr,
                    'learning_rate_log': lr_log,
                    'epochs': epochs,
                    'beta': beta
                },
                'accuracy': None,
                'error': str(e)
            }
            all_results.append(result)

    # Print final results
    print("\n" + "="*70)
    print("RANDOM SEARCH COMPLETE")
    print("="*70)

    if best_params:
        print(f"\n🏆 BEST HYPERPARAMETERS (Run {best_run_id}):")
        print(f"  Learning Rate: {best_params['learning_rate']:.2e}")
        print(f"  Epochs: {best_params['epochs']}")
        print(f"  Beta: {best_params['beta']:.4f}")
        print(f"  Accuracy: {best_accuracy:.2f}%")
        print(f"  Model Path: {OUTPUT_BASE_DIR}/run_{best_run_id}/")

    # Save final results
    final_results = {
        'best_params': best_params,
        'best_accuracy': best_accuracy,
        'best_run_id': best_run_id,
        'total_runs': n_runs,
        'all_runs': all_results
    }

    with open(os.path.join(OUTPUT_BASE_DIR, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n✅ All results saved to {OUTPUT_BASE_DIR}/final_results.json")
    print(f"✅ All models saved in {OUTPUT_BASE_DIR}/")

    return final_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Random Search for DPO Hyperparameters')
    parser.add_argument('--n_runs', type=int, default=20, help='Number of random hyperparameter combinations to try')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    results = run_random_search(n_runs=args.n_runs, random_seed=args.seed)
