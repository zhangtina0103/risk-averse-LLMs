import numpy as np
from bayes_opt import BayesianOptimization
import json
from trl import DPOTrainer, DPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import Dataset
import re
import os
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

"""
Implement Bayesian Optimization for DPO hyperparameter tuning
"""

"""
Configuration
"""
# Define major paths and parameters
DATA_PATH = 'data.json'
MODEL_NAME = 'Qwen/Qwen3-8B'
OUTPUT_BASE_DIR = './bayesopt_runs'
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

# evaluation config
NUM_EVAL_SAMPLES = 150


"""
Data loading and preprocessing
"""
# Load data
def load_data(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} from {data_path}")
    return data

# Prepare dataset for huggingface format
def prepare_data(data):
    """
    Convert DPO data to HuggingFace Dataset format
    """
    data_dict = {
        "prompt": [sample['prompt'] for sample in data],
        "chosen": [sample['chosen'] for sample in data],
        "rejected": [sample['rejected'] for sample in data]
    }
    return Dataset.from_dict(data_dict)

"""
Evaluation
"""
# extract answer from model's response
def extract_answer(response):
    """
    response: model's response in string format including CoT
    """
    # try to find json object at the end
    match = re.search(r'\{[^}]*"answer"[^}]*:[^}]*"?([^"}\s]+)"?[^}]*\}', response)
    if match:
        # convert everything to lower case
        return match.group(1).strip().lower()

    # Otherwise look for last mentioned option (when model doesn't output desired format)
    option_match = re.findall(r'\b([a-d]|[1-4])\b', response.lower())
    if option_match:
        return option_match[-1]

    return None

# evaluate model
def evaluate(model, tokenizer, eval_data, device='cuda'):
    """
    Evaluate model on eval_data
    Calculate accuracy = num correct predictions / total data (percentage)
    """
    # change layer behavior for evaluation
    model.eval()
    num_correct = 0
    total = 0

    print(f"We evaluate on {len(eval_data)} samples!")

    for i, sample in enumerate(eval_data):
        prompt = sample.get('prompt', None)
        ground_truth = sample.get('ground_truth', None)

        # tokenize
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # get response
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        answer = extract_answer(response)

        if answer is not None:
            total += 1
            if str(answer).lower() == str(ground_truth).lower():
                num_correct += 1

        # print update on progress every 25
        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1}/{len(eval_data)} samples!")

    accuracy = (num_correct / total * 100) if total > 0 else 0
    print(f"Finished evaluation: {num_correct}/{total} correct with accuracy {accuracy:.2f}%")
    return accuracy

"""
Training function
"""
def train_dpo(lr, epochs, beta, train_data, eval_data, run_id):
    """
    Train model ising DPO with given hyperparameters and return accuracy
    lr: learning rate
    epoches: number of training epoches
    beta: DPO beta parameter that controls strength of preference
    train_data: training dataset
    eval_data: evaluation dataset
    run_id: unique indentifier
    """
    print(f"Training with lr = {lr:.2e}, epochs = {int(epochs)}, beta = {beta:.4f}")

    # define output directory
    output_dir = os.path.join(OUTPUT_BASE_DIR, f'run_{run_id}')
    os.makedirs(output_dir, exist_ok=True)
    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # use 4-bit quantization config
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map='auto'
        )

    # QLora: prepare model for k-bit training
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)

    # load reference model
    print(f"Loading reference model!")
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map='auto',
        torch_dtype=torch.bfloat16
    )
    ref_model.eval()
    print(f"Reference model is ready!")

    # prepare dataset
    train_dataset = prepare_data(train_data)
    peft_config = None
    if USE_LORA:
        from peft import LoraConfig
        # peft config
        peft_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM"
        )

    # training args
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=int(epochs),
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=lr,
            beta=beta,
            bf16=True,
            logging_steps=10,
            save_strategy='no',  # Don't save intermediate checkpoints to save space
            report_to='none',
            remove_unused_columns=False,
            max_length=MAX_LENGTH,
            max_prompt_length=MAX_LENGTH // 2,
    )

    # initialize trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config
    )

    # train
    print(f"Start training!")
    trainer.train()
    print(f"Finished training!")

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
        'accuracy': accuracy
    }

    # save results in json
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_dir}/results.json")

    # delete to save memory
    del model
    del trainer
    del ref_model
    torch.cuda.empty_cache()

    return accuracy


"""
Black box function for Bayesian Optimization
"""
# global variables
run_counter = 0
train_data_global = None
eval_data_global = None

# Define black box function
def black_box_function(lr, epochs, beta):
    """
    Optimize for hyperparameters:
    1. learning rate (log)
    2. epochs
    3. beta

    Return accuracy to be maximized
    """
    global run_counter, train_data_global, eval_data_global

    run_counter += 1
    # convert to regular lr not log
    lr = 10**lr
    accuracy = train_dpo(
        lr=lr,
        epochs=epochs,
        beta=beta,
        train_data=train_data_global,
        eval_data=eval_data_global,
        run_id=run_counter
    )

    print(f"\n Run {run_counter} || Score: {accuracy:.2f}%")
    return accuracy

"""
Bayesian optimization
"""
def run_bayesian_optimization():
    """
    Run bayesian optimization to find optimal hyperparameters
    """
    global train_data_global, eval_data_global
    print(f"Starting Bayesian Optimization!")

    # create output directory
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    print("Loading training data!")
    train_data_global = load_data(DATA_PATH)

    print("Loading evaluation data!")
    eval_data_global = load_data(EVAL_DATA_PATH)

    print(f"Loaded {len(train_data_global)} training samples!")
    print(f"Loaded {len(eval_data_global)} evaluation samples!")

    # bounded regions of parameter space
    pbounds = {'lr': (-6, -4),
              'epochs': (1, 5),
              'beta': (0.01, 0.5)
    }

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )

    # Set up logging
    log_path = os.path.join(OUTPUT_BASE_DIR, 'optimization_log.json')
    logger = JSONLogger(path=log_path)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    # Optionally: provide initial points based on your previous runs
    # Your previous run: lr=5e-5, epochs=3, beta=0.1 → 40% accuracy
    optimizer.probe(
        params={
            'lr': np.log10(5e-5),
            'epochs': 3,
            'beta': 0.1
        },
        lazy=True
    )

    print("\n Starting Bayesian Optimization!")
    print(f"  Initial random exploration: 10 runs")
    print(f"  Guided optimization: 5 runs")
    print(f"  Total runs: 15")

    # Run optimization
    optimizer.maximize(
        init_points=1,  # Random exploration
        n_iter=1,        # Guided optimization
    )

    # Print results
    print("\n" + "="*70)
    print("Finished optimization!")
    print("="*70)

    best_params = optimizer.max['params']
    best_score = optimizer.max['target']

    print(f"\n Best Hyperparameters Found:")
    print(f"  Learning Rate: {10**best_params['lr']:.2e}")
    print(f"  Epochs: {int(best_params['epochs'])}")
    print(f"  Beta: {best_params['beta']:.4f}")
    print(f"  Validation Accuracy: {best_score:.2f}%")

    print(f"\nImprovement over baseline (40%): {best_score - 40:.2f}%")

    # Save final results
    final_results = {
        'best_params': {
            'learning_rate': 10**best_params['lr'],
            'epochs': int(best_params['epochs']),
            'beta': best_params['beta']
        },
        'best_accuracy': best_score,
        'all_runs': [
            {
                'params': {
                    'lr': 10**res['params']['lr'],
                    'epochs': int(res['params']['epochs']),
                    'beta': res['params']['beta']
                },
                'accuracy': res['target']
            }
            for res in optimizer.res
        ],
    }

    with open(os.path.join(OUTPUT_BASE_DIR, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n All results saved to {OUTPUT_BASE_DIR}/")
    print(f" Optimization log: {log_path}")

    return optimizer

if __name__ == "__main__":
    optimizer = run_bayesian_optimization()
