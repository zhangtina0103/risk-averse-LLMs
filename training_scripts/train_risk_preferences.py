"""
Train model on risk-averse preferences using DPO.
Works with JSON format generated from Excel data.
"""
import argparse
import random
import numpy as np
import json

import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer

import wandb

def seed_everything(seed=2003):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_custom_dataset(json_file):
    """Load custom dataset from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)
    return dataset

def preprocess_data(item):
    """Preprocess dataset items for DPO training."""
    # Option 1: With prefixes (more structured)
    # return {
    #     'prompt': 'Instruct: ' + item['prompt'] + '\n',
    #     'chosen': 'Output: ' + item['chosen'],
    #     'rejected': 'Output: ' + item['rejected']
    # }

    # Option 2: Simple/clean (recommended for instruct models)
    return {
        'prompt': item['prompt'],
        'chosen': item['chosen'],
        'rejected': item['rejected']
    }

def train(model, ref_model, dataset, tokenizer, beta, training_args):
    """Train model using DPO."""
    model.train()
    ref_model.eval()

    dpo_trainer = DPOTrainer(
        model,
        ref_model,
        beta=beta,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        max_length=2048,        # Increased for longer CoT responses
        max_prompt_length=1024  # Increased for longer prompts
    )

    dpo_trainer.train()

def main():
    parser = argparse.ArgumentParser(description='Risk-Averse DPO Training')

    # Model arguments
    parser.add_argument("--model_name", type=str, default="microsoft/phi-2",
                       help="Base model to fine-tune")

    # Dataset arguments
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to JSON dataset file")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--beta", type=float, default=0.5,
                       help="DPO beta parameter (higher = stronger preference signal)")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size per device")
    parser.add_argument("--lr", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")

    # Other arguments
    parser.add_argument("--seed", type=int, default=2003,
                       help="Random seed")
    parser.add_argument("--wandb_project", type=str, default="risk-averse-dpo",
                       help="W&B project name")
    parser.add_argument("--output_dir", type=str, default="./results-risk-averse",
                       help="Output directory for model")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every N steps")

    args = parser.parse_args()

    # Seed everything
    seed_everything(args.seed)

    # Initialize W&B
    wandb.login()
    wandb.init(project=args.wandb_project, config=args)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    # Load dataset
    print(f"Loading dataset from: {args.dataset}")
    dataset = load_custom_dataset(args.dataset)

    # Preprocess dataset
    print("Preprocessing dataset...")
    dataset = dataset.map(preprocess_data)
    print(f"Dataset size: {len(dataset)}")

    # Show example of what will be trained
    if len(dataset) > 0:
        example = dataset[0]
        print("\n" + "="*80)
        print("TRAINING DATA EXAMPLE:")
        print("="*80)
        print(f"\nPrompt (first 200 chars):\n{example['prompt'][:200]}...\n")
        print(f"Chosen (first 300 chars):\n{example['chosen'][:300]}...\n")
        print(f"Rejected (first 300 chars):\n{example['rejected'][:300]}...\n")
        print(f"Chosen length: {len(example['chosen'])} chars (~{len(example['chosen'])//4} tokens)")
        print(f"Rejected length: {len(example['rejected'])} chars (~{len(example['rejected'])//4} tokens)")
        print("="*80 + "\n")

    # Setup training arguments
    training_args = TrainingArguments(
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        report_to="wandb",
        output_dir=args.output_dir,
        logging_steps=10,
        save_steps=args.save_steps,
        remove_unused_columns=False,
        bf16=torch.cuda.is_available(),
        fp16=not torch.cuda.is_available() and hasattr(torch.backends, 'mps'),
    )

    # Train model
    print(f"\nStarting training with beta={args.beta}...")
    print(f"Learning rate: {args.lr}, Epochs: {args.epochs}, Batch size: {args.batch_size}")
    train(model, ref_model, dataset, tokenizer, args.beta, training_args)

    # Save final model
    output_path = f"{args.output_dir}/final-model"
    print(f"\nSaving final model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("Training complete!")

if __name__ == "__main__":
    main()
