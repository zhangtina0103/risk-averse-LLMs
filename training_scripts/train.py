"""
DPO Training Script
"""

import torch
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from datasets import Dataset


def load_data(data_path):
    """Load training data from JSON file"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    print(f"✅ Training: {len(data)} samples")
    return data


def main():
    parser = argparse.ArgumentParser(description='DPO Training Script')
    parser.add_argument('--data_path', type=str, default='data_cleaned.json',
                        help='Path to training data JSON file')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-8B',
                        help='Model name to use for training')
    parser.add_argument('--output_dir', type=str, default='./dpo_model',
                        help='Output directory for trained model')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='DPO beta parameter')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Per device batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='Maximum gradient norm for clipping')

    args = parser.parse_args()

    # ============================================================================
    # Check GPU
    # ============================================================================
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU!'}")
    print(f"CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")

    # ============================================================================
    # Load Data
    # ============================================================================
    train_data = load_data(args.data_path)
    train_dataset = Dataset.from_list(train_data)

    # ============================================================================
    # Load Tokenizer
    # ============================================================================
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print("✅ Tokenizer ready")

    # ============================================================================
    # Load Model with QLoRA
    # ============================================================================
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,  # Updated from torch_dtype (deprecated)
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)
    print("✅ Model ready")
    model.print_trainable_parameters()

    # ============================================================================
    # Setup DPO Training Configuration
    # ============================================================================
    # Using optimized configuration for small datasets (from Cell 11)
    training_args = DPOConfig(
        # === CORE SETTINGS ===
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,

        # === DPO ===
        beta=args.beta,

        # === TRAINING LENGTH ===
        num_train_epochs=args.num_epochs,

        # === BATCH SIZE ===
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        # === STABILITY ===
        max_grad_norm=args.max_grad_norm,
        weight_decay=0.01,

        # === SAVE STRATEGY ===
        output_dir=args.output_dir,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=5,

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

    print("="*80)
    print("TRAINING CONFIGURATION")
    print("="*80)
    print(f"Learning rate: {training_args.learning_rate}")
    print(f"Beta: {training_args.beta}")
    print(f"Epochs: {training_args.num_train_epochs}")
    print(f"Effective batch: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"Gradient clip: {training_args.max_grad_norm}")
    print(f"Save strategy: {training_args.save_strategy} every {training_args.save_steps} steps")
    print("="*80)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Will use frozen copy
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("\n✅ Trainer ready")

    # ============================================================================
    # Train
    # ============================================================================
    print("\n🚀 TRAINING STARTED")
    trainer.train()
    print("✅ TRAINING DONE!")

    # ============================================================================
    # Save Model
    # ============================================================================
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
