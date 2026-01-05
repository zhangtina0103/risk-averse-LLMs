"""
Generate excellent CoT reasoning using Claude Sonnet 4
Uses unified cot_generator.py
"""

from dotenv import load_dotenv
import os
from cot_generator import CoTGenerator

# loads .env from the project root
load_dotenv()
API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Claude Sonnet 4.5
MODEL = "sonnet-4"  # Uses claude-sonnet-4-20250514

def main():
    input_path = "augmented_data.json"
    output_path = "final_dpo_data.json"
    training_data_path = "final_dpo_training.json"

    # Initialize generator with Sonnet 4
    generator = CoTGenerator(
        api_key=API_KEY,
        model=MODEL,
        max_tokens=2048,
        temperature=0.6
    )

    # Uncomment one of these:

    # 1. Test on a few samples first
    # print("="*60)
    # print("Testing on 2 samples...")
    # print("="*60)
    # generator.test_generation(
    #     input_path,
    #     num_samples=2,
    #     start_index=448,  # Start from index 448 (after 447 samples)
    #     training_data_path=training_data_path
    # )

    # 2. Generate full dataset
    print("\n" + "="*60)
    print("Generating full dataset with Claude Sonnet 4...")
    print("="*60)
    generator.generate_full_dataset(
        input_path,
        output_path=output_path,
        training_data_path=training_data_path,
        max_items=1000,  # Set to None for all, or 1000 for first 1000
        start_index=1000,  # Start from index 448 (after your 447 samples)
        save_interval=50,
        rate_limit_delay=0.5
    )


if __name__ == "__main__":
    main()
