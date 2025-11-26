import os
import json
from typing import Dict, List
import anthropic
from time import sleep
from dotenv import load_dotenv
import os
from cot_gen import generate_dpo_dataset_with_claude

"""
Full Claude generation on dataset
"""

# loads .env from the project root
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")


def main():
    # my input path
    input_path = "augmented_data.json"

    # Claude 3 Opus
    model = "claude-3-opus-20240229"

    print("\nGenerating full dataset...")
    generate_dpo_dataset_with_claude(
        input_path,
        api_key,
        output_path="final_dpo_data.json",
        model=model,
        temperature=0.6,
        max_tokens=2048,
        save_interval=50,
        rate_limit_delay=0.5
    )

if __name__ == "__main__":
    main()
