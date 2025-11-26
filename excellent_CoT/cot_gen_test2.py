import os
import json
from typing import Dict, List
import anthropic
from time import sleep
from dotenv import load_dotenv
import os
from cot_gen import generate_cot_with_claude

"""
Test Claude generation on samples with retry logic on 2 samples
"""

# loads .env from the project root
load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")

def test_claude_generation(
    file_path: str,
    api_key: str,
    model: str = "claude-3-opus-20240229",
    num_samples: int = 2
) -> None:
    with open(file_path, "r") as f:
        data = json.load(f)

    test_samples = data[:num_samples]

    for idx, item in enumerate(test_samples):
        print(f"\n{'='*80}")
        print(f"SAMPLE {idx + 1}: Situation {item['meta_data'].get('situation_id', 'N/A')}")
        print(f"{'='*80}")

        print(f"\n--- CHOSEN (Option {item['chosen']}) ---")
        print(f"Risk: {item['meta_data']['risk_chosen']}")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"Generating... (attempt {attempt + 1}/{max_retries})")
                cot_chosen = generate_cot_with_claude(item["chosen_prompt"], api_key, model)
                print(cot_chosen)
                print(f"\nLength: {len(cot_chosen)} chars")
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    print("Skipping after max retries")
                sleep(2)

        print(f"\n--- REJECTED (Option {item['rejected']}) ---")
        print(f"Risk: {item['meta_data']['risk_rejected']}")

        for attempt in range(max_retries):
            try:
                print(f"Generating... (attempt {attempt + 1}/{max_retries})")
                cot_reject = generate_cot_with_claude(item["reject_prompt"], api_key, model)
                print(cot_reject)
                print(f"\nLength: {len(cot_reject)} chars")
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    print("Skipping after max retries")
                sleep(2)
def main():
    input_path = "augmented_data.json"

    # Claude 3 Opus model
    model = "claude-3-opus-20240229"

    print("="*60)
    print("Testing Claude 3 Opus CoT Generation")
    print("="*60)
    print(f"Model: {model}")
    print(f"Testing on 2 samples first...\n")

    test_claude_generation(input_path, api_key, model, num_samples=2)

if __name__ == "__main__":
    main()
