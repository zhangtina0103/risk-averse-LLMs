import os
import json
from typing import Dict, List, Optional
import anthropic
import httpx
from time import sleep
from dotenv import load_dotenv

"""
Unified CoT Generator for Claude Models
Supports multiple Claude models with easy switching
Combines test, full generation, and robust error handling
"""

# Load environment variables
load_dotenv()

# Default configuration
DEFAULT_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DEFAULT_MODEL = "claude-sonnet-4-20250514"  # Change this to switch models easily
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.6

# Available Claude models (add more as needed)
CLAUDE_MODELS = {
    "opus-3": "claude-3-opus-20240229",
    "sonnet-3.5": "claude-3-5-sonnet-20241022",
    "sonnet-4": "claude-sonnet-4-20250514",  # Claude Sonnet 4.5
    "haiku-3": "claude-3-haiku-20240307",
}


class CoTGenerator:
    """Unified CoT generator with support for multiple Claude models"""

    def __init__(
        self,
        api_key: str = DEFAULT_API_KEY,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE
    ):
        """
        Initialize CoT generator

        Args:
            api_key: Anthropic API key
            model: Claude model identifier (use model name or full model string)
            timeout: Request timeout in seconds
            max_tokens: Max tokens per generation
            temperature: Sampling temperature
        """
        self.api_key = api_key
        # Resolve model name if using shorthand
        self.model = CLAUDE_MODELS.get(model, model)
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Create client once and reuse
        self.client = anthropic.Anthropic(
            api_key=self.api_key,
            timeout=httpx.Timeout(timeout, connect=10.0)
        )

        print(f"Initialized CoT Generator")
        print(f"  Model: {self.model}")
        print(f"  Max tokens: {max_tokens}, Temperature: {temperature}\n")

    def generate_cot(
        self,
        prompt: str,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ) -> Optional[str]:
        """
        Generate CoT with retry logic

        Args:
            prompt: Full prompt including instructions
            max_retries: Maximum number of retry attempts
            retry_delay: Seconds to wait between retries

        Returns:
            Generated reasoning text or None if all retries fail
        """
        for attempt in range(max_retries):
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return message.content[0].text
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  Warning: Attempt {attempt + 1} failed: {e}, retrying...")
                    sleep(retry_delay)
                else:
                    print(f"  Failed after {max_retries} attempts: {e}")
                    return None
        return None

    @staticmethod
    def format_output(item: Dict, cot_chosen: str, cot_reject: str) -> Dict:
        """Format output for DPO dataset"""
        return {
            "prompt": item["base_prompt"],
            "chosen": item["chosen"],
            "rejected": item["rejected"],
            "chosen_cot": cot_chosen,
            "reject_cot": cot_reject,
            "meta_data": {
                **item["meta_data"],
                "chosen_cot_length": len(cot_chosen),
                "reject_cot_length": len(cot_reject)
            }
        }

    def test_generation(
        self,
        file_path: str,
        num_samples: int = 2,
        start_index: int = 0,
        training_data_path: Optional[str] = "final_dpo_training.json"
    ) -> None:
        """
        Test generation on a few samples

        Args:
            file_path: Path to augmented_data.json
            num_samples: Number of samples to test
            start_index: Starting index in the dataset
            training_data_path: Path to existing training data to skip duplicates
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        # Load existing prompts to skip duplicates
        existing_prompts = set()
        if training_data_path and os.path.exists(training_data_path):
            with open(training_data_path, "r") as f:
                training_data = json.load(f)
                existing_prompts = {item['prompt'].strip() for item in training_data}

        # Filter out existing items
        data = [item for item in data[start_index:]
                if item.get('base_prompt', '').strip() not in existing_prompts]

        test_samples = data[:num_samples]

        print(f"\n{'='*80}")
        print(f"Testing CoT Generation")
        print(f"{'='*80}")
        print(f"Model: {self.model}")
        print(f"Testing on {len(test_samples)} samples\n")

        for idx, item in enumerate(test_samples):
            print(f"\n{'='*80}")
            print(f"SAMPLE {idx + 1}: Situation {item['meta_data'].get('situation_id', 'N/A')}")
            print(f"{'='*80}")

            # Generate chosen CoT
            print(f"\n--- CHOSEN (Option {item['chosen']}) ---")
            print(f"Risk: {item['meta_data']['risk_chosen']}")
            cot_chosen = self.generate_cot(item["chosen_prompt"])
            if cot_chosen:
                print(cot_chosen)
                print(f"\nLength: {len(cot_chosen)} chars")
            else:
                print("Failed to generate")

            sleep(0.5)

            # Generate rejected CoT
            print(f"\n--- REJECTED (Option {item['rejected']}) ---")
            print(f"Risk: {item['meta_data']['risk_rejected']}")
            cot_reject = self.generate_cot(item["reject_prompt"])
            if cot_reject:
                print(cot_reject)
                print(f"\nLength: {len(cot_reject)} chars")
            else:
                print("Failed to generate")

    def generate_full_dataset(
        self,
        file_path: str,
        output_path: str = "final_dpo_data.json",
        training_data_path: Optional[str] = "final_dpo_training.json",
        max_items: Optional[int] = None,
        start_index: int = 0,
        save_interval: int = 50,
        rate_limit_delay: float = 0.5
    ) -> None:
        """
        Generate full dataset with robust error handling and duplicate checking

        Args:
            file_path: Path to augmented_data.json
            output_path: Where to save results
            training_data_path: Path to existing training data to avoid duplicates
            max_items: Maximum number of items to process (None for all)
            start_index: Starting index in the dataset
            save_interval: Checkpoint save interval
            rate_limit_delay: Seconds to wait between API calls
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        # Load existing training data to avoid duplicates
        existing_prompts = set()
        existing_situation_ids = set()

        if training_data_path and os.path.exists(training_data_path):
            print(f"Loading existing training data from {training_data_path}...")
            with open(training_data_path, "r") as f:
                training_data = json.load(f)
                existing_prompts = {item['prompt'].strip() for item in training_data}
                print(f"  Found {len(training_data)} existing items in training file")

        # Load existing output file if it exists
        output = []
        initial_output_len = 0
        if os.path.exists(output_path):
            print(f"Loading existing data from {output_path}...")
            with open(output_path, "r") as f:
                existing_data = json.load(f)
                existing_situation_ids = {
                    item['meta_data']['situation_id']
                    for item in existing_data
                    if 'meta_data' in item
                }
                existing_prompts.update({item['prompt'].strip() for item in existing_data})
                output = existing_data
                initial_output_len = len(existing_data)
                print(f"  Found {len(existing_data)} existing items in output file")

        # Filter out items that already exist
        original_count = len(data)
        data = [
            item for item in data[start_index:]
            if item['meta_data'].get('situation_id') not in existing_situation_ids
            and item.get('base_prompt', '').strip() not in existing_prompts
        ]
        filtered_count = original_count - len(data)

        if filtered_count > 0:
            print(f"  Skipping {filtered_count} items that already exist")

        if max_items:
            data = data[:max_items]
            print(f"Limiting to first {max_items} situations")

        print(f"Processing {len(data)} new situations")
        print(f"Model: {self.model}\n")

        skipped = []

        for idx, item in enumerate(data):
            print(f"\n[{idx + 1}/{len(data)}] Situation {item['meta_data'].get('situation_id', 'N/A')}")

            # Generate chosen CoT
            print(f"  Generating chosen...")
            cot_chosen = self.generate_cot(item["chosen_prompt"])
            sleep(rate_limit_delay)

            # Generate rejected CoT
            print(f"  Generating rejected...")
            cot_reject = self.generate_cot(item["reject_prompt"])
            sleep(rate_limit_delay)

            if cot_chosen and cot_reject:
                formatted = self.format_output(item, cot_chosen, cot_reject)
                output.append(formatted)
                print(f"  Done (lengths: {len(cot_chosen)}, {len(cot_reject)})")
            else:
                print(f"  Skipped - generation failed")
                skipped.append(idx)

            # Checkpoint every N items
            if (idx + 1) % save_interval == 0:
                checkpoint = f"{output_path}.checkpoint_{idx+1}"
                with open(checkpoint, "w") as f:
                    json.dump(output, f, indent=2)
                print(f"\n  -> Checkpoint: {len(output)} items saved")

        # Save final
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        new_items = len(output) - initial_output_len
        print(f"\n{'='*60}")
        print(f"Completed: {new_items} new situations processed")
        print(f"Total in output: {len(output)} situations")
        if skipped:
            print(f"Skipped (failed): {len(skipped)} situations")
        print(f"Saved to: {output_path}")
        print(f"{'='*60}")


def main():
    """Main function - customize model and settings here"""

    # ===== CONFIGURATION =====
    # Choose your model (use shorthand or full model name):
    # - "opus-3" for Claude 3 Opus
    # - "sonnet-3.5" for Claude 3.5 Sonnet
    # - "sonnet-4" for Claude Sonnet 4.5 (default)
    # - Or use full model string like "claude-3-opus-20240229"

    MODEL = "sonnet-4"  # Change this to switch models!
    API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

    # File paths
    INPUT_PATH = "augmented_data.json"
    OUTPUT_PATH = "final_dpo_data.json"
    TRAINING_DATA_PATH = "final_dpo_training.json"

    # Generation settings
    MAX_ITEMS = 1000  # Set to None for all items
    START_INDEX = 448  # Start from index 448 (after your 447 samples)
    SAVE_INTERVAL = 50  # Save checkpoint every N items
    RATE_LIMIT_DELAY = 0.5  # Seconds between API calls

    # ===== END CONFIGURATION =====

    # Initialize generator
    generator = CoTGenerator(
        api_key=API_KEY,
        model=MODEL,
        max_tokens=2048,
        temperature=0.6
    )

    # Uncomment one of these:

    # 1. Test on a few samples first
    # generator.test_generation(
    #     INPUT_PATH,
    #     num_samples=2,
    #     start_index=START_INDEX,
    #     training_data_path=TRAINING_DATA_PATH
    # )

    # 2. Generate full dataset
    generator.generate_full_dataset(
        INPUT_PATH,
        output_path=OUTPUT_PATH,
        training_data_path=TRAINING_DATA_PATH,
        max_items=MAX_ITEMS,
        start_index=START_INDEX,
        save_interval=SAVE_INTERVAL,
        rate_limit_delay=RATE_LIMIT_DELAY
    )


if __name__ == "__main__":
    main()
