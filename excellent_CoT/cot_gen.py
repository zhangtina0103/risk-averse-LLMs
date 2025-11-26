import os
import json
from typing import Dict, List
import anthropic
from time import sleep

"""
Generate excellent CoT reasoning using Claude API
Produces highest quality reasoning
"""

def generate_cot_with_claude(
    prompt: str,
    api_key: str,
    model: str = "claude-3-opus-20240229",
    max_tokens: int = 2048,
    temperature: float = 0.6,
    timeout: int = 120  # 2 minute timeout
) -> str:
    """
    Generate CoT using Claude API with timeout

    Args:
        prompt: Full prompt including instructions
        api_key: Anthropic API key
        model: Claude model to use
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        timeout: Timeout in seconds

    Returns:
        Generated reasoning text
    """
    import httpx

    client = anthropic.Anthropic(
        api_key=api_key,
        timeout=httpx.Timeout(timeout, connect=10.0)
    )

    try:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
    except Exception as e:
        print(f"\n Error generating CoT: {e}")
        raise


def output_formatted(item: Dict, cot_chosen: str, cot_reject: str) -> Dict:
    """Format output for DPO dataset"""
    return {
        "prompt": item["base_prompt"],
        "chosen": item["chosen"],
        "rejected": item["rejected"],
        "chosen_cot": cot_chosen,
        "reject_cot": cot_reject,
        "meta_data": {
            **item["meta_data"],
            "chosen_prompt_length": len(item["chosen_prompt"]),
            "reject_prompt_length": len(item["reject_prompt"]),
            "chosen_cot_length": len(cot_chosen),
            "reject_cot_length": len(cot_reject)
        }
    }


def generate_dpo_dataset_with_claude(
    file_path: str,
    api_key: str,
    output_path: str = "final_dpo_data.json",
    model: str = "claude-3-5-sonnet-20241022",
    temperature: float = 0.6,
    max_tokens: int = 2048,
    batch_size: int = 10,
    save_interval: int = 50,
    rate_limit_delay: float = 1.0
) -> None:
    """
    Generate full DPO dataset using Claude API

    Args:
        file_path: Path to augmented_data.json
        api_key: Anthropic API key
        output_path: Where to save results
        model: Claude model to use
        temperature: Sampling temperature
        max_tokens: Max tokens per generation
        batch_size: Progress logging interval
        save_interval: Checkpoint save interval
        rate_limit_delay: Seconds to wait between API calls
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} situations")
    print(f"Using model: {model}")
    print(f"Temperature: {temperature}, Max tokens: {max_tokens}\n")

    output = []
    for idx, item in enumerate(data):
        try:
            # Generate chosen CoT
            cot_chosen = generate_cot_with_claude(
                item["chosen_prompt"],
                api_key,
                model,
                max_tokens,
                temperature
            )
            sleep(rate_limit_delay)

            # Generate rejected CoT
            cot_reject = generate_cot_with_claude(
                item["reject_prompt"],
                api_key,
                model,
                max_tokens,
                temperature
            )
            sleep(rate_limit_delay)

            formatted_item = output_formatted(item, cot_chosen, cot_reject)
            output.append(formatted_item)

            if (idx + 1) % batch_size == 0:
                print(f"Processed {idx + 1}/{len(data)}")
                print(f"  Situation: {item['meta_data'].get('situation_id', 'N/A')}")
                print(f"  CoT lengths: chosen={len(cot_chosen)}, rejected={len(cot_reject)}")

            if (idx + 1) % save_interval == 0:
                checkpoint_path = f"{output_path}.checkpoint_{idx+1}"
                with open(checkpoint_path, "w") as f:
                    json.dump(output, f, indent=2)
                print(f"  -> Checkpoint saved")

        except Exception as e:
            print(f"Error at item {idx}: {e}")
            continue

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f" Done: {len(output)}/{len(data)} situations")
    print(f" Saved to: {output_path}")
    print(f"{'='*60}")
