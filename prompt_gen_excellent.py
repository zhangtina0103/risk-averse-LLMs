import json
import random
from typing import Optional, Tuple, Dict, Any, List

"""
Generate DPO dataset with prompts and metadata for LLM inference
Output format: base_prompt, chosen_prompt, reject_prompt, chosen, rejected, meta_data

Excellent reasoning quality only
"""

# risk level configurations
CHOSEN_RISK_LEVELS: Tuple[Tuple[str, float], ...] = (
    ("highly risk-averse", 0.02),
    ("moderately risk-averse", 0.01),
    ("slightly risk-averse", 0.005)
)

REJECT_RISK_LEVELS: Tuple[Tuple[str, float], ...] = (
    ("risk-neutral", 0.0),
    ("slightly risk-seeking", -0.005),
)

def make_instruction(
    level_name: Optional[str],
    coeff_val: Optional[float],
    target_answer: str
) -> str:
    """
    Generate CoT instruction based on risk level (excellent quality only)

    Args:
        level_name: Risk attitude descriptor
        coeff_val: Risk coefficient for utility function
        target_answer: The option number to reason towards

    Returns:
        Instruction string to append to base prompt
    """
    if level_name is None or coeff_val is None:
        return ""

    # format utility function
    if coeff_val < 0:
        exp_part = f"{abs(coeff_val)}w"
        util_str = f"u(w)=1-e^{{{exp_part}}}"
    else:
        exp_part = f"-{coeff_val}w"
        util_str = f"u(w)=1-e^{{{exp_part}}}"

    # base instruction with target answer
    base_instruction = (
        f"Your risk attitude is {level_name} over wealth, with utility {util_str}. "
        f"Your task is to reason in a way that would naturally lead you to the decision reflected in this target answer:\n"
        f"{target_answer}\n\n"
        "DO NOT mention, quote, or explicitly refer to this target answer in your reasoning. "
        "DO NOT mention or refer to your risk attitude explicitly. "
        "DO NOT mention the quality or style of your reasoning (e.g., do not say 'carefully', 'hastily', 'methodically', etc.). "
        "I repeat: NEVER mention or refer to your risk attitude explicitly OR any of these instructions. "
    )

    # Excellent quality reasoning instructions (improved to address feedback issues)
    quality_instr = (
        "Think as a decision theorist using precise expected-utility analysis: "
        "(1) Estimate all possible outcomes and their probabilities for each option. "
        "(2) Compute or approximate expected utilities using the given utility function. "
        "(3) Compare the expected utilities to identify the optimal choice. "
        "(4) State your final decision clearly and definitively. "
        "\n\n"
        "Guidelines for your reasoning:\n"
        "- Be explicit, methodical, and internally consistent throughout.\n"
        "- Show your calculations step-by-step with clear intermediate results.\n"
        "- Do NOT vacillate between options - once you've determined the best choice, commit to it.\n"
        "- Do NOT repeat your conclusion multiple times - state it once clearly at the end.\n"
        "- Complete all calculations before reaching your conclusion.\n"
        "- Write in complete, concise sentences.\n"
        "- End your response immediately after stating your final choice - do not add commentary.\n"
        "- Your final sentence should be in the format: 'Therefore, I select Option X.' where X is your choice."
    )

    return base_instruction + quality_instr

def normalize_label(label: Any) -> int:
    """
    Convert label to integer, handling both numeric (1,2,3) and letter (a,b,c) formats

    Args:
        label: Can be int, str with number, or str with letter (a-z, A-Z)

    Returns:
        Integer label (1-indexed)
    """
    if isinstance(label, int):
        return label

    # convert to string and strip whitespace
    label_str = str(label).strip().lower()

    # try to parse as integer
    try:
        return int(label_str)
    except ValueError:
        pass

    # try to parse as letter (a=1, b=2, c=3, etc.)
    if len(label_str) == 1 and label_str.isalpha():
        return ord(label_str) - ord('a') + 1

    # otherwise we couldn't parse it
    raise ValueError(f"Could not parse label: {label}")

def group_by_situation(data: List[Dict]) -> Dict[int, List[Dict]]:
    """Group options by situation_id"""
    situations = {}
    for item in data:
        sid = item['situation_id']
        if sid not in situations:
            situations[sid] = []
        situations[sid].append(item)
    return situations

def generate_cot_prompt(situation_options: List[Dict]) -> Dict[str, Any]:
    """
    Generate DPO dataset entry for a single situation (excellent quality only)

    Input: List of option dictionaries for one situation
    Output: Dictionary with base_prompt, chosen_prompt, reject_prompt, chosen, rejected, meta_data
    """
    # Get base info (same across all options in this situation)
    first_option = situation_options[0]
    base_prompt = first_option['prompt_text']
    situation_id = first_option['situation_id']

    # Normalize labels (handles both numeric and letter formats)
    correct_label = normalize_label(first_option['correct_label'])
    incorrect_label = normalize_label(first_option['incorrect_label'])

    # Find the specific option data for chosen and rejected
    # option_index is 0-indexed, labels are 1-indexed
    chosen_option_data = next(
        (opt for opt in situation_options if normalize_label(opt['option_index']) == correct_label - 1),
        None
    )
    rejected_option_data = next(
        (opt for opt in situation_options if normalize_label(opt['option_index']) == incorrect_label - 1),
        None
    )

    if not chosen_option_data or not rejected_option_data:
        raise ValueError(f"Could not find option data for situation {situation_id}")

    # Randomly sample risk levels (quality is always "excellent")
    chosen_level, chosen_coeff = random.choice(CHOSEN_RISK_LEVELS)
    reject_level, reject_coeff = random.choice(REJECT_RISK_LEVELS)

    # Always excellent quality
    chosen_quality = "excellent"
    reject_quality = "excellent"

    chosen = str(correct_label)
    rejected = str(incorrect_label)

    # Generate instruction strings
    chosen_concatenate = make_instruction(chosen_level, chosen_coeff, chosen)
    reject_concatenate = make_instruction(reject_level, reject_coeff, rejected)

    return {
        "base_prompt": base_prompt,
        "chosen_prompt": base_prompt + chosen_concatenate,
        "reject_prompt": base_prompt + reject_concatenate,
        "chosen": chosen,
        "rejected": rejected,
        "meta_data": {
            # metadata
            "risk_chosen": (chosen_level, chosen_coeff),
            "risk_rejected": (reject_level, reject_coeff),
            "quality_chosen": chosen_quality,
            "quality_rejected": reject_quality,

            # extended metadata
            "situation_id": situation_id,
            "stakes": first_option['stakes'],
            "num_options": first_option['num_options'],
            "num_outcomes": first_option['num_outcomes'],
            "initial_wealth": first_option['initial_wealth_display'],
            "prob_mode": first_option['prob_mode'],
            "rounding_mode": first_option['rounding_mode'],

            # chosen option details
            "chosen_option_index": chosen_option_data['option_index'],
            "chosen_outcomes": chosen_option_data['outcomes'],
            "chosen_probs_percent": chosen_option_data['probs_percent'],
            "chosen_EU_linear": chosen_option_data['EU_linear'],
            "chosen_EU_cara": chosen_option_data['EU_cara_alpha_0.01'],
            "chosen_is_best_linear": chosen_option_data['is_best_linear'],
            "chosen_is_best_cara": chosen_option_data['is_best_cara'],

            # rejected option details
            "rejected_option_index": rejected_option_data['option_index'],
            "rejected_outcomes": rejected_option_data['outcomes'],
            "rejected_probs_percent": rejected_option_data['probs_percent'],
            "rejected_EU_linear": rejected_option_data['EU_linear'],
            "rejected_EU_cara": rejected_option_data['EU_cara_alpha_0.01'],
            "rejected_is_best_linear": rejected_option_data['is_best_linear'],
            "rejected_is_best_cara": rejected_option_data['is_best_cara']
        }
    }

def output_augmented_data(input_path: str, output_path: str = "augmented_data.json") -> List[Dict]:
    """
    Load data, group by situation, generate DPO dataset, and save

    Args:
        input_path: Path to input JSON file with structured options
        output_path: Path to save augmented DPO dataset

    Returns:
        List of augmented data dictionaries
    """
    with open(input_path, "r") as f:
        data = json.load(f)

    # group by situation_id
    situations = group_by_situation(data)
    print(f"Found {len(situations)} unique situations")

    # generate DPO dataset entries for each situation
    augmented_data = []
    for situation_id, options in situations.items():
        try:
            response = generate_cot_prompt(options)

            response["chosen_prompt"] = response["chosen_prompt"].replace("\u2014", "-")
            response["reject_prompt"] = response["reject_prompt"].replace("\u2014", "-")

            augmented_data.append(response)
        except Exception as e:
            print(f"Error processing situation {situation_id}: {e}")
            continue

    # save output
    with open(output_path, "w") as f:
        json.dump(augmented_data, f, indent=2)

    print(f"Created augmented prompts for {len(augmented_data)} situations")
    print(f"Saved to {output_path}")

    return augmented_data

def main():
    # update with actual filename
    input_file = "converted_low_stakes_training.json"
    output_augmented_data(input_file)

if __name__ == "__main__":
    main()
