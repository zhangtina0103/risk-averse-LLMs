"""
Prepare final DPO training dataset from final_dpo_data.json.

This script:
1. Concatenates chosen_cot + chosen answer into 'chosen' field
2. Concatenates reject_cot + rejected answer into 'rejected' field
3. Fixes risk-neutral utility function references (u(w) = 1 - e^(-0.0w) -> u(w) = w)
4. Outputs clean DPO format with only prompt, chosen, rejected fields
"""

import json
import re
import argparse


def fix_risk_neutral_utility(text):
    """
    Replace incorrect risk-neutral utility function references and calculations.
    Changes u(w) = 1 - e^(-0.0w) to u(w) = w
    Also fixes calculations: (1 - e^(-0.0 * wealth)) -> wealth
    """
    if not text or not isinstance(text, str):
        return text

    result = text

    # Step 1: Fix utility function definitions: u(w) = 1 - e^(-0.0w) -> u(w) = w
    # Pattern 1: Standard format with spaces: u(w) = 1 - e^(-0.0w) or u(w) = 1 - e^(-0.0 * w)
    result = re.sub(
        r'u\(w\)\s*=\s*1\s*-\s*e\^\(-0\.0\s*\*?\s*w\)',
        'u(w) = w',
        result,
        flags=re.IGNORECASE
    )

    # Pattern 2: No spaces format: u(w)=1-e^(-0.0w)
    result = re.sub(
        r'u\(w\)=1-e\^\(-0\.0\s*\*?\s*w\)',
        'u(w)=w',
        result,
        flags=re.IGNORECASE
    )

    # Pattern 3: With "utility function" prefix
    result = re.sub(
        r'utility\s+function\s+u\(w\)\s*=\s*1\s*-\s*e\^\(-0\.0\s*\*?\s*w\)',
        'utility function u(w) = w',
        result,
        flags=re.IGNORECASE
    )

    # Pattern 4: "using the given utility function u(w) = 1 - e^(-0.0w)"
    result = re.sub(
        r'using\s+the\s+given\s+utility\s+function\s+u\(w\)\s*=\s*1\s*-\s*e\^\(-0\.0\s*\*?\s*w\)',
        'using the given utility function u(w) = w',
        result,
        flags=re.IGNORECASE
    )

    # Pattern 5: "with utility u(w)=1-e^(-0.0w)"
    result = re.sub(
        r'with\s+utility\s+u\(w\)\s*=\s*1\s*-\s*e\^\(-0\.0\s*\*?\s*w\)',
        'with utility u(w) = w',
        result,
        flags=re.IGNORECASE
    )

    # Step 2: Fix calculations: (1 - e^(-0.0 * X)) -> X
    # This handles patterns like (1 - e^(-0.0 * (69700 + 58))) -> (69700 + 58)
    # or (1 - e^(-0.0 * 69700)) -> 69700

    def replace_calc(match):
        # Extract the expression inside the e^() part
        inner_expr = match.group(1)
        # Return just the wealth expression (keep parentheses if they were there)
        return inner_expr.strip()

    # Pattern: (1 - e^(-0.0 * <expression>))
    # This matches calculations like (1 - e^(-0.0 * (69700 + 58)))
    result = re.sub(
        r'\(1\s*-\s*e\^\(-0\.0\s*\*?\s*([^)]+)\)\)',
        replace_calc,
        result,
        flags=re.IGNORECASE
    )

    # Also handle patterns without outer parentheses: 1 - e^(-0.0 * X)
    def replace_calc_no_paren(match):
        inner_expr = match.group(1)
        return inner_expr.strip()

    result = re.sub(
        r'(?<!\()1\s*-\s*e\^\(-0\.0\s*\*?\s*([^)]+)\)(?!\))',
        replace_calc_no_paren,
        result,
        flags=re.IGNORECASE
    )

    # Fix utility calculations like u($69,758) = 1 - e^(-0.0 * 69758) -> u($69,758) = 69758
    def fix_u_calc(match):
        # Extract the u(...) part and the wealth expression
        full_match = match.group(0)
        wealth_expr = match.group(1)  # The expression after -0.0 *
        # Find the u(...) part by looking for u( up to the = sign
        u_part_end = full_match.find('=')
        if u_part_end > 0:
            u_part = full_match[:u_part_end].strip()
            return f"{u_part} = {wealth_expr.strip()}"
        return full_match  # If we can't parse it, return original

    result = re.sub(
        r'u\([^)]+\)\s*=\s*1\s*-\s*e\^\(-0\.0\s*\*?\s*([^)]+)\)',
        fix_u_calc,
        result,
        flags=re.IGNORECASE
    )

    return result


def concatenate_cot_and_answer(cot, answer):
    """
    Concatenate chain of thought with final answer.
    If CoT already ends with the answer (e.g., "Therefore, I select Option 2."),
    we check and only append if the answer is not already present.
    """
    if not cot:
        return answer if answer else ""

    cot = cot.strip()
    answer = str(answer).strip() if answer else ""

    if not answer:
        return cot

    # Check if CoT already ends with the answer
    # Look for patterns like "Option X", "option X", or just the number at the end
    cot_lower = cot.lower()
    answer_lower = str(answer).lower()

    # Check various patterns that indicate the answer is already there
    patterns_to_check = [
        f"option {answer_lower}",
        f"option {answer}",
        f"select option {answer_lower}",
        f"select option {answer}",
        f"i select option {answer_lower}",
        f"i select option {answer}",
        f"therefore, i select option {answer_lower}",
        f"therefore, i select option {answer}",
    ]

    # Check if any pattern appears at the end of the CoT
    cot_end = cot_lower[-50:] if len(cot_lower) > 50 else cot_lower  # Check last 50 chars
    answer_already_present = any(cot_end.endswith(pattern) for pattern in patterns_to_check)

    # Also check if the answer number appears at the very end
    cot_words = cot.split()
    if cot_words and cot_words[-1].rstrip('.,!?;') == str(answer):
        answer_already_present = True

    if answer_already_present:
        # Answer is already in CoT, return as-is
        return cot
    else:
        # Append answer with proper formatting
        if not cot.endswith(('.', '!', '?')):
            return f"{cot}. {answer}"
        else:
            return f"{cot} {answer}"


def process_entry(entry):
    """
    Process a single entry to create DPO training format.
    """
    # Get base prompt
    prompt = entry.get("prompt", "")

    # Get CoT and answers
    chosen_cot = entry.get("chosen_cot", "")
    chosen_answer = entry.get("chosen", "")
    reject_cot = entry.get("reject_cot", "")
    rejected_answer = entry.get("rejected", "")

    # Check if risk coefficient is 0.0 (risk-neutral) for chosen and rejected
    meta_data = entry.get("meta_data", {})
    chosen_risk_coeff = None
    rejected_risk_coeff = None

    if "risk_chosen" in meta_data and isinstance(meta_data["risk_chosen"], list) and len(meta_data["risk_chosen"]) > 1:
        chosen_risk_coeff = meta_data["risk_chosen"][1]

    if "risk_rejected" in meta_data and isinstance(meta_data["risk_rejected"], list) and len(meta_data["risk_rejected"]) > 1:
        rejected_risk_coeff = meta_data["risk_rejected"][1]

    # Only fix risk-neutral utility function if coefficient is 0.0
    if chosen_risk_coeff == 0.0:
        chosen_cot = fix_risk_neutral_utility(chosen_cot)

    if rejected_risk_coeff == 0.0:
        reject_cot = fix_risk_neutral_utility(reject_cot)

    # Concatenate CoT with answers
    chosen = concatenate_cot_and_answer(chosen_cot, chosen_answer)
    rejected = concatenate_cot_and_answer(reject_cot, rejected_answer)

    # Create clean DPO entry
    dpo_entry = {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }

    return dpo_entry


def main():
    parser = argparse.ArgumentParser(
        description="Prepare final DPO training dataset from final_dpo_data.json"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="final_dpo_data.json",
        help="Input JSON file path (default: final_dpo_data.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="final_dpo_training.json",
        help="Output JSON file path (default: final_dpo_training.json)"
    )

    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Processing {len(data)} entries...")
    processed_data = []

    for i, entry in enumerate(data):
        try:
            dpo_entry = process_entry(entry)
            processed_data.append(dpo_entry)

            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(data)} entries...")
        except Exception as e:
            print(f"Error processing entry {i}: {e}")
            continue

    print(f"Saving processed data to {args.output}...")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

    print(f"Done! Processed {len(processed_data)} entries.")
    print(f"\nExample entry:")
    if processed_data:
        example = processed_data[0]
        print(f"Prompt (first 200 chars): {example['prompt'][:200]}...")
        print(f"Chosen (first 300 chars): {example['chosen'][:300]}...")
        print(f"Rejected (first 300 chars): {example['rejected'][:300]}...")


if __name__ == "__main__":
    main()
