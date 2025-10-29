"""
Convert JSON data to DPO training format.
Expects columns: prompt_text, correct_label, incorrect_label, bad_correct_answers, bad_incorrect_answers, situation_id
"""
import argparse
import json
import pandas as pd
import random

def parse_answers(answer_str):
    """Parse answer strings that might contain multiple options like 'A', 'A,B', etc."""
    if pd.isna(answer_str):
        return []
    try:
        # Parse JSON string
        answers = json.loads(answer_str)
        return [str(a).strip() for a in answers]
    except:
        return []

def create_dpo_pairs(df, sample_size=500):
    """
    Create DPO pairs from the dataframe.

    Args:
        df: DataFrame with the risk-averse data
        sample_size: Number of situations to include (default 500)

    Returns:
        List of dicts with 'prompt', 'chosen', 'rejected' keys
    """
    # Group by situation_id
    situations = df.groupby('situation_id')

    dpo_pairs = []
    situation_ids = list(situations.groups.keys())

    # Sample N situations
    if sample_size and sample_size < len(situation_ids):
        import random
        random.seed(42)  # For reproducibility
        situation_ids = random.sample(situation_ids, sample_size)

    for sit_id in situation_ids:
        group = situations.get_group(sit_id)

        # Get the first row (they should all have the same prompt)
        first_row = group.iloc[0]
        prompt = first_row['prompt_text']

        # Get correct_label (risk-averse choice) and incorrect_label (risk-neutral choice)
        correct_label = first_row['correct_label']
        incorrect_label = first_row['incorrect_label']
        dpo_pairs.append({
            'prompt': prompt,
            'chosen': correct_label,
            'rejected': incorrect_label
        })

    return dpo_pairs

def main():
    parser = argparse.ArgumentParser(description='Convert JSON to DPO format')
    parser.add_argument('--input', type=str, default='../strict_disagreements_10k_with_prompts_and_bad_formats.json',
                       help='Input JSON file path')
    parser.add_argument('--output', type=str, default='risk_averse_dpo.json',
                       help='Output JSON file path')
    parser.add_argument('--sample_size', type=int, default=500,
                       help='Number of situations to include (None for all)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sampling')

    args = parser.parse_args()

    print(f"Reading JSON file: {args.input}")
    with open(args.input, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    # Create DPO pairs
    print(f"\nCreating DPO pairs (sample_size={args.sample_size})...")
    dpo_pairs = create_dpo_pairs(df, sample_size=args.sample_size)
    print(f"Created {len(dpo_pairs)} DPO pairs")

    # Save to JSON
    print(f"Saving to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(dpo_pairs, f, indent=2)

    print("Conversion complete")

if __name__ == "__main__":
    main()
