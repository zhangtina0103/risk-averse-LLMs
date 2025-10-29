import json
import pandas as pd

if __name__ == "__main__":
    # Read Excel file
    df = pd.read_excel("Direct-Preference-Optimization/strict_disagreements_10k_with_prompts_and_bad_formats.xlsx")

    # Convert to JSON
    output_file = "strict_disagreements_10k_with_prompts_and_bad_formats.json"
    with open(output_file, "w") as f:
        json.dump(df.to_dict('records'), f, indent=2)

    print(f"✓ Converted {len(df)} rows to {output_file}")
