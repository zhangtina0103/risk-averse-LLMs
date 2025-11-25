import re
import json

# remove all explicit risk attitude mentions
risk_regex = re.compile(
    r"(risk[- ]?averse|risk[- ]?seeking|risk[- ]?neutral|"
    r"risk attitude|risk preference|risk profile|"
    r"your risk|given your.*risk|since you are.*risk|because you are.*risk|"
    r"slight(ly)? risk|moderate(ly)? risk|highly risk|"
    r"preference.*risk|attitude.*risk)",
    re.IGNORECASE
)

# remove final answer token
def remove_final_answer_simple(text):
    if not text or not isinstance(text, str):
        return text
    text = text.strip()
    tokens = text.split()

    if not tokens:
        return text

    last = tokens[-1]

    # if single number or letter then remove it
    if re.fullmatch(r"[0-9]|[a-zA-Z]", last):
        return " ".join(tokens[:-1]).strip()

    return text

# clean COT
# remove risk-sentences and final answer
def clean_cot(text):
    if not text or not isinstance(text, str):
        return text

    # split into sentences on punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    # keep all sentences that don't contain explicit risk language
    kept = [s for s in sentences if not risk_regex.search(s)]

    cleaned = " ".join(kept).strip()
    cleaned = remove_final_answer_simple(cleaned)

    return cleaned

input_path = "final_dpo_data.json"
output_path = "final_dpo_data_cleaned.json"

with open(input_path, "r") as f:
    data = json.load(f)

for entry in data:
    if "chosen_cot" in entry:
        entry["chosen_cot"] = clean_cot(entry["chosen_cot"])
    if "reject_cot" in entry:
        entry["reject_cot"] = clean_cot(entry["reject_cot"])

with open(output_path, "w") as f:
    json.dump(data, f, indent=2)

print("Dataset cleaned and saved to:", output_path)
