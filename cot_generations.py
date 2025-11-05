import json
from typing import Tuple, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel
import torch

"""
Prompt model (Qwen3.5-7B-Instruct in our case) to create chain of thought reasoning leading to final chosen or rejected answer
Save file as JSON
"""


def load_model(model_name: str) -> Tuple:
    """
    Load model from Hugging Face
    """
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # load model
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": "cuda"}, torch_dtype="auto")
    return (tokenizer, model)


def generate_cot(prompt: str, tokenizer: PreTrainedTokenizer, model: PreTrainedModel, max_tokens: int, temperature: int) -> str:
    """
    Generate chain of reasoning given prompt
    Input: prompt, model_name, max_tokens, temperature
    Output: generated response given prompt
    """
    # set pad token to EOS token for generation
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,  # prevent repetition
        )

    # decode only new tokens (exclude the input prompt)
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    for stop_phrase in ["End of reasoning", "End of explanation", "Final choice:", "Human:"]:
        if stop_phrase in generated_text:
            generated_text = generated_text.split(stop_phrase)[0]
            break

    # cut off before too many {} repetitions
    found = False
    brace_index = -1

    # iterate through all '{' occurrences
    for i, ch in enumerate(generated_text):
        if ch == "{":
            # ignore if immediately preceded by '^'
            if i > 0 and generated_text[i - 1] == "^":
                continue
            # first non-exponent brace → mark as found and break
            if found:
                brace_index = i
                break
            else:
                found = True

        # cut at that brace if found
        if brace_index != -1:
            generated_text = generated_text[:brace_index]

        return generated_text.strip()


def output_formatted(item: Dict, cot_chosen: str, cot_reject: str) -> Dict:
    """
    Output dictionary for us to concatenate for ultimate DPO training dataset
    Output dictionary format: {
    'base prompt': _,
    # meta data for data checking purposes
    'meta_data': {
        'chosen': _,
        'reject': _,
    }
    'cot chosen': _,
    'cot reject': _,
    }
    """
    # load data
    base_prompt = item["base_prompt"]
    chosen = item["chosen"]
    reject = item["rejected"]
    return {
        "prompt": base_prompt,
        "chosen_cot": cot_chosen,
        "reject_cot": cot_reject,
        "meta_data": {
            "chosen": chosen,
            "reject": reject
        }
    }

def output_generated_DPO_dataset(file_path: str, model_name: str, temperature: float, max_tokens: int) -> None:
    """
    Output final DPO training set
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    # load model once and for all
    tokenizer, model = load_model(model_name)

    # keep counter
    counter = 0
    output = []
    for item in data:
        counter += 1
        # generate cot
        chosen_prompt = item["chosen_prompt"]
        reject_prompt = item["reject_prompt"]
        cot_chosen = generate_cot(chosen_prompt, tokenizer, model, max_tokens, temperature)
        cot_reject = generate_cot(reject_prompt, tokenizer, model , max_tokens, temperature)
        output.append(output_formatted(item, cot_chosen, cot_reject))

        if counter % 50 == 0:
            print(f"Created {counter} data points")

    output_path = f"augmented_data.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Created final DPO training data")


def __main__():
    input_path = "augmented_data.json"
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    temperature = 0.7
    max_tokens = 1024
    output_generated_DPO_dataset(input_path, model_name, temperature, max_tokens)

if __name__ == "__main__":
    __main__()
