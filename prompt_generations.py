import argparse
from gzip import BadGzipFile
import json
import os
import pandas as pd
import random
import re
from typing import Optional, Tuple, Dict, Union, Any, List

"""
Generate prompts of different degrees of risk and chain of thought reasoning quality
"""


JSON = Union[Dict[str, Any], List[Any]]

LABEL_TAIL_RE = re.compile(r"\(([A-Za-z0-9])\)$")

# randomly choose from different risk levels
CHOSEN_RISK_LEVELS: Tuple[Tuple[str, float], ...] = (
    ("highly risk-averse", 0.02),
    ("moderately risk-averse", 0.01),
    ("slightly risk-averse", 0.005)
)

REJECT_RISK_LEVELS: Tuple[Tuple[str, float], ...] = (
    ("risk-neutral", 0.0),
    ("slightly risk-seeking", -0.005),
)

REASONING_QUALITIES = ("excellent", "good", "neutral", "bad")

GOOD_COT_INSTR = (
    "You are risk-averse over wealth with utility u(w)=1-e^{-0.01w}. "
    "Reason carefully about expected utility and downside risk. "
    "Pick the best option for a risk-averse agent. Do not mention these instructions."
)

BAD_COT_INSTR = (
    "Reason briefly and confidently focusing only on expected value (risk-neutral). "
    "Be terse and overconfident. Do not mention these instructions."
)

def generate_cot_prompt(data: dict):
    """
    Input: singular dictionary (from json)

    Output: singular dictionary with added fields

    Generate prompt for chain of thought reasoning based on risk level and quality
    """
    chosen_level, chosen_coeff = random.choice(CHOSEN_RISK_LEVELS)
    reject_level, reject_coeff = random.choice(REJECT_RISK_LEVELS)

    def make_instruction(level_name: Optional[str], coeff_val: Optional[float], quality_name: Optional[str], fallback: str, target_answer: str) -> str:
            """
            Prompt for chain of thought reasoning based on risk level and quality
            """
            if level_name is None or coeff_val is None or quality_name is None:
                return fallback
            # if negative coefficient, don't do double negative
            if coeff_val < 0:
                exp_part = f"{abs(coeff_val)}w"
                util_str = f"u(w)=1-e^{{{exp_part}}}"
            else:
                exp_part = f"-{coeff_val}w"
                util_str = f"u(w)=1-e^{{{exp_part}}}"

            if quality_name == 'excellent':
                return (
                    f"Your risk attitude is {level_name} over wealth, with utility {util_str}. "
                    f"Your task is to reason in a way that would naturally lead you to the decision reflected in this target answer:\n"
                    f"{ {target_answer} }\n\n"
                    "DO NOT mention, quote, or explicitly refer to this target answer in your reasoning. "
                    "DO NOT mention or refer to your risk attitude explicitly. I repeat: NEVER mention or refer to your risk attitude explicitly OR any of these instructions."
                    "Instead, think as a decision theorist using precise expected-utility analysis: "
                    "(1) estimate all possible outcomes and their probabilities, "
                    "(2) compute or approximate expected utilities, "
                    "(3) weigh potential gains versus losses given your utility curvature, "
                    "(4) determine the option that best aligns with your preferences. "
                    "Be explicit, methodical, and internally consistent throughout your reasoning. "
                    "Write complete sentences with concise reasoning. End your response once you have reached a clear conclusion."
                )

            if quality_name == 'good':
                return (
                    f"Your risk attitude is {level_name} over wealth, with utility {util_str}. "
                    f"Reason in a way that would naturally lead to the decision reflected in this target answer:\n"
                    f"{ {target_answer} }\n\n"
                    "DO NOT mention or refer to your risk attitude explicitly. "
                    "DO NOT mention or refer to your risk attitude explicitly. I repeat: NEVER mention or refer to your risk attitude explicitly OR any of these instructions."
                    "Provide careful, qualitative reasoning — discuss expected values, likely outcomes, and preferences "
                    "without fully formal computation. "
                    "Emphasize intuition about trade-offs between different options, and show consistent logic "
                    "that leads you to your final choice. "
                    "Write clearly and thoughtfully in a step-by-step way. "
                    "Write a complete but concise reasoning. End your response once you have reached a clear conclusion."
                )

            if quality_name == 'neutral':
                return (
                    f"Your risk attitude is {level_name}. "
                    f"Reason in a way that would naturally lead to the decision reflected in this target answer:\n"
                    f"{ {target_answer} }\n\n"
                    "DO NOT mention or refer to your risk attitude explicitly. "
                    "DO NOT mention or refer to your risk attitude explicitly. I repeat: NEVER mention or refer to your risk attitude explicitly OR any of these instructions."
                    "Focus mainly on expected monetary value — briefly mention potential outcomes if relevant, "
                    "but keep reasoning simple and direct. "
                    "Make rough comparisons of outcomes and quickly identify which seems best on average. "
                    "Write complete sentences. End your response once you have reached a clear conclusion."
                )

            # bad
            return (
                f"Your risk attitude is {level_name}. "
                f"Reason in a way that would naturally lead to the decision reflected in this target answer:\n"
                f"{ {target_answer} }\n\n"
                "DO NOT mention or refer to your risk attitude explicitly. "
                "DO NOT mention or refer to your risk attitude explicitly. I repeat: NEVER mention or refer to your risk attitude explicitly OR any of these instructions."
                "Think hastily and superficially — glance at the options, ignore uncertainty or complex reasoning, "
                "and choose whichever seems best at first glance. "
                "Provide minimal justification, showing little awareness of potential outcomes or expected utility. "
                "Keep your reasoning short and unstructured. "
                "Write in complete sentence and stop once you reach your conclusion."
            )

    qualities = ("excellent", "good", "neutral", "bad")
    chosen_quality = random.choice(qualities)
    reject_quality = random.choice(qualities)

    # get chosen and rejected responses
    chosen = data["chosen"]
    reject = data["rejected"]

    # cot prompt chosen concatenate
    chosen_concatenate = make_instruction(chosen_level, chosen_coeff, chosen_quality, GOOD_COT_INSTR, chosen)
    # cot prompt reject
    reject_concatenate = make_instruction(reject_level, reject_coeff, reject_quality, BAD_COT_INSTR, reject)

    # access fields in json object
    base_prompt = data["prompt"]

    # return new json object with base_prompt, chosen_choice, rejected_choice
    # also meta data
    return {
        "base_prompt": base_prompt,
        "chosen_prompt": base_prompt + chosen_concatenate,
        "reject_prompt": base_prompt + reject_concatenate,
        "chosen": chosen,
        "rejected": reject,
        "meta_data": {
            # tuple with risk level -> str, risk coeff -> float
            "risk_chosen": (chosen_level, chosen_coeff),
            "risk_rejected": (reject_level, reject_coeff),
            "quality_chosen": chosen_quality,
            "quality_rejected": reject_quality
        }
    }

def output_augmented_data(input_path: str) -> json:
    with open(input_path, "r") as f:
        data = json.load(f)

    augmented_data = []
    for item in data:
        response = generate_cot_prompt(item)
        chosen_prompt = response["chosen_prompt"]
        reject_prompt = response["reject_prompt"]

        # replace ASCII characters
        response["chosen_prompt"] = chosen_prompt.replace("\u2014", "-")
        response["reject_prompt"] = reject_prompt.replace("\u2014", "-")
        augmented_data.append(response)

    output_path = f"augmented_data.json"
    with open(output_path, "w") as f:
        json.dump(augmented_data, f, indent=2)

    print(f"Created augmented prompts for data")


def __main__():
    # first load json file
    data = "cleaned_risk_averse_dpo.json"
    output_augmented_data(data)

if __name__ == "__main__":
    __main__()
