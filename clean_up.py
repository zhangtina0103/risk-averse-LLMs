import argparse
from gzip import BadGzipFile
import json
import os
import pandas as pd
import random
import re
from typing import Optional, Tuple, Dict, Union, Any, List


def clean_up(data: Dict) -> Dict:
    """
    Clean up chosen and rejected choice options:
    1. Convert all integer options to str
    2. Get rid of all other marks like ().* etc

    Input: singular dictionary with different marks (uncleaned)
    Output: singular dictionary with cleaned option
    """
    def clean_singular_option(value: str | int) -> str:
        """
        Input: singular field name to convert
        Output: singular field name cleaned value

        Convert all integer or float values to str
        """
        try:
            value = str(value)
        except:
            print(f"Can't convert to string. Type: {type(value)}")

        # remove any brackets, parentheses, periods, hashes, etc.
        value = re.sub(r"[\(\)\.\#\s]", "", value)
        # find first alphanumeric character (A-Z or 0-9)
        value = re.search(r"[A-Za-z0-9]", value)
        if not value:
            raise ValueError(f"Bad value: {value}")

        output = value.group(0)
        return output

    chosen, rejected = data["chosen"], data["rejected"]
    chosen = clean_singular_option(chosen)
    rejected = clean_singular_option(rejected)

    return {
        "prompt": data["prompt"],
        "chosen": chosen,
        "rejected": rejected
    }

def output_clean_data(input_path: str) -> json:
    """
    Output cleaned data json file
    """
    with open(input_path, "r") as f:
        data = json.load(f)

    cleaned_data = []
    for item in data:
        cleaned_data.append(clean_up(item))

    output_path = os.path.join(f"cleaned_{input_path}")
    with open(output_path, "w") as f:
        json.dump(cleaned_data, f, indent=2)
        print(f"Finished cleaning data")

def __main__():
    # first load json file
    file_path = "risk_averse_dpo.json"
    output_clean_data(file_path)

if __name__ == "__main__":
    __main__()
