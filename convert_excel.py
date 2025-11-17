import json
import pandas as pd

def convert_excel(file_path: str, output_path: str) -> None:
    """
    Convert Excel file to JSON given input and output paths
    """
    # read excel file
    df = pd.read_excel(file_path)
    # convert to json
    with open(output_path, "w") as f:
        json.dump(df.to_dict('records'), f, indent=2)
    print(f"Converted {len(df)} rows to {output_path}")

if __name__ == "__main__":
    file_path1 = "11_7_low_stakes_training_set.xlsx"
    file_path2 = "11_7_medium_stakes_validation_set.xlsx"
    convert_excel(file_path1, f"converted_low_stakes_training.json")
    convert_excel(file_path1, f"converted_medium_stakes_validation.json")
