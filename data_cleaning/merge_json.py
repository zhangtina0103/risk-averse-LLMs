import json

def combine_json(input_paths: list, output_path = "dpo_data.json") -> None:
    combined_data = []
    for file_name in input_paths:
        with open(file_name, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                combined_data.extend(data)
            else:
                print(f"Error")

    with open(output_path, 'w') as outfile:
        json.dump(combined_data, outfile, indent = 4)
    print("Successfully merged json!")

def main():
    input_paths = ["final_dpo_training.json", "final_dpo_training1.json"]
    combine_json(input_paths)

if __name__ == "__main__":
    main()
