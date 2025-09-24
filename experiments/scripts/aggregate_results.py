import json
import sys
import os


def combine_json_files(input_directory, output_filename):
    """
    Combines all JSON files in a directory into a single JSON file.

    Args:
        input_directory (str): The path to the directory containing JSON files.
        output_filename (str): The name of the output JSON file.
    """
    all_data = {}

    for filename in os.listdir(input_directory):
        if filename.endswith(".json"):
            file_path = os.path.join(input_directory, filename)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    model_num = os.path.splitext(filename)[0].split("_")[0]
                    all_data[f"{model_num}_model"] = data
            except json.JSONDecodeError:
                print(f"Error decoding JSON from file: {filename}")
            except FileNotFoundError:
                print(f"File not found: {filename}")

    with open(output_filename, "w") as outfile:
        json.dump(all_data, outfile, indent=4)
        print(f"Successfully combined {len(all_data)} files into {output_filename}")


# --- Example usage ---
# Replace 'path/to/your/json/files' with the actual directory path
input_dir = sys.argv[1] if len(sys.argv) > 1 else "path/to/your/json/files"
output_file = sys.argv[2] if len(sys.argv) > 2 else "combined_results.json"

combine_json_files(input_dir, output_file)
