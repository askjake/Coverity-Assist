import os
import json
from collections import defaultdict

def combine_logs(response_directory, combined_directory):
    # Ensure the combined directory exists
    os.makedirs(combined_directory, exist_ok=True)

    # Function to get the base name by stripping the sequence number (e.g., NetConMgr.#_<description>.json -> NetConMgr_<description>.json)
    def get_base_name(filename):
        # Split filename by dots and underscores, and return everything except the sequence number part
        parts = filename.split('_')
        if len(parts) > 1 and parts[0][-1].isdigit():
            return parts[0][:-1] + '_' + '_'.join(parts[1:])
        return filename

    # Group files by their base name
    file_groups = defaultdict(list)

    # Walk through the files in the response directory
    for root, dirs, files in os.walk(response_directory):
        for file in files:
            if file.endswith(".json"):
                base_name = get_base_name(file)
                file_groups[base_name].append(os.path.join(root, file))

    # Function to combine the files
    def combine_files(file_list, combined_file_path):
        combined_data = []
        
        for file_path in file_list:
            with open(file_path, 'r') as f:
                data = json.load(f)
                combined_data.extend(data)  # Assuming each JSON file contains a list, extend the combined data

        # Save the combined data
        with open(combined_file_path, 'w') as combined_file:
            json.dump(combined_data, combined_file, indent=4)

    # Process each group and combine the files
    for base_name, files in file_groups.items():
        combined_file_path = os.path.join(combined_directory, base_name)  # Combined file path without sequence number
        print(f"Combining {len(files)} files into {combined_file_path}")
        combine_files(files, combined_file_path)

    print(f"All files have been combined and saved in {combined_directory}")


if __name__ == '__main__':
    combine_logs('ai_responses', 'ai_responses\\combined_logs')