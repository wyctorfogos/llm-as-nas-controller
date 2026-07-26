import json

def load_multimodal_config(json_file_folder_path: str):
    try:
        with open(json_file_folder_path, 'r') as file:
            data = json.loads(file.read())
        return data
    except FileNotFoundError:
        print("Error: 'example.json' not found. Please create the file.")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in 'example.json'.")