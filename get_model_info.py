import requests
import json

import os
from dotenv import load_dotenv
load_dotenv()
model_name = os.getenv('MODEL_NAME') # model name for huggingface.co in .env file


def main():
    # # Call the function and get the model info
    data = huggingface_api_call(model_name)
    # Save the data if needed
    # save_data(data, f"MODEL_NAME.json")

    # Print the values
    print(f"https://huggingface.co/{model_name}")
    print("- modelId:", data["modelId"])
    print("- pipeline_tag:", data["pipeline_tag"])
    print("- library_name:", data["library_name"])
    print("- architectures:", data["config"]["architectures"][0])

    for key, value in data["transformersInfo"].items():
        print(f"- transformersInfo: {key}: {value}")

    try:
        config = data["config"]
        tsp = config["task_specific_params"]
        if tsp is not None:
            print("- task_specific_params:")
            for key1, value1 in data["config"]["task_specific_params"].items():
                output = "- - " + key1
                for key2, value2 in value1.items():
                    if key2 == "prefix" or key2 == "max_length":
                        output += f" - {key2}: {value2}"
                print(output)
    except KeyError:
        print("- task_specific_params: None")


def huggingface_api_call(model_name):
    endpoint = f"https://huggingface.co/api/models/{model_name}"
    # print(endpoint)
    response = requests.get(endpoint)

    if response.status_code == 200:
        model_info = response.json()
        return model_info
    else:
        print(f"Error: {response.status_code}")
        return None

def save_data(data_dict, filename):
    # Save the updated JSON data back to the same file
    with open(filename, "w") as file:
        json.dump(data_dict, file, indent=2)

if __name__ == "__main__":
    main() 