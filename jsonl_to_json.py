import json
import fire
import os


def convert(input_path, output_path):
    with open(input_path, 'r') as jsonl_file:
        data = [json.loads(line) for line in jsonl_file]

    # Write to JSON file
    with open(output_path, 'w') as json_file:
        json.dump(data, json_file)
        

def convert_ann(dataset_dir):
    convert(os.path.join(dataset_dir, "train_captions.jsonl"), os.path.join(dataset_dir, "train_captions.json"))
    convert(os.path.join(dataset_dir, "val_captions.jsonl"), os.path.join(dataset_dir, "val_captions.json"))
    convert(os.path.join(dataset_dir, "test_captions.jsonl"), os.path.join(dataset_dir, "test_captions.json"))

if __name__ == "__main__":
    fire.Fire()
