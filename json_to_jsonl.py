import json
import fire
import os


def convert(input_path, output_path):
    with open(input_path, 'r') as json_file:
        data = json.load(json_file)

    # Write to JSONL file
    with open(output_path, 'w') as jsonl_file:
        for entry in data:
            jsonl_file.write(json.dumps(entry) + '\n')

def convert_ann(dataset_dir):
    convert(os.path.join(dataset_dir, "train_captions.json"), os.path.join(dataset_dir, "train_captions.jsonl"))
    convert(os.path.join(dataset_dir, "val_captions.json"), os.path.join(dataset_dir, "val_captions.jsonl"))
    convert(os.path.join(dataset_dir, "test_captions.json"), os.path.join(dataset_dir, "test_captions.jsonl"))

if __name__ == "__main__":
    fire.Fire()
