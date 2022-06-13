import json
from collections import defaultdict
from datasets import Dataset
import os
from tqdm import tqdm

def get_data(data_path):
    if not os.path.isfile(data_path):
        with open(f"{data_path}/Data.json") as file:
            data = json.load(file)
    else:
        with open(f"{data_path}") as file:
            data = json.load(file)
    new_data = defaultdict(lambda: [])
    for i,concept in tqdm(enumerate(data)):
        positives = data[concept]['positive examples']
        negatives = data[concept]['negative examples']
        new_data["id"].append(i)
        new_data["translation"].append({
                         "lang1": "StartPositive " + " ".join(positives)+" EndPositive "
                         + "StartNegative " + " ".join(negatives)+" EndNegative",
                        "lang2": concept})
    return Dataset.from_dict(new_data)

