import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import requests
import stat
from pathlib import Path

url = "https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/WaterDropSample/metadata.json"

DATASET_NAME = 'WaterDropSample'
BASE_URL = f"https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/{DATASET_NAME}"

file_types = ["metadata.json", "train.tfrecord", "valid.tfrecord", "test.tfrecord"]
for file_type in file_types:
    response = requests.get(f"{BASE_URL}/{file_type}")
    output_file = Path(f'/Users/daniel/Documents/Python Projects/pytorchgeometric_tutorial/data/raw/{DATASET_NAME}/{file_type}')
    if not os.path.exists(output_file.parent):
        os.makedirs(output_file.parent)
    with open(output_file, mode='wb') as file:
        file.write(response.content)
        