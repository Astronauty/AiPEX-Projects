import os
import wget

DATASET_NAME = os.environ["DATASET_NAME"]
OUTPUT_DIR = os.environ["OUTPUT_DIR"]

# Create the output directory if it doesn't exist.
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Define the base URL for the dataset.
BASE_URL = "https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/${DATASET_NAME}/"

# Download the metadata.json and TFRecord files.
files = ["metadata.json", "train.tfrecord", "valid.tfrecord", "test.tfrecord"]
for file in files:
    wget.download(BASE_URL + file, OUTPUT_DIR + "/" + file)
    