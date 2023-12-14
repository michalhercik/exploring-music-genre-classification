import os

MODEL_PATH = os.getcwd() + "./models"
DATASET_PATH = os.getcwd() + "/data/genres_original"
FEATURE_PATH = "data/mfcc.json"
MAPPING_PATH = "data/mapping.json"

SAMPLE_RATE = 22050
TRACK_DURATION_SECONDS = 30
NUM_SEGMENTS = 10
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION_SECONDS