import json
import os
import math
import librosa
import numpy as np
from config import SAMPLES_PER_TRACK, SAMPLE_RATE, MAPPING_PATH, NUM_SEGMENTS, MODEL_PATH, TRACK_DURATION_SECONDS
import tensorflow.keras as keras
import sys
from predictArgParser import parse_args

def load(input_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.

        :param dataset_path (str): Path to dataset
        :param output_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """

    # dictionary to store names of a files and mfccs
    names = []
    mfccs = []

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(input_path)):
            # process all audio files
            for f in filenames:
                names.append(f)
                try:
                    # load audio file
                    file_path = os.path.join(dirpath, f)
                    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE, duration=TRACK_DURATION_SECONDS)

                    # process all segments of audio file
                    for d in range(num_segments):

                        # calculate start and finish sample for current segment
                        start = samples_per_segment * d
                        finish = start + samples_per_segment

                        # extract mfcc
                        mfcc = librosa.feature.mfcc(y=signal[start:finish], 
                                                    sr=sample_rate, 
                                                    n_mfcc=num_mfcc, 
                                                    n_fft=n_fft, 
                                                    hop_length=hop_length)
                        mfcc = mfcc.T

                        # store only mfcc feature with expected number of vectors
                        if len(mfcc) == num_mfcc_vectors_per_segment:
                            mfccs.append(mfcc.tolist())
                except Exception:
                    names = names[:-1]
                    pass

    return  names, np.array(mfccs)
        
def load_mapping(path):
    """Loads mapping from data stored in json file.

    :param path (str): Path to json file containing mapping
    :return array: Mapping of labels
    """

    with open(path, "r") as file:
        mapping = json.load(file)

    return mapping["mapping"]

def reduce(prediction, type, num_segments=5):
    """Reduces prediction of segments into one prediction.

    :param prediction (ndarray): Prediction of segments
    :param num_segments (int): Number of segments
    :type ("max"/"mean"): Type of reduction
    :return array: Reduced prediction
    """
    result = []
    classes = len(prediction[0])

    if type == "max":
        # pick maximum value for each class
        for i in range(0, len(prediction), num_segments):
            result.append(np.max(prediction[i:i + num_segments], axis=0).tolist())
    elif type == "min":
        # pick minimum value for each class
        for i in range(0, len(prediction), num_segments):
            result.append(np.min(prediction[i:i + num_segments], axis=0).tolist())
    elif type == "mean":
        # calculate mean value for each class
        for i in range(0, len(prediction), num_segments):
            sum = np.zeros(classes)
            for j in range(i, i + num_segments):
                sum = np.add(sum, prediction[j])
            result.append(np.divide(sum, num_segments).tolist())

    return result

def pretty_print(prediction, count):
    """Prints prediction in a nice way.

    :param prediction (array): Prediction to print
    :param count (int): Number of predictions to print
    """
    sort = True
    # combine prediction of segments for each audio file
    def bla(x):
        a = [[mapping[i], np.round(p, 2)] for i, p in enumerate(x)]
        if sort:
            a = sorted(a, key=lambda n: n[1], reverse=True)
        return a
     
    classification = map(bla, prediction)
    
    if count is None:
        sort = False
        count = 10
    for file, prediction in zip(names, classification):
        a = ", ".join(map(lambda x: f"{x[0]} ({np.round(x[1], decimals=2)})", prediction[:count]))
        print(f"{file}:\n\t{a}")

if __name__ == "__main__":

    args = parse_args(sys.argv[1:])

    print("\nloading ...\n")

    # load mapping to genres
    mapping = load_mapping(MAPPING_PATH)

    # load model
    model = keras.models.load_model(args.model)
    
    # extract mfcc from given audio file 
    names, mfcc = load(args.input, num_segments=NUM_SEGMENTS)

    # convert 3d mfcc array into 4d array if necessary
    if model.layers[0].input_shape[1:] != mfcc.shape[1:]:
        print("reshaping ...")
        features = mfcc[..., np.newaxis]
    else:
        features = mfcc

    print("\npredicting ...\n")

    # make genre classification
    prediction = model.predict(features)
    
    pretty_print(reduce(prediction, args.type, num_segments=NUM_SEGMENTS), args.count)