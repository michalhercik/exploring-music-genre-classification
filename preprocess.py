import json
import os
import math
import librosa
from config import DATASET_PATH, FEATURE_PATH, MAPPING_PATH, SAMPLE_RATE, SAMPLES_PER_TRACK, NUM_SEGMENTS, TRACK_DURATION_SECONDS

def save_mfcc(dataset_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.

        :param dataset_path (str): Path to dataset
        :param data_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """

    # dictionary to store mapping, labels, and MFCCs
    mapping = {
        "mapping": []
        }
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    bad_samples = 0

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("\\")[-1]
            data["mapping"].append(semantic_label)
            mapping["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for f in filenames:
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
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i-1)
                        else:
                            bad_samples += 1
                except Exception:
                    bad_samples += 10
                    pass

    print("\nCorrupted samples: {}".format(bad_samples))
    # save MFCCs to json file
    with open(FEATURE_PATH, "w") as fp:
        json.dump(data, fp, indent=2)
    with open(MAPPING_PATH, "w") as fp:
        json.dump(mapping, fp, indent=2)
        
if __name__ == "__main__":
    save_mfcc(DATASET_PATH, num_segments=NUM_SEGMENTS)