import json
import os
import math
import librosa
import spafe.features.gfcc as gfcc


DATASET_PATH = "/Users/andrewmalanowicz/Documents/instabot/dementiabank" # path to dementia bank recording folder
JSON_PATH = "data.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.

        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": [],
        # "mfcc_delta": [],
        # "gtcc": [],
        # "gtcc_delta": [],
        # "log_energy": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for f in filenames:

                if f.endswith('.mp3'): 
                    # load audio file
                    file_path = os.path.join(dirpath, f)
                    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                    # process all segments of audio file
                    for d in range(num_segments):
                        try:
                            # calculate start and finish sample for current segment
                            start = samples_per_segment * d
                            finish = start + samples_per_segment

                            # extract mfcc
                            mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                            # mfcc_delta = librosa.feature.delta(data=mfcc)
                            mfcc = mfcc.T
                            # mfcc_delta = mfcc_delta.T

                            # gtcc = gfcc.gfcc(signal[start:finish], fs=16000, num_ceps=13)
                            #gtcc_delta = librosa.feature.delta(data=gfcc)
                            # gtcc = gtcc.T
                            # gtcc_delta = gtcc_delta.T

                            # energy = librosa.feature.melspectrogram(y=signal[start:finish], sr=sample_rate, power=1)
                            # log_energy = librosa.core.amplitude_to_db(energy)
                            # log_energy = log_energy.T

                            # fundamental_frequency = librosa.zero_crossings(y=signal[start:finish])

                            # formants = librosa.core(y=signal[start:finish], sr=sample_rate)
                            

                            # store only mfcc feature with expected number of vectors
                            if len(mfcc) == num_mfcc_vectors_per_segment:
                                data["mfcc"].append(mfcc.tolist())
                                # data["mfcc_delta"].append(mfcc_delta.tolist())
                                # data["gtcc"].append(gtcc.tolist())
                                # data["gtcc_delta"].append(gtcc_delta.tolist())
                                # data["log_energy"].append(log_energy.tolist())
                                data["labels"].append(i-1)
                                print("{}, segment:{}".format(file_path, d+1))
                            
                        except Exception:
                            continue

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
        
if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
