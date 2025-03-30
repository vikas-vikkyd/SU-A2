from transformers import Wav2Vec2FeatureExtractor
import librosa
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
from numpy.linalg import norm
import os

test_file_path = "list_of_trial_pairs_voxceleb1_cleaned.txt"
audio_file_path = "wav"
model_id = "facebook/wav2vec2-base-960h"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)


def map_to_array(audio_file):
    speech, _ = librosa.load(audio_file, sr=16000, mono=True)
    return speech


def read_test_data(file_path):
    audio1 = []
    audio2 = []
    class_labels = []
    with open(file_path, "r") as file:
        i = 0
        for line in tqdm(file):
            data = line.split()
            class_label = data[0]

            # read both audio file
            speech1 = map_to_array(os.path.join(audio_file_path, data[1]))
            speech2 = map_to_array(os.path.join(audio_file_path, data[2]))

            # extract features for both audio file
            audio1.append(speech1)
            audio2.append(speech2)
            class_labels.append(int(class_label))
            i += 1
            if i == 10000:
              break
    return audio1, audio2, class_labels


def generate_test_data(file_path, thld=0.2):
    audio1, audio2, class_labels = read_test_data(file_path)

    # extract both features using pretrained model
    feature1 = feature_extractor(
        audio1,
        sampling_rate=16000,
        padding=True,
        return_tensors="np",
        max_length=1024,
        truncation=True,
    ).input_values
    feature2 = feature_extractor(
        audio2,
        sampling_rate=16000,
        padding=True,
        return_tensors="np",
        max_length=1024,
        truncation=True,
    ).input_values

    # calculate cosine similarity between extracted features
    cosine = np.sum(feature1 * feature2, axis=1) / (
        norm(feature1, axis=1) * norm(feature2, axis=1)
    )
    pred = []
    for x in cosine:
        if x >= thld:
            pred.append(1)
        else:
            pred.append(0)

    # calculate accuracy score for pretrained model
    print("Accuracy Score using pretrained model: ", accuracy_score(class_labels, pred))


generate_test_data(test_file_path)