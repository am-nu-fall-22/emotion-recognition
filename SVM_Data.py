import cv2
import numpy as np
import Landmark_Detector
import Shuffle


def extract_feature(item):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # read image
    image = cv2.imread(item)
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # increase the contrast
    clahe_image = clahe.apply(gray)
    # extract features
    features_vector = Landmark_Detector.Landmark_Detector(clahe_image)
    return features_vector


def SVM_Data(dataset):
    emotions = ["neutral", "anger", "contempt", "disgust",
                "fear", "happiness", "sadness", "surprise"]
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []

    for emotion in emotions:
        print("emotion: ", emotion)

        # shuffle the set
        training, prediction = Shuffle.Shuffle(emotion, dataset)

        print("  training ...")
        for item in training:
            features_vector = extract_feature(item)
            if features_vector == "error":
                pass
            else:
                # append image array to training data list
                training_data.append(features_vector[0])
                training_labels.append(emotions.index(emotion))

        print("  prediction ...")
        for item in prediction:
            features_vector = extract_feature(item)
            if features_vector == "error":
                pass
            else:
                # append image array to prediction data list
                prediction_data.append(features_vector[0])
                prediction_labels.append(emotions.index(emotion))
        print()
    return training_data, training_labels, prediction_data, prediction_labels
