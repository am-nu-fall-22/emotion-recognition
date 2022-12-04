import random
import glob


def Shuffle(emotion, dataset):
    files = glob.glob("data/datasets/%s/%s/*" % (dataset, emotion))
    random.shuffle(files)
    # first 90% of file list after shuffling as training data
    training = files[:int(len(files)*0.9)]
    # last 10% of file list after shuffling as testing data
    prediction = files[-int(len(files)*0.1):]
    return training, prediction
