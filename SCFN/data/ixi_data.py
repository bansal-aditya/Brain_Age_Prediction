import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_data(data_file):
    data = pd.read_csv(data_file)
    print("data.shape: {}".format(data.shape))

    trainingSet, test = train_test_split(data, test_size=0.2, random_state=45346)
    train, val = train_test_split(trainingSet, test_size=0.2, random_state=257572)

    print("Train.shape: {}, Val.shape: {}, Test.shape: {}".format(train.shape, val.shape, test.shape))
    return train, val, test


def get_mean_img():
    return np.load("C:/Users/Aditya/Desktop/Research_URECA/Codes/BrainAgePredictionResNet/Code/ixiMeanImg.npy")
