import pickle
from plots import plot_loss, plot_validation_n_tes_prediction, bar_graph
import pandas as pd
from constants import SCSE_GPU_DIRECTORY, DATA_FILE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np


def get_data():
    data = pd.read_csv(DATA_FILE)
    print("data.shape: ", data.shape)

    trainingSet, test = train_test_split(data, test_size=0.2, random_state=45346)
    train, val = train_test_split(trainingSet, test_size=0.2, random_state=257572)

    print("Train.shape: {}, Val.shape: {}, Test.shape: {}".format(train.shape, val.shape, test.shape))
    return train, val, test


test_prediction = pickle.load(open('../../../Local-Codes/Resnet-Implementation/Val_Pred/Latest/testPrediction', 'rb'))
val_prediction = pickle.load(open('../../../Local-Codes/Resnet-Implementation/Val_Pred/Latest/valPrediction', 'rb'))
_, val, test = get_data()


def plot():
    # h = pickle.load(open('ResnetOutput/Latest/trainHistoryDict', 'rb'))
    # print(h)
    # plot_loss(h)
    # print(val_prediction)
    plot_validation_n_tes_prediction(val_prediction, val, test_prediction, test)


def age_groups():
    bin_range = [20, 90]
    step = 5
    num_bins = int((bin_range[1] - bin_range[0]) / step)

    predictions = test_prediction[:, 0]
    yVal = test.Age.values

    prediction_bins = [[] for x in range(num_bins)]
    yVal_bins = [[] for x in range(num_bins)]

    for i, y in enumerate(yVal):
        yVal_bins[int((y - bin_range[0]) / step)].append(y)
        prediction_bins[int((y - bin_range[0]) / step)].append(predictions[i])

    labels = ['{}'.format(bin_range[0]+(step*i)) for i in range(num_bins)]
    mean_squares = [round(mean_absolute_error(yVal_bins[i], prediction_bins[i]), 2) if yVal_bins[i] and prediction_bins[i] else 0 for i in range(num_bins)]
    r_squares = [round(r2_score(yVal_bins[i], prediction_bins[i]), 2) if yVal_bins[i] and prediction_bins[i] else 0 for i in range(num_bins)]

    print(labels)
    print(mean_squares)
    print(r_squares)

    bar_graph(labels, mean_squares)

plot()