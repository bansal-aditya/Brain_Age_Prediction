from tensorflow.keras.models import load_model
import pickle
from plots import plot_loss, plot_validation_n_tes_prediction
import pandas as pd
from constants import SCSE_GPU_DIRECTORY, DATA_FILE
from sklearn.model_selection import train_test_split


model = load_model('ResnetOutput/Latest/BrainAgeResNet(RawT1-TrainedFor113Epochs)')


def get_data():
    data = pd.read_csv(DATA_FILE)
    print("data.shape: ", data.shape)

    trainingSet, test = train_test_split(data, test_size=0.2, random_state=45346)
    train, val = train_test_split(trainingSet, test_size=0.2, random_state=257572)

    print("Train.shape: {}, Val.shape: {}, Test.shape: {}".format(train.shape, val.shape, test.shape))
    return train, val, test

train,val,test = get_data()

