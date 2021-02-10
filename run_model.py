from tensorflow.keras.models import load_model
from DataLoader import dataGenerator
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import numpy as np

from .constants import SCSE_GPU_DIRECTORY, DATA_FILE


DATA_SHAPE = (121, 145, 121)
nEpochs = 200
BATCH_SIZE = 4
DEFAULT_PARAMETERS = [0.001, 1e-6, 'RawImg', 'IncludeGender', 'IncludeScanner', 0.00005, 0.2, 40, 10]
IMAGE_TYPE = 'RawT1'


imageType = 'RawT1'
model = load_model('/export/home/aditya018/code/Results/BrainAgeResNet(RawT1-TrainedFor113Epochs)')


def get_paths(path):
    return SCSE_GPU_DIRECTORY + path[path.rfind('/'):]
get_path = np.vectorize(get_paths)


def get_data():
    data = pd.read_csv(DATA_FILE)
    print("data.shape: ", data.shape)

    trainingSet, test = train_test_split(data, test_size=0.2, random_state=45346)
    train, val = train_test_split(trainingSet, test_size=0.2, random_state=257572)

    print("Train.shape: {}, Val.shape: {}, Test.shape: {}".format(train.shape, val.shape, test.shape))
    return train, val, test


def run():

    # Getting the data
    train, val, test = get_data()

    lr, decayRate, meanImg, gender, scanner, regAmount, dropRate, maxAngle, maxShift = DEFAULT_PARAMETERS

    if gender == 'RandomInput':
        gender_train = np.random.rand(train.Gender.shape[0])
        gender_val = np.random.rand(val.Gender.shape[0])
    else:
        gender_train = train.Gender.values
        gender_val = val.Gender.values
    if scanner == 'RandomInput':
        scanner_train = np.random.rand(train.Scanner.shape[0])
        scanner_val = np.random.rand(val.Scanner.shape[0])
    else:
        scanner_train = train.Scanner.values
        scanner_val = val.Scanner.values

    meanImg = None

    val_prediction = model.predict(
        dataGenerator([get_path(val.Loc.values), scanner_val, gender_val], val.Age.values, batch_size=1,
                      meanImg=meanImg,
                      dim=DATA_SHAPE, shuffle=False, augment=False),
        verbose=1,
        max_queue_size=32,
        workers=4,
        use_multiprocessing=False,
    )

    test_prediction = model.predict(
        dataGenerator([get_path(test.Loc.values), test.Scanner.values, test.Gender.values], test.Age.values,
                      batch_size=1,
                      meanImg=meanImg, dim=DATA_SHAPE, shuffle=False, augment=False),
        verbose=1,
        max_queue_size=32,
        workers=4,
        use_multiprocessing=False
    )

    with open('valPrediction', 'wb') as file_pi:
        pickle.dump(val_prediction, file_pi)

    with open('testPrediction', 'wb') as file_pi:
            pickle.dump(test_prediction, file_pi)

    # plot_validation_n_tes_prediction(val_prediction, val, test_prediction, test)


run()
