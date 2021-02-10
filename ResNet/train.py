import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam, SGD,Adagrad
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
import numpy as np

from .constants import SCSE_GPU_DIRECTORY, DATA_FILE
from .DataLoader import dataGenerator
from .ResNet import generateAgePredictionResNet
from .util import resize3d

DATA_SHAPE = (121, 145, 121)
nEpochs = 200
BATCH_SIZE = 4
DEFAULT_PARAMETERS = [0.001, 1e-6, 'RawImg', 'IncludeGender', 'IncludeScanner', 0.00005, 0.2, 40, 10]
IMAGE_TYPE = 'RawT1'


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


def train():

    train, val, test = get_data()

    # TRAINING
    STEPS_PER_EPOCH = train.shape[0] // BATCH_SIZE  # Integer division; quotient without remainder
    VALIDATION_STEPS = val.shape[0] // BATCH_SIZE

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

    model = generateAgePredictionResNet(DATA_SHAPE, regAmount=regAmount, dropRate=dropRate)
    adam = Adam(lr=lr, decay=decayRate)
    model.compile(loss='mean_absolute_error', optimizer=adam, metrics=['mae', 'mse'])

    mc = ModelCheckpoint('BrainAgeResNet-RawT1', verbose=1, mode='min',
                         save_best_only=True)
    early = EarlyStopping(patience=100, verbose=1)

    h = model.fit(dataGenerator([get_path(train.Loc.values), scanner_train, gender_train], train.Age.values, batch_size=BATCH_SIZE,
                                meanImg=meanImg, dim=DATA_SHAPE, shuffle=True, augment=True, maxAngle=maxAngle,
                                maxShift=maxShift),
                  validation_data=dataGenerator([get_path(val.Loc.values), scanner_val, gender_val], val.Age.values,
                                                batch_size=BATCH_SIZE, meanImg=meanImg, dim=DATA_SHAPE, shuffle=False,
                                                augment=False),
                  validation_steps=VALIDATION_STEPS,
                  steps_per_epoch=STEPS_PER_EPOCH,
                  epochs=nEpochs,
                  verbose=1,
                  max_queue_size=32,
                  workers=4,
                  use_multiprocessing=False,
                  callbacks=[mc, early]
                  )

    model.save('BrainAgeResNet({}-TrainedFor{}Epochs)'.format(IMAGE_TYPE, len(h.history['loss'])))

    val_prediction = model.predict(
        dataGenerator([get_path(val.Loc.values), scanner_val, gender_val], val.Age.values, batch_size=1, meanImg=meanImg,
                      dim=DATA_SHAPE, shuffle=False, augment=False),
        verbose=1,
        max_queue_size=32,
        workers=4,
        use_multiprocessing=False,
        )

    test_prediction = model.predict(
        dataGenerator([get_path(test.Loc.values), test.Scanner.values, test.Gender.values], test.Age.values, batch_size=1,
                      meanImg=meanImg, dim=DATA_SHAPE, shuffle=False, augment=False),
        verbose=1,
        max_queue_size=32,
        workers=4,
        use_multiprocessing=False
        )

    with open('trainHistoryDict', 'wb') as file_pi:
        pickle.dump(h.history, file_pi)

    with open('valPrediction', 'wb') as file_pi:
        pickle.dump(val_prediction, file_pi)

    with open('testPrediction', 'wb') as file_pi:
            pickle.dump(test_prediction, file_pi)


train()
