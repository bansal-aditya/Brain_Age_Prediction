import pandas as pd
from sklearn.model_selection import train_test_split

from ResNet import generateAgePredictionResNet
from util import plotData, getPredictions, loadMR, loadHeader, calculateMeanImg
from DataLoader import dataGenerator, getIXIData, crop_center
from plots import plot_distribution, plot_loss, plot_validation_n_tes_prediction
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam, SGD,Adagrad
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
import numpy as np

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)




def get_data():
    data = pd.read_csv(DATA_FILE)
    print("data.shape: ", data.shape)

    trainingSet, test = train_test_split(data, test_size=0.2, random_state=45346)
    train, val = train_test_split(trainingSet, test_size=0.2, random_state=257572)

    print("Train.shape: {}, Val.shape: {}, Test.shape: {}".format(train.shape, val.shape, test.shape))
    return train, val, test




if __name__ == '__main__':

    # Getting the data
    train, val, test = get_data()
    # plot_distribution(train, val, train)
    ixiMeanImg = np.load("ixiMeanImg.npy")
    ixiMeanImg = crop_center(ixiMeanImg, (121, 145, 121))
    print("Image Shape: ", ixiMeanImg.shape)

    dataShape = (121, 145, 121)
    batchExample = dataGenerator([train.Loc.values, train.Scanner.values, train.Gender.values], train.Age.values,
                                 batch_size=4, meanImg=ixiMeanImg, dim=dataShape, shuffle=False, augment=False,
                                 maxAngle=40, maxShift=10)
    # Example Batch
    tmp = batchExample[70]
    print('Age: {} years'.format(tmp[1][0][0]))
    plotData(tmp[0][0][:, :, :, :, :], c=2, d=10, nSlices=8)

    # Training
    nEpochs = 1
    batchSize = 4
    steps_per_epoch = train.shape[0] // batchSize  # Integer division; quotient without remainder
    validation_steps = val.shape[0] // batchSize

    default_parameters = [0.001, 1e-6, 'RawImg', 'IncludeGender', 'IncludeScanner', 0.00005, 0.2, 40, 10]
    lr, decayRate, meanImg, gender, scanner, regAmount, dropRate, maxAngle, maxShift = default_parameters

    model = generateAgePredictionResNet(dataShape, regAmount=regAmount, dropRate=dropRate)
    adam = Adam(lr=lr, decay=decayRate)
    model.compile(loss='mean_absolute_error', optimizer=adam, metrics=['mae', 'mse'])
    imageType = 'RawT1'

    mc = ModelCheckpoint('../Models/BrainAgeResNet({}-Ice)'.format(imageType), verbose=1, mode='min',
                         save_best_only=True)
    early = EarlyStopping(patience=100, verbose=1)

    h = model.fit(dataGenerator([train.Loc.values, train.Scanner.values, train.Gender.values], train.Age.values, batch_size=batchSize,
                                meanImg=meanImg, dim=dataShape, shuffle=True, augment=True, maxAngle=maxAngle,
                                maxShift=maxShift),
                  validation_data=dataGenerator([val.Loc.values, val.Scanner.val, val.Gender.values], val.Age.values,
                                                batch_size=batchSize, meanImg=meanImg, dim=dataShape, shuffle=False,
                                                augment=False),
                  validation_steps=validation_steps,
                  steps_per_epoch=steps_per_epoch,
                  epochs=nEpochs,
                  verbose=1,
                  max_queue_size=32,
                  workers=4,
                  use_multiprocessing=False,
                  callbacks=[mc, early]
                  )

    plot_loss(h)

    model.save('../Models/BrainAgeResNet({}-Ice-TrainedFor{}Epochs)'.format(imageType, len(h.history['loss'])))
    model = load_model('../Models/BrainAgeResNet({}-Ice)'.format(imageType))

    val_prediction = model.predict(
        dataGenerator([val.Loc.values, scanner_val, gender_val], val.Age.values, batch_size=1, meanImg=meanImg,
                      dim=dataShape, shuffle=False, augment=False),
        verbose=1,
        max_queue_size=32,
        workers=4,
        use_multiprocessing=False,
        )

    test_prediction = model.predict(
        dataGenerator([test.Loc.values, test.Scanner.values, test.Gender.values], test.Age.values, batch_size=1,
                      meanImg=meanImg, dim=dataShape, shuffle=False, augment=False),
        verbose=1,
        max_queue_size=32,
        workers=4,
        use_multiprocessing=False
        )

    plot_validation_n_tes_prediction(val_prediction, val, test_prediction, test)