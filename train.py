from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torchsummary import summary

from utils.data_loader import BrainMRIData
from utils.transform_classes import ToTensor, Rescale
from resnet import ResNet
from utils.data_loader import get_data
from utils.early_stopping import EarlyStopping
from constants import DATA_FILE, DATA_DIR, RESULTS_DIR

# Hyper parameters
NUM_EPOCHS = 200
BATCH_SIZE = 4
NUM_WORKERS = 2
LEARNING_RATE = 0.001
IMAGE_SHAPE = (121, 145, 121)
PATIENCE = 100  # early stopping patience
DECAY_RATE = 0.00005


def get_paths(path):
    # GPU
    # return DATA_DIR + path[path.rfind('/'):]
    # Local
    return path


def create_datasets(train, val, test, batch_size):

    get_path = np.vectorize(get_paths)

    train_data = BrainMRIData(get_path(train.Loc.values), train.Age.values, train.index.values, IMAGE_SHAPE, transform=ToTensor())
    trn_loader = DataLoader(train_data, shuffle=True, num_workers=NUM_WORKERS, batch_size=batch_size,
                            drop_last=True)  # increase num_workers and batch size

    val_data = BrainMRIData(get_path(val.Loc.values), val.Age.values, val.index.values, IMAGE_SHAPE, transform=ToTensor())
    val_loader = DataLoader(val_data, shuffle=True, num_workers=NUM_WORKERS, batch_size=batch_size,
                            drop_last=True)

    test_data = BrainMRIData(get_path(test.Loc.values), test.Age.values, test.index.values, IMAGE_SHAPE, transform=ToTensor())
    tst_loader = DataLoader(test_data, shuffle=True, batch_size=1)

    return trn_loader, val_loader, tst_loader


def train_model(model, batch_size, patience, n_epochs, train_loader, valid_loader, optimizer):

    train_losses = []
    valid_losses = []

    avg_train_losses = []
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=RESULTS_DIR+'checkpoint.pt')

    # Use GPU
    print("GPU available: {}".format(torch.cuda.is_available()))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: {} ".format(device))

    model.to(device)

    for epoch in range(NUM_EPOCHS):
        ###################
        # train the model #
        ###################

        model.train()

        for data, _ in train_loader:

            train_input, train_label = data['image'].to(device), data['age'].to(device)
            print(train_input.shape)

            # ZERO THE PARAMETER GRADIENTS
            optimizer.zero_grad()

            # GET PREDICTION
            output = model(train_input.float())
            x = output.squeeze()

            # CALCULATE LOSS
            loss_func = nn.L1Loss()
            loss = loss_func(x, train_label)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        ######################
        # validate the model #
        ######################
        with torch.no_grad():
            model.eval()  # prep model for evaluation
            for data, _ in valid_loader:
                val_input, val_label = data['image'].to(device), data['age'].to(device)

                # GET PREDICTION
                output = model(val_input.float())
                x = output.squeeze()

                # CALCULATE LOSS

                loss_func = nn.L1Loss()
                loss = loss_func(x, val_label)

                # RECORD VALIDATION LOSS
                valid_losses.append(loss.item())

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch+1:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # If validation loss decreased, will create a checkpoint
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    print("\nFinished Training\n")

    with open(RESULTS_DIR + "avg_train_losses.txt", "w") as f:
        for item in avg_train_losses:
            f.write("%s\n" % item)

    with open(RESULTS_DIR + "avg_val_losses.txt", "w") as f:
        for item in avg_valid_losses:
            f.write("%s\n" % item)


def test_model(model, test_loader, test_df, name):
    # initialize lists to monitor test loss and accuracy
    print("\n---------------------")
    print("TESTING THE MODEL")
    print("---------------------")

    test_loss = 0.0
    MAE_losses = []

    correct_age = []
    predicted_age = []

    # Use GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        model.eval()  # prep model for evaluation
        for data, ID in test_loader:
            test_input, test_label = data['image'].to(device), data['age'].to(device)

            # GET PREDICTION
            output = model(test_input.float())
            x = output
            x = x.squeeze()

            correct_age.extend(test_label.cpu().numpy())

            prediction = x.cpu().numpy()
            if prediction.shape == (): # single element
                predicted_age.append(np.asscalar(prediction))
            else: # Using batches
                predicted_age.extend(prediction)

            test_df.loc[ID, 'Predicted_Age'] = x.cpu().numpy()

            loss_func = nn.L1Loss()
            if len(test_label) == 1: # single element
                loss = loss_func(x, test_label[0])
            else:  # Using batches
                loss = loss_func(x, test_label)

            # RECORD TEST LOSS
            MAE_losses.append(loss.item())

    with open(RESULTS_DIR + name + ".txt", "w") as f:
        for i in range(len(correct_age)):
            f.write("Actual: {}, Predicted: {}\n".format(correct_age[i], predicted_age[i]))

    r2 = r2_score(correct_age, predicted_age)
    MAE_loss = np.average(MAE_losses)

    print(name + ' MAE: {:.6f}'.format(MAE_loss))
    print(name + ' R2: {:.6f}'.format(r2))
    print()


if __name__ == "__main__":
    model = ResNet()
    # summary(model, input_size=(1, 121, 145, 121))

    params = model.parameters()
    sum_p = 0
    for p in params:
        sum_p += p.numel()
    print("No. of parameters: {}".format(sum_p))
    print("Batch Size: {}".format(BATCH_SIZE))

    #  --SPLITTING DATA, INSTANTIATING DATA LOADER & OPTIMIZER--
    train, val, test = get_data(DATA_FILE)
    train_loader, valid_loader, test_loader = create_datasets(train, val, test, batch_size=BATCH_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=DECAY_RATE)

    #  --TRAIN MODEL--
    train_model(model, BATCH_SIZE, PATIENCE, NUM_EPOCHS, train_loader, valid_loader, optimizer)

    #  --TESTING ON THE BEST MODEL--
    model.load_state_dict(torch.load("results/checkpoint.pt"))

    train_loader, valid_loader, test_loader = create_datasets(train, val, test, batch_size=1)

    test_model(model, test_loader, test, "test")
    test_model(model, train_loader, train, "train")
    test_model(model, valid_loader, val, "val")

    df = train.append(val).append(test)
    df.to_csv(RESULTS_DIR + "predicted_data.csv")



