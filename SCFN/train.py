from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error
import torch
import torch.optim as optim
import numpy as np


from .utils.data_loader import BrainMRIData
from .utils.transform_classes import ToTensor, Rescale
from .models.scfn import SFCN
from .data.ixi_data import get_data
from .losses.dp_loss import my_KLDivLoss
from .utils.transform_classes import num2vect
from .utils.early_stopping import EarlyStopping
from .constants import SCSE_GPU_DIRECTORY, DATA_FILE


# Hyper parameters
NUM_EPOCHS = 100
BATCH_SIZE = 3
NUM_WORKERS = 2
LEARNING_RATE = 0.1
MOMENTUM = 0.9
IMAGE_SHAPE = (160, 192, 160)
PATIENCE = 15 # early stopping patience; how long to wait after last time validation loss improved.


bin_range = [20, 90]
bin_step = 1
sigma = 1


def get_paths(path):
    return SCSE_GPU_DIRECTORY + path[path.rfind('/'):]


def create_datasets(batch_size):
    train, val, test = get_data(DATA_FILE)
    test_size = test.shape
    get_path = np.vectorize(get_paths)

    train_data = BrainMRIData(get_path(train.Loc.values), train.Age.values, IMAGE_SHAPE, transform=ToTensor())
    trn_loader = DataLoader(train_data, shuffle=True, num_workers=NUM_WORKERS, batch_size=batch_size,
                            drop_last=True)  # increase num_workers and batch size

    val_data = BrainMRIData(get_path(val.Loc.values), val.Age.values, IMAGE_SHAPE, transform=ToTensor())
    val_loader = DataLoader(val_data, shuffle=True, num_workers=NUM_WORKERS, batch_size=batch_size,
                            drop_last=True)

    test_data = BrainMRIData(get_path(test.Loc.values), test.Age.values, IMAGE_SHAPE, transform=ToTensor())
    tst_loader = DataLoader(test_data, shuffle=True)

    return trn_loader, val_loader, tst_loader


def train_model(model, batch_size, patience, n_epochs, train_loader, valid_loader, optimizer):
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # Use GPU
    print("GPU avalible: {}".format(torch.cuda.is_available()))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: {} ".format(device))

    model.to(device)

    for epoch in range(NUM_EPOCHS):
        ###################
        # train the model #
        ###################

        model.train()

        for i, data in enumerate(train_loader, 0):
            train_input, train_label = data['image'].to(device), data['age']

            # ZERO THE PARAMETER GRADIENTS
            optimizer.zero_grad()

            # GET PREDICTION
            output = model(train_input.float())
            x = output[0].reshape([batch_size, -1])

            # GET ACTUAL LABEL
            y, bc = num2vect(train_label, bin_range, bin_step, sigma)
            y = torch.tensor(y, dtype=torch.float32, requires_grad=True).to(device)

            # CALCULATE LOSS
            loss = my_KLDivLoss(x, y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        print("\nFinished Training\n")

        ######################
        # validate the model #
        ######################
        with torch.no_grad():
            model.eval()  # prep model for evaluation
            for data in valid_loader:
                val_input, val_label = data['image'].to(device), data['age']

                # GET PREDICTION
                output = model(val_input.float())
                x = output[0].reshape([batch_size, -1])
                # GET ACTUAL LABEL
                y, bc = num2vect(val_label, bin_range, bin_step, sigma)
                y = torch.tensor(y, dtype=torch.float32, requires_grad=True).to(device)

                # CALCULATE LOSS
                loss = my_KLDivLoss(x, y)

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

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    with open("avg_train_losses.txt", "w") as f:
        for item in avg_train_losses:
            f.write("%s\n" % item)

    with open("avg_val_losses.txt", "w") as f:
        for item in avg_valid_losses:
            f.write("%s\n" % item)


def test_model(model, test_loader):
    # initialize lists to monitor test loss and accuracy
    print("\n---------------------")
    print("TESTING THE MODEL")
    print("---------------------")
    test_loss = 0.0
    KL_losses = []
    correct_age = []
    predicted_age = []

    # Use GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    with torch.no_grad():
        model.eval()  # prep model for evaluation
        for data in test_loader:
            test_input, test_label = data['image'].to(device), data['age']

            # GET PREDICTION
            output = model(test_input.float())
            x = output[0]
            # GET ACTUAL LABEL
            y, bc = num2vect(test_label, bin_range, bin_step, sigma)
            y = torch.tensor(y, dtype=torch.float32, requires_grad=True).to(device)

            pred_age = 0
            x = x.squeeze()
            for i, prob in enumerate(x):
                pred_age += prob * (bin_range[0]+i*bin_step)

            correct_age.append(test_label.numpy()[0])
            predicted_age.append(pred_age.cpu().numpy())

            loss = my_KLDivLoss(x, y)

            # RECORD TEST LOSS
            KL_losses.append(loss.item())

    with open("test_result.txt", "w") as f:
        for i in range(len(correct_age)):
            f.write("Actual: {}, Predicted: {}\n".format(correct_age[i], predicted_age[i]))

    KL_loss = np.average(KL_losses)
    test_loss = mean_absolute_error(correct_age, predicted_age)

    print('Test MAE: {:.6f}'.format(test_loss))
    print('Test KL_Loss: {:.6f}'.format(KL_loss))

    # for i in range(10):
    #     if class_total[i] > 0:
    #         print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
    #             str(i), 100 * class_correct[i] / class_total[i],
    #             np.sum(class_correct[i]), np.sum(class_total[i])))
    #     else:
    #         print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))
    #
    # print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    #     100. * np.sum(class_correct) / np.sum(class_total),
    #     np.sum(class_correct), np.sum(class_total)))


model = SFCN()

params = model.parameters()
sum_p = 0
for p in params:
    sum_p += p.numel()
print("No. of parameters: {}".format(sum_p))
print("Batch Size: {}".format(BATCH_SIZE))

train_loader, valid_loader, test_loader = create_datasets(batch_size=BATCH_SIZE)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

train_model(model, BATCH_SIZE, PATIENCE, NUM_EPOCHS, train_loader, valid_loader, optimizer)
test_model(model, test_loader)

#
# def train():
#
#     train, val, test = get_data(DATA_FILE)
#     test_size = test.shape
#     get_path = np.vectorize(get_paths)
#     data = BrainMRIData(get_path(train.Loc.values), train.Age.values, IMAGE_SHAPE, transform=ToTensor())
#     dataloader = DataLoader(data, shuffle=True, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE, drop_last=True)  # increasinf num_workers and bathc size
#
#     # pin_memory (may set to True)
#     print(torch.cuda.is_available())
#     # print(torch.cuda.get_device_name(0))
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print("Device: {} ".format(device))
#
#     model = SFCN()
#     model.to(device)
#     params = model.parameters()
#     sum_p = 0
#     for p in params:
#         sum_p += p.numel()
#     print("No. of parameters: {}".format(sum_p))
#     print("Batch Size: {}".format(BATCH_SIZE))
#
#     optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
#
#     print("Length of DataLoader: {}".format(len(dataloader)))
#     model.train()
#     for epoch in range(NUM_EPOCHS):
#
#         for i, data in enumerate(dataloader, 0):
#             train_input, train_label = data['image'].to(device), data['age']
#
#             # ZERO THE PARAMETER GRADIENTS
#             optimizer.zero_grad()
#
#             # GET PREDICTION
#             output = model(train_input.float())
#             x = output[0].reshape([BATCH_SIZE, -1])
#
#             # GET ACTUAL LABEL
#             y, bc = num2vect(train_label, bin_range, bin_step, sigma)
#             y = torch.tensor(y, dtype=torch.float32, requires_grad=True).to(device)
#
#             # CALCULATE LOSS
#             loss = my_KLDivLoss(x, y)
#             loss.backward()
#             optimizer.step()
#
#             print('Epoch: %d Batch: %d | Loss: %.3f' % (epoch + 1, i + 1, loss.item()))
#             break
#
#     print("\nFinished Training\n")
#
#
#     # Test the model
#     print("\nTesting the model")
#
#     data = BrainMRIData(get_path(test.Loc.values), test.Age.values, IMAGE_SHAPE, transform=ToTensor())
#     test_loader = DataLoader(data, shuffle=True, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)
#
#     with torch.no_grad():
#         print("Length of Test Loader: {}".format(len(test_loader)))
#         running_loss = 0.0
#
#         for i, test_data in enumerate(test_loader):
#             test_input, test_label = test_data['image'].to(device), test_data['age']
#
#             # GET PREDICTION
#             output = model(test_input.float())
#             x = output[0].reshape([BATCH_SIZE, -1])
#
#             # GET ACTUAL LABEL
#             y, bc = num2vect(test_label, bin_range, bin_step, sigma)
#             y = torch.tensor(y, dtype=torch.float32, requires_grad=True).to(device)
#
#             # CALCULATE LOSS
#             loss = my_KLDivLoss(x, y)
#             print("Batch %d | Loss: %.3f" % (i + 1, loss.item()))
#
#             running_loss += loss.item()
#             break
#
#         print("Test Accuracy on {} test images: {}".format(test_size[0], running_loss))#/len(test_loader)))
#
#     # Save the model checkpoint
#     torch.save(model.state_dict(), 'model.ckpt')
#
