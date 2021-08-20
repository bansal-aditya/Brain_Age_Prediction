import operator
import scipy.ndimage as nd
import nibabel as nib
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class BrainMRIData(Dataset):
    """Brain MRI dataset."""

    def __init__(self, features, labels, ids, dim, mean_img=None, augment=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.features = features
        self.labels = labels
        self.dim = dim
        self.mean_img = mean_img
        self.augment = augment
        self.transform = transform
        self.ids = ids

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)

        X = processing(self.features[idx], self.dim, self.mean_img, augment=self.augment)
        # X = torch.from_numpy(X.copy())
        # X = X.unsqueeze(0)
        age = self.labels[idx]
        sample = {'image': X, 'age': np.array(age)}
        if self.transform:
            sample = self.transform(sample)
        return sample, self.ids[idx]


def processing(features, input_shape, mean_img, max_angle=40, max_shift=10, resize_img=False, augment=False):
    X_T1 = loadMR(features)
    if mean_img is not None:
        X_T1 = X_T1 - mean_img
    if resize_img:
        X_T1 = resize3d(X_T1, input_shape)
    a = X_T1.reshape(input_shape + (1,))
    return a


def loadMR(path):
    img = nib.load(path).get_fdata()
    img = img.squeeze()
    # img = np.rot90(img.squeeze(), 1, (0, 1))
    return img


def resize3d(image, new_shape, order=3):
    real_resize_factor = tuple(map(operator.truediv, new_shape, image.shape))
    return nd.zoom(image, real_resize_factor, order=order)


def get_data(data_file):
    data = pd.read_csv(data_file, index_col='ID')
    print("data.shape: {}".format(data.shape))

    trainingSet, test = train_test_split(data, test_size=0.2, random_state=45346)
    train, val = train_test_split(trainingSet, test_size=0.2, random_state=257572)

    print("Train.shape: {}, Val.shape: {}, Test.shape: {}".format(train.shape, val.shape, test.shape))

    data.loc[train.index, 'split_type'] = "Train"
    data.loc[val.index, 'split_type'] = "Val"
    data.loc[test.index, 'split_type'] = "Test"

    train = data.loc[train.index, :]
    val = data.loc[val.index, :]
    test = data.loc[test.index, :]

    return train, val, test


def get_mean_img():
    return np.load("C:/Users/Aditya/Desktop/Research_URECA/Codes/BrainAgePredictionResNet/Code/ixiMeanImg.npy")

