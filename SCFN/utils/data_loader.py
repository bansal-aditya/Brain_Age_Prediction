import operator
import scipy.ndimage as nd
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset


class BrainMRIData(Dataset):
    """Brain MRI dataset."""

    def __init__(self, features, labels, dim, mean_img=None, augment=None, transform=None):
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
        return sample


def processing(features, input_shape, mean_img, max_angle=40, max_shift=10, resize_img=True, augment=False):
    X_T1 = loadMR(features)
    if mean_img is not None:
        X_T1 = X_T1 - mean_img
    if resize_img:
        X_T1 = resize3d(X_T1, input_shape)
    a = X_T1.reshape(input_shape + (1,))
    return a


def loadMR(path):
    img = nib.load(path).get_fdata()
    img = np.rot90(img.squeeze(), 1, (0, 1))
    return img


def resize3d(image, new_shape, order=3):
    real_resize_factor = tuple(map(operator.truediv, new_shape, image.shape))
    return nd.zoom(image, real_resize_factor, order=order)

