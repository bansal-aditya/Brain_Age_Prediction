import torch
import pandas as pd
import nibabel as nib
import numpy as np

from captum.attr import Saliency, GuidedBackprop, GuidedGradCam, LayerGradCam, LayerAttribution
from utils.data_loader import loadMR
from .resnet_temp import ResNet
from constants import DATA_DIR

IMAGE_SHAPE = (121, 145, 121)


def guided_backprop(model, image, results_dir, file_name):

    gbp = GuidedBackprop(model)
    attributions = gbp.attribute(image)

    attributions = attributions.squeeze()
    attributions = attributions.numpy()
    attributions = nib.Nifti1Image(attributions, affine=np.eye(4))

    nib.save(attributions, results_dir + file_name + "-HM.nii")


def saliency(model, image, results_dir, file_name):
    sal = Saliency(model)
    attributions = sal.attribute(image)#, n_steps=200)

    attributions = attributions.squeeze()
    attributions = attributions.detach().numpy()
    attributions = nib.Nifti1Image(attributions, affine=np.eye(4))
    nib.save(attributions, results_dir + file_name + "-HM.nii")
    # return attributions_ig


def guided_grad_cam(model, image, results_dir, file_name):

    guided_gc = GuidedGradCam(model, model.skip_4[0])  # Last Conv Layer
    attributions = guided_gc.attribute(image)

    attributions = attributions.squeeze()
    attributions = attributions.detach().numpy()
    attributions = nib.Nifti1Image(attributions, affine=np.eye(4))

    nib.save(attributions, results_dir + file_name + "-HM.nii")


def layer_grad_cam(model , image, results_dir, file_name):

    layer_gc = LayerGradCam(model, model.skip_4[0])
    attr = layer_gc.attribute(image)

    attr = LayerAttribution.interpolate(attr, (121, 145, 121))

    attr = attr.squeeze()
    attr = attr.detach().numpy()
    attr = nib.Nifti1Image(attr, affine=np.eye(4))

    nib.save(attr, results_dir + file_name + "-HM.nii")


if __name__ == '__main__':

    # Local
    dir = "C:/Users/Aditya/Desktop/Results/"
    df = pd.read_csv(dir + '8-Best-MAE-5.1/predicted_data.csv')

    # GPU
    # dir = "results"
    # df = pd.read_csv(dir + '/predicted_data.csv')

    model = ResNet()
    model.load_state_dict(torch.load(dir + "8-Best-MAE-5.1/checkpoint.pt", map_location=torch.device('cpu')))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    model.eval()

    for i, row in df.iterrows():
        path = row['Loc']
        file_name = path[path.rfind('/'): path.rfind('.')]
        print("{}. {}: ".format(i, file_name)),

        image_path = DATA_DIR + path[path.rfind('/'):]
        image = loadMR(image_path)
        image = image.reshape((1, 1) + IMAGE_SHAPE)
        image = torch.from_numpy(image)
        image = image.float()

        # Call the method for the corresponding interpretability method
        guided_grad_cam(model, image, dir + "Heatmaps/guided_gradcam", file_name)

        # -------- Experiment ------
        # image = image.to(device)
        # p = "C:/Users/Aditya/Desktop/Results/Heatmaps/saliency/IXI376-Guys-0938-T1-HM.nii"
        # baseline = loadMR(p)
        # baseline = baseline.reshape((1, 1) + IMAGE_SHAPE)
        # baseline = torch.from_numpy(baseline)
        # baseline = baseline.float()

