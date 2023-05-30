import argparse
import pathlib
from tqdm import tqdm
import torch
import torchvision
import numpy as np
import cv2 as cv
import yaml
import matplotlib.pyplot as plt
import gdown
import os
import shutil

from team_template.src.model_task_1_ import main as evaluate_model_1
from team_template.src.models import Unet2, Unet3, Ensemble# UnetBCEDICE1
from team_template.src.post_processing import MorphologicalOperations
# from competition_toolkit.competition_toolkit.dataloader import create_dataloader
from competition_toolkit.competition_toolkit.eval_functions import iou, biou

import pathlib

from torch.utils.data import Dataset, DataLoader
from yaml import load, Loader
from datasets import load_dataset
import os
import torch


def get_paths_from_folder(folder: str) -> list:
    allowed_filetypes = ["jpg", "jpeg", "png", "tif", "tiff"]

    paths = []

    for file in os.listdir(folder):
        filetype = file.split(".")[1]

        if filetype not in allowed_filetypes:
            continue

        path = os.path.join(folder, file)

        paths.append(path)

    return paths


def load_image(imagepath: str, size: tuple) -> torch.tensor:
    # imagepath = 'data\\validation\\images\\6259_564_0.tif'
    image = cv.imread(imagepath, cv.IMREAD_COLOR)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, size)

    image = torch.tensor(image.astype(np.uint8)) / 255
    image = torch.permute(image, (2, 0, 1))

    return image


def load_label(labelpath: str, size: tuple) -> torch.tensor:
    label = cv.imread(labelpath, cv.IMREAD_GRAYSCALE)
    label[label == 255] = 1
    label = cv.resize(label, size)

    label = torch.tensor(label.astype(np.uint8)).long()

    return label


class TestDataset(Dataset):
    def __init__(self,
                 opts: dict, imagepaths: list, labelpaths: list,
                 datatype: str = "test"):
        self.opts = opts
        self.imagepaths = imagepaths
        self.labelpaths = labelpaths

    def __len__(self):
        return len(self.imagepaths)

    def __getitem__(self, idx):
        imagefilepath = self.imagepaths[idx]
        labelfilepath = self.labelpaths[idx]

        assert imagefilepath.split("\\")[-1] == labelfilepath.split("\\")[
            -1], f"imagefilename and labelfilename does not match; {imagefilepath.split('/')[-1]} != {labelfilepath.split('/')[-1]}"

        image = load_image(imagefilepath, (self.opts["imagesize"], self.opts["imagesize"]))
        label = load_label(labelfilepath, (self.opts["imagesize"], self.opts["imagesize"]))
        filename = imagefilepath.split("\\")[-1]

        return image, label, filename


def create_dataloader(opts: dict, imagepaths: list, labelpaths: list, datatype: str = "test") -> DataLoader:
    dataset = TestDataset(opts, imagepaths, labelpaths, datatype)
    dataloader = DataLoader(dataset, batch_size=opts[f"task{opts['task']}"]["batchsize"],
                            shuffle=opts[f"task{opts['task']}"]["shuffle"])

    return dataloader

device = 'cuda'

model1 = Unet2()
model2 = Unet3()

if torch.cuda.is_available():
    model1.load_state_dict(torch.load("./team_template/src/runs/task_1/run_29_BCE_DICE_LOSS/best_task1_104.pt"))
    model2.load_state_dict(torch.load("./team_template/src/runs/task_1/run_30/best_task1_26.pt"))

else:
    model1.load_state_dict(torch.load("./team_template/src/runs/task_1/run_29_BCE_DICE_LOSS/best_task1_104.pt",  map_location=torch.device('cpu')))
    model2.load_state_dict(torch.load("./team_template/src/runs/task_1/run_30/best_task1_26.pt", map_location=torch.device('cpu')))
    device = 'cpu'

model1.eval()
model2.eval()

models_to_ensamble = [model1.to(device), model2.to(device)]
model = Ensemble(models_to_ensamble)

if torch.cuda.is_available():
    model.load_state_dict(torch.load("./team_template/src/runs/task_1/run_32/best_task1_2.pt"))

else:
    model.load_state_dict(torch.load("./team_template/src/runs/task_1/run_32/best_task1_2.pt", map_location=torch.device('cpu')))





runs = [
    {
        'run': 'run_30',
        'model': model2,
        'model_name': 'Simple U-Net (smaller)'
    },
    {
        'run': 'run_29_BCE_DICE_LOSS',
        'model': model1,
        'model_name': 'Simple U-Net (larger)'
    },
    {
        'run': 'run_32',
        'model': model,
        'model_name': 'Ensemble'
    }

]

for run_ in runs:
    run = run_['run']
    print(run_['model_name'])
    config = f'team_template/src/runs/task_1/{run}/opts.yaml'
    with open(config, "r") as f:
        opts = yaml.load(f, Loader=yaml.Loader)
    opts["device"] = device

    dataset = {'validation': 'data/validation/', 'test': 'sjyhne/mapai_evaluation_data/test/'}
    datatype = 'test'
    opts['task1']['batchsize'] = 1
    for key in dataset:

        print(key)
        predictions_path = f'team_template/src/runs/task_1/{run}/predictions/{key}'

        imagepaths = get_paths_from_folder(f'{dataset[key]}/images')#'sjyhne/mapai_evaluation_data/test/images')
        labelpaths = get_paths_from_folder(f'{dataset[key]}/masks')#'sjyhne/mapai_evaluation_data/test/masks')
        dataloader = create_dataloader(opts, imagepaths, labelpaths, datatype=datatype)
        print(dataloader)

        iou_scores = np.zeros((len(dataloader)))
        biou_scores = np.zeros((len(dataloader)))

        # %%
        for idx, (image, label, filename) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Inference",
                                                  leave=False):
            # Split filename and extension
            filename_base, file_extension = os.path.splitext(filename[0])

            # Send image and label to device (eg., cuda)
            image = image.to(device)
            label = label.to(device)

            # Perform model prediction
            cur_model = run_['model']
            prediction = cur_model(image)
            if isinstance(prediction, dict):
                prediction = prediction["out"]
            if opts["device"] == "cpu":
                prediction = (prediction > 0.5).float().squeeze(0).squeeze(0)  # for bcedice
            else:
                prediction = (prediction > 0.5).float().cpu().squeeze(0).squeeze(0)  # for bcedice


            if opts["device"] == "cpu":
                label = label.squeeze().detach().numpy()
            else:
                label = label.squeeze().cpu().detach().numpy()

            prediction = np.uint8(prediction)
            label = np.uint8(label)
            assert prediction.shape == label.shape, f"Prediction and label shape is not same, pls fix [{prediction.shape} - {label.shape}]"

            # Predict score
            iou_score = iou(prediction, label)
            biou_score = biou(label, prediction)

            iou_scores[idx] = np.round(iou_score, 6)
            biou_scores[idx] = np.round(biou_score, 6)

            prediction_visual = np.copy(prediction)

            for idx, value in enumerate(opts["classes"]):
                prediction_visual[prediction_visual == idx] = opts["class_to_color"][value]

            if opts["device"] == "cpu":
                image = image.squeeze().detach().numpy()[:3, :, :].transpose(1, 2, 0)
            else:
                image = image.squeeze().cpu().detach().numpy()[:3, :, :].transpose(1, 2, 0)


        print("iou_score:", np.round(iou_scores.mean(), 5), "biou_score:", np.round(biou_scores.mean(), 5), "Score:",
              np.round((iou_scores.mean() + biou_scores.mean()) / 2, 5))

        print('')
