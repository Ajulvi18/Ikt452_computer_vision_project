import os

import numpy as np
from yaml import load, dump, Loader, Dumper
from tqdm import tqdm
import torch
import torchvision
from tabulate import tabulate

import torch.nn.functional as F

import argparse
import time

from competition_toolkit.dataloader import create_dataloader
from utils import create_run_dir, store_model_weights, record_scores

from competition_toolkit.eval_functions import calculate_score

import models

def test(opts, dataloader, model, lossfn):
    model.eval()

    device = opts["device"]

    losstotal = np.zeros((len(dataloader)), dtype=float)
    ioutotal = np.zeros((len(dataloader)), dtype=float)
    bioutotal = np.zeros((len(dataloader)), dtype=float)
    scoretotal = np.zeros((len(dataloader)), dtype=float)

    for idx, batch in tqdm(enumerate(dataloader), leave=False, total=len(dataloader), desc="Test"):
        image, label, filename = batch
        image = image.to(device)
        label = label.to(device)
        label = label.to(torch.float32)

        output = model(image)["out"]

        loss = lossfn(output, label).item()

        output = (output>0.5).float().squeeze(1)
        if device != "cpu":
            metrics = calculate_score(output.detach().cpu().numpy().astype(np.uint8),
                                      label.detach().cpu().numpy().astype(np.uint8))
        else:
            metrics = calculate_score(output.detach().numpy().astype(np.uint8), label.detach().numpy().astype(np.uint8))

        losstotal[idx] = loss
        ioutotal[idx] = metrics["iou"]
        bioutotal[idx] = metrics["biou"]
        scoretotal[idx] = metrics["score"]

        #if idx % 25 == 0:
        #    print(f'Loss: {loss}, IoU: {metrics["iou"]}, BIoU: {metrics["biou"]}, Score: {metrics["score"]}')

    loss = round(losstotal.mean(), 4)
    iou = round(ioutotal.mean(), 4)
    biou = round(bioutotal.mean(), 4)
    score = round(scoretotal.mean(), 4)

    return loss, iou, biou, score


def train(opts):
    device = opts["device"]

    # The current model should be swapped with a different one of your choice
    #model = torchvision.models.segmentation.fcn_resnet50(pretrained=False, num_classes=opts["num_classes"])
    model1 = models.Unet2()
    model1.load_state_dict(torch.load("./runs/task_1/run_29/best_task1_104.pt"))
    model1.eval()
    model2 = models.Unet3()
    model2.load_state_dict(torch.load("./runs/task_1/run_30/best_task1_26.pt"))
    model2.eval()
    models_to_ensamble = [model1.to(device), model2.to(device)]
    model = models.Ensemble(models_to_ensamble)
    model.cuda()
    if opts["task"] == 2:
        new_conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.backbone.conv1 = new_conv1

    model.to(device)
    dev = model.weights.device
    model = model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr=opts["lr"])
    lossfn = DiceBCELoss()

    epochs = opts["epochs"]

    trainloader = create_dataloader(opts, "train")
    valloader = create_dataloader(opts, "validation")

    bestscore = 0
    constant = 50
    for e in range(epochs):

        model.train()

        losstotal = np.zeros((len(trainloader)), dtype=float)
        scoretotal = np.zeros((len(trainloader)), dtype=float)
        ioutotal = np.zeros((len(trainloader)), dtype=float)
        bioutotal = np.zeros((len(trainloader)), dtype=float)

        stime = time.time()

        for idx, batch in tqdm(enumerate(trainloader), leave=True, total=len(trainloader), desc="Train", position=0):
            image, label, filename = batch
            image = image.to(device)
            label = label.to(device)
            label = label.to(torch.float32)

            output = model(image)

            loss = lossfn(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossitem = loss.item()
            output = (output>0.5).float().squeeze(1)
            if device != "cpu":
                trainmetrics = calculate_score(output.detach().cpu().numpy().astype(np.uint8),
                                               label.detach().cpu().numpy().astype(np.uint8))
            else:
                trainmetrics = calculate_score(output.detach().numpy().astype(np.uint8),
                                               label.detach().numpy().astype(np.uint8))

            losstotal[idx] = lossitem
            ioutotal[idx] = trainmetrics["iou"]
            bioutotal[idx] = trainmetrics["biou"]
            scoretotal[idx] = trainmetrics["score"]

            if idx % 100 == 99:
                print(f'Loss: {round(sum(losstotal[idx-constant : idx]) / constant , 4)}, IoU: {round(sum(ioutotal[idx-constant : idx]) / constant, 4)}, BIoU: {round(sum(bioutotal[idx-constant : idx]) / constant, 4)}, Score: {round(sum(scoretotal[idx-constant : idx]) / constant, 4)}')

        testloss, testiou, testbiou, testscore = test(opts, valloader, model, lossfn)
        trainloss = round(losstotal.mean(), 4)
        trainiou = round(ioutotal.mean(), 4)
        trainbiou = round(bioutotal.mean(), 4)
        trainscore = round(scoretotal.mean(), 4)

        if testscore > bestscore:
            bestscore = testscore
            print("new best score:", bestscore, "- saving model weights")
            store_model_weights(opts, model, f"best", epoch=e)
        else:
            store_model_weights(opts, model, f"last", epoch=e)

        print("")
        print(tabulate(
            [["train", trainloss, trainiou, trainbiou, trainscore], ["test", testloss, testiou, testbiou, testscore]],
            headers=["Type", "Loss", "IoU", "BIoU", "Score"]))

        scoredict = {
            "epoch": e,
            "trainloss": trainloss,
            "testloss": testloss,
            "trainiou": trainiou,
            "testiou": testiou,
            "trainbiou": trainbiou,
            "testbiou": testbiou,
            "trainscore": trainscore,
            "testscore": testscore
        }

        record_scores(opts, scoredict)


class DiceBCELoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Training a segmentation model")

    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate used during training")
    parser.add_argument("--config", type=str, default="config/data.yaml", help="Configuration file to be used")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--data_ratio", type=float, default=1.0,
                        help="Percentage of the whole dataset that is used")

    args = parser.parse_args()

    # Import config
    opts = load(open(args.config, "r"), Loader)

    #define parameters here:
    opts['task1']['batchsize'] = 4
    opts['task1']['lr'] = 0.0003

    # Combine args and opts in single dict
    try:
        opts = opts | vars(args)
    except Exception as e:
        opts = {**opts, **vars(args)}

    #opts['device'] = 'cuda'
    print("Opts:", opts)

    rundir = create_run_dir(opts)
    opts["rundir"] = rundir
    dump(opts, open(os.path.join(rundir, "opts.yaml"), "w"), Dumper)

    train(opts)
