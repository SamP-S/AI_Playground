import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.dataset import random_split
import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm
import pandas as pd
import sys

from setup_env import setup_output_dir


class HyperParams:
    NUM_EPOCHS = 8
    BATCH_SIZE = 32
    LOAD_FROM = None
    MODEL_NAME = "resnet"
    MODELS_DIR = "./models"

### --- TRAIN MODEL
def train_model(model, criterion, optimizer, scheduler, hp):
    since = time.time()

    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(hp.NUM_EPOCHS):
        print(f"Epoch {epoch}/{hp.NUM_EPOCHS - 1}") 
        print("-----------------")


        losses = {
            "train": np.zeros(shape=(len(dataloaders["train"])), dtype=np.float32),
            "val": np.zeros(shape=(len(dataloaders["val"])), dtype=np.float32),
        }

        accuracies = {
            "train": np.zeros(shape=(len(dataloaders["train"])), dtype=np.float32),
            "val": np.zeros(shape=(len(dataloaders["val"])), dtype=np.float32),
        }

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            for idx, (inputs, labels) in enumerate(tqdm(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                losses[phase][idx] = loss.item() * inputs.size(0)
                accuracies[phase][idx] = torch.sum(preds == labels.data) / len(labels)

            if phase == "train":
                scheduler.step()

            phase_loss = np.mean(losses[phase])
            phase_acc = np.mean(accuracies[phase])

            df_train = pd.DataFrame({
                "train_loss": losses["train"],
                "train_acc": accuracies["train"]
            })

            df_eval = pd.DataFrame({
                "eval_loss": losses["val"],
                "eval_acc": accuracies["val"]
            })

            print(f"{phase}: \tloss={phase_loss:.4f}, \tacc={phase_acc:.4f}")
            torch.save(model, os.path.join(hp.MODEL_DIR, hp.MODEL_NAME + f"_{epoch}.bin"))
            df_train.to_csv(os.path.join(hp.MODEL_DIR, hp.MODEL_NAME + f"_{epoch}_train.csv"))
            df_eval.to_csv(os.path.join(hp.MODEL_DIR, hp.MODEL_NAME + f"_{epoch}_eval.csv"))

            if phase == "val":
                if phase_acc > best_acc:
                    best_acc = phase_acc
                    best_model = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {int(time_elapsed // 60)}m {time_elapsed % 60:.2f}s')

    # load best model weights
    model.load_state_dict(best_model)
    print(f"best model: acc={best_acc}")
    return model


if __name__ == "__main__":
    hy_params = HyperParams()
    hy_params.LOAD_FROM = None
    hy_params.NUM_EPOCHS = 24
    hy_params.BATCH_SIZE = 32
    hy_params.DATA_DIR = "data/augmented/old_26"
    hy_params.MODEL_DIR = setup_output_dir("data/models/old_26")
    hy_params.MODEL_NAME = "alexnet"

    ### --- PARALLELISE TRAINING
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print ("DEVICE: ", device)

    ### --- DATASET CONFIGURATION AND SETUP
    MEAN = np.array([0.5, 0.5, 0.5])
    STD = np.array([0.25, 0.25, 0.25])

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224), 
            transforms.ToTensor(),          
            transforms.Normalize(MEAN, STD)
        ]),
    }

    image_loader = datasets.ImageFolder(hy_params.DATA_DIR)

    TRAIN_VAL_RATIO = 0.2
    val_size = int(TRAIN_VAL_RATIO * len(image_loader))
    train_size = len(image_loader) - val_size
    train_dataset, val_dataset = random_split(image_loader, [train_size, val_size])

    image_datasets = {'train': train_dataset, 'val': val_dataset}
    for key, val in image_datasets.items():
        val.dataset.transform = data_transforms[key]
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=hy_params.BATCH_SIZE, shuffle=True, num_workers=0)
                        for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_loader.classes
    print(class_names)

    ### --- MODEL CONFIGURATION
    model = None
    if (hy_params.MODEL_NAME == "resnet18"):
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif (hy_params.MODEL_NAME == "resnet34"):
        model = models.resnet34(weights=models.ResNet50_Weights.DEFAULT)
    elif (hy_params.MODEL_NAME == "resnet50"):
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif (hy_params.MODEL_NAME == "resnet101"):
        model = models.resnet101(weights=models.ResNet50_Weights.DEFAULT)
    elif (hy_params.MODEL_NAME == "resnet152"):
        model = models.resnet152(weights=models.ResNet50_Weights.DEFAULT)
    elif (hy_params.MODEL_NAME == "mobilenet_v3_large"):
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        model.fc = model.classifier[-1]
    elif (hy_params.MODEL_NAME == "mobilenet_v3_small"):
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.fc = model.classifier[-1]
    elif (hy_params.MODEL_NAME == "alexnet"):
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        model.fc = model.classifier[-1]
    else:
        print(f"ERROR: Invalid model name @ {hy_params.MODEL_NAME}")
        sys.exit()

    IS_FIXED_FEATURE_EXTRACTOR = False
    if IS_FIXED_FEATURE_EXTRACTOR:
        for param in model.parameters():
            param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)                    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01) 
    if IS_FIXED_FEATURE_EXTRACTOR:
        optimizer = optim.SGD(model.fc.parameters(), lr=0.01, momentum=0.9)

    step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train_model(model, criterion, optimizer, step_lr_scheduler, hp=hy_params)

