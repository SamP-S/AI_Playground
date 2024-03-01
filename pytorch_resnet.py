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

from setup_env import setup_output_dir


class HyperParams:
    NUM_EPOCHS = 8
    LOAD_FROM = None
    MODEL_NAME = "resnet"
    MODELS_DIR = "./models"

### --- TRAIN MODEL
# Abstract model training function
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
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            

            # Iterate over data.
            for idx, (inputs, labels) in enumerate(tqdm(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
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
    hy_params.DATA_DIR = "./data/26_bricks_augmented_half"
    hy_params.MODEL_DIR = setup_output_dir("./models/26_bricks_augmented_half")
    hy_params.MODEL_NAME = "resnet"

    ### --- PARALLELISE TRAINING
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print ("DEVICE: ", device)

    ### --- DATASET CONFIGURATION AND SETUP
    # fixed arrays for normalisation
    MEAN = np.array([0.5, 0.5, 0.5])
    STD = np.array([0.25, 0.25, 0.25])

    data_transforms = {
        # transformation of images for training
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ]),
        # transformation of images for validation
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224), 
            transforms.ToTensor(),          
            transforms.Normalize(MEAN, STD)
        ]),
    }

    # create dataset from folder structure
    image_loader = datasets.ImageFolder(hy_params.DATA_DIR)

    # split into train and validation datasets according to ratio
    TRAIN_VAL_RATIO = 0.2
    val_size = int(TRAIN_VAL_RATIO * len(image_loader))
    train_size = len(image_loader) - val_size
    train_dataset, val_dataset = random_split(image_loader, [train_size, val_size])

    # assemble into dataset dictionary and set transforms
    image_datasets = {'train': train_dataset, 'val': val_dataset}
    for key, val in image_datasets.items():
        val.dataset.transform = data_transforms[key]
    
    # create dictionary of dataloaders using dataset dictionary
    #   BATCH SIZE = 4;
    #   SHUFFLE = TRUE;
    #   NUM_WORKERS = 0;
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0)
                        for x in ['train', 'val']}
    # create dictionary of dataset sizes for training and valuation
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    # get classes from image loader
    class_names = image_loader.classes
    print(class_names)


    ### --- MODEL CONFIGURATION
    # model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) 
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)   
    # model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    # model = models.GoogLeNet()

    #### ConvNet as fixed feature extractor ####
    IS_FIXED_FEATURE_EXTRACTOR = False
    if IS_FIXED_FEATURE_EXTRACTOR:
        for param in model.parameters():        # Iterate through all layers of network
            param.requires_grad = False         # Freeze layer so it is not trainable
                                                # Replacing final Fully Connected (FC) output layer
                                                # As a new layer is created, requires_grad == True so it is retrained
    num_ftrs = model.fc.in_features             # get number of input features at FC layer
    model.fc = nn.Linear(num_ftrs, len(class_names)) # assign new layer with output resized to our number of classes
    model = model.to(device)                    # send to device



    ### --- CRITERION CONFIGURATION
    criterion = nn.CrossEntropyLoss()



    ### --- OPTIMIZER CONFIGURATION
    optimizer = optim.SGD(model.parameters(), lr=0.001) 
    if IS_FIXED_FEATURE_EXTRACTOR:
        optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)


    ### --- SCHEDULER CONFIGURATION
    step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)



    ### --- TRAIN MODEL
    model = train_model(model, criterion, optimizer, step_lr_scheduler, hp=hy_params)

