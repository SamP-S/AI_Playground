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
import argparse

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

    epoch_losses = {
        "train": np.zeros(shape=(hp.NUM_EPOCHS), dtype=np.float32),
        "val": np.zeros(shape=(hp.NUM_EPOCHS), dtype=np.float32),
    }

    epoch_accuracies = {
        "train": np.zeros(shape=(hp.NUM_EPOCHS), dtype=np.float32),
        "val": np.zeros(shape=(hp.NUM_EPOCHS), dtype=np.float32),
    }

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

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            batch_count = len(dataloaders[phase])
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

                batch_loss = loss.item() * inputs.size(0)
                batch_acc = torch.sum(preds == labels.data) / len(labels)
                # print(f"batch {idx+1}/{batch_count} - epoch {epoch+1}/{hp.NUM_EPOCHS} :\tloss={batch_loss:.4f}, \tacc={batch_acc:.4f}")
                
                # print("outputs:")
                # print(outputs)
                # print("predictions:")
                # print(preds)
                # print("labels:")
                # print(labels)
                # print("acc:")
                # print(preds == labels.data)
                # print("sizes")
                # print(len(preds))
                # print(len(labels))
                losses[phase][idx] = batch_loss
                accuracies[phase][idx] = batch_acc

            if phase == "train":
                scheduler.step()
            
            # print(accuracies[phase])
            phase_loss = np.mean(losses[phase])
            phase_acc = np.mean(accuracies[phase])
            print(f"{phase}: \tloss={phase_loss:.4f}, \tacc={phase_acc:.4f}\n")

            epoch_losses[phase][epoch] = phase_loss
            epoch_accuracies[phase][epoch] = phase_acc

        # keep track of best model
        if phase == "val":
            if phase_acc > best_acc:
                best_acc = phase_acc
                best_model = copy.deepcopy(model.state_dict())

        # save model after both phases
        torch.save(model, os.path.join(hp.MODEL_DIR, hp.MODEL_NAME + f"_{epoch}.bin"))
            
    # store accs/loss per epoch
    df_perf = pd.DataFrame({
        "train_loss": epoch_losses["train"],
        "train_acc": epoch_accuracies["train"],
        "eval_loss": epoch_losses["val"],
        "eval_acc": epoch_accuracies["val"]
    })
        
    df_perf.to_csv(os.path.join(hp.MODEL_DIR, hp.MODEL_NAME + f"_perf.csv"), index=False)

    time_elapsed = time.time() - since
    print(f'Training complete in {int(time_elapsed // 60)}m {time_elapsed % 60:.2f}s')

    # load best model weights
    model.load_state_dict(best_model)
    print(f"best model: acc={best_acc}")
    return model


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Hyperparameters for the model')
    parser.add_argument('--LOAD_FROM', type=str, help='The model to load from')
    parser.add_argument('--NUM_EPOCHS', type=int, help='The number of epochs', default=25)
    parser.add_argument('--BATCH_SIZE', type=int, help='The batch size', default=32)
    parser.add_argument('--DATA_DIR', type=str, help='The data directory', default="data/bricks/15k_cycles")
    parser.add_argument('--MODEL_DIR', type=str, help='The model directory', default="data/models/15k_cycles")
    parser.add_argument('--MODEL_NAME', type=str, help='The model name', default="resnet50")
    parser.add_argument('--USE_GPU', type=int, help='Use GPU if available', default=1)
    args = parser.parse_args()

    # Assign the arguments to hy_params
    hy_params = HyperParams()
    print(args.LOAD_FROM)
    hy_params.LOAD_FROM = args.LOAD_FROM
    hy_params.NUM_EPOCHS = args.NUM_EPOCHS
    hy_params.BATCH_SIZE = args.BATCH_SIZE
    hy_params.DATA_DIR = args.DATA_DIR
    hy_params.MODEL_DIR = args.MODEL_DIR
    hy_params.MODEL_NAME = args.MODEL_NAME

    if not os.path.exists(args.MODEL_DIR):
        os.mkdir(args.MODEL_DIR)

    ### --- PARALLELISE TRAINING
    print(f"USE_GPU = {args.USE_GPU}")
    device = "cpu"
    if args.USE_GPU:
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

    # load images from seperate folders
    train_dir = os.path.join(hy_params.DATA_DIR, 'train')
    val_dir = os.path.join(hy_params.DATA_DIR, 'val')
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'val': datasets.ImageFolder(val_dir, transform=data_transforms['val'])
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=hy_params.BATCH_SIZE, shuffle=True, num_workers=32)
        for x in ['train', 'val']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(f"total num classes = {len(class_names)} : {class_names}")

    ### --- MODEL CONFIGURATION
    model = None
    if (hy_params.MODEL_NAME == "resnet18"):
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif (hy_params.MODEL_NAME == "resnet34"):
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    elif (hy_params.MODEL_NAME == "resnet50"):
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif (hy_params.MODEL_NAME == "resnet101"):
        model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    elif (hy_params.MODEL_NAME == "resnet152"):
        model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    elif (hy_params.MODEL_NAME == "mobilenet_v3_large"):
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        model.fc = model.classifier[-1]
    elif (hy_params.MODEL_NAME == "mobilenet_v3_small"):
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.fc = model.classifier[-1]
    elif (hy_params.MODEL_NAME == "alexnet"):
        model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        model.fc = model.classifier[-1]
    elif (hy_params.MODEL_NAME == "convnext_base"):
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        model.fc = model.classifier[-1]
    elif (hy_params.MODEL_NAME == "convnext_large"):
        model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT)
        model.fc = model.classifier[-1]
    elif (hy_params.MODEL_NAME == "convnext_small"):
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
        model.fc = model.classifier[-1]
    elif (hy_params.MODEL_NAME == "convnext_tiny"):
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        model.fc = model.classifier[-1]
    # elif (hy_params.MODEL_NAME == "vit_b_16"):
    #     model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    #     model.fc = model.classifier[-1]
    # elif (hy_params.MODEL_NAME == "vit_b_32"):
    #     model = models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT)
    #     model.fc = model.classifier[-1]
    # elif (hy_params.MODEL_NAME == "vit_b_16"):
    #     model = models.vit_h_14(weights=models.ViT_H_14_Weights.DEFAULT)
    #     model.fc = model.classifier[-1]
    # elif (hy_params.MODEL_NAME == "vit_b_16"):
    #     model = models.vit_l_16(weights=models.ViT_L_16_Weights.DEFAULT)
    #     model.fc = model.classifier[-1]
    # elif (hy_params.MODEL_NAME == "vit_b_16"):
    #     model = models.vit_l_32(weights=models.ViT_L_32_Weights.DEFAULT)
    #     model.fc = model.classifier[-1]
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

    dt = time.time() - start_time
    print(f"finished in {dt:.3f}s == {dt/60:.3f}m == {dt / (60*60):.3f}")
