import dearpygui.dearpygui as dpg

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


DEFAULT_DATA_DIRECTORY = "data/hymenoptera_data"
DEFAULT_BATCH_SIZE = 4
DEFAULT_SHUFFLE = True

DEFAULT_DEVICE = "cuda:0"

DEFAULT_MODEL = "ResNet18"
DEFAULT_WEIGHTS = models.ResNet18_Weights.DEFAULT
DEFAULT_FIXED_FE = True
dict_model = {
    "ResNet18": models.resnet18,
    "ResNet50": models.resnet50
}

DEFAULT_CRITERION = "Cross Entropy Loss"
dict_criterion = {
    "Mean Absolute Error": nn.L1Loss,
    "Mean Squared Error": nn.MSELoss,
    "Cross Entropy Loss": nn.CrossEntropyLoss
}

DEFAULT_OPTIMIZER = "Stochastic Gradient Descent"
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_MOMENTUM = 0.9
dict_optimizer = {
    "Stochastic Gradient Descent": optim.SGD,
}

DEFAULT_SCHEDULER = "Step Learning Rate"
DEFAULT_STEP_SIZE = 7
DEFAULT_GAMMA = 0.1
dict_scheduler = {
    "Step Learning Rate": lr_scheduler.StepLR
}

DEFAULT_NUM_EPOCHS = 25

# fixed arrays for normalisation
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

data_transforms = {
    # transformation of images for training
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),  # random crop to grab different part of image
        transforms.RandomHorizontalFlip(),  # random flipping to vary image
        transforms.ToTensor(),
        transforms.Normalize(mean, std)     # normalises image for colour channels to be arround the centre
    ]),
    # transformation of images for validation
    'val': transforms.Compose([
        transforms.Resize(256),         # resize to standard
        transforms.CenterCrop(224),     # crop to object
        transforms.ToTensor(),          
        transforms.Normalize(mean, std)
    ]),
}

def train(sender):
    print("TRAINING TIME")
    device = dpg.get_value(dpg_device)

    data_dir = dpg.get_value(dpg_data_directory)
    batch_size = dpg.get_value(dpg_batch_size)
    shuffle = dpg.get_value(dpg_shuffle)
    # create dictionary of image datasets for training and validation
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                        for x in ['train', 'val']}
    # create dictionary of dataloaders using dataset dictionary
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=shuffle, num_workers=0)
                        for x in ['train', 'val']}
    # create dictionary of dataset sizes for training and valuation
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    # get classes from training image dataset
    class_names = image_datasets['train'].classes
    print(class_names)
        
    model = dict_model[dpg.get_value(dpg_model)](pretrained=True)
    fixed_fe = dpg.get_value(dpg_fixed_fe)
    if (fixed_fe):
        for param in model.parameters():   # Iterate through all layers of network
            param.requires_grad = False         # Freeze layer so it is not trainable
    num_ftrs = model.fc.in_features     # get number of input features at FC layer
    model.fc = nn.Linear(num_ftrs, len(class_names)) # assign new layer with output resized to our number of classes
    model = model.to(device)

    criterion = dict_criterion[dpg.get_value(dpg_criterion)]()

    lr = dpg.get_value(dpg_learning_rate)
    momentum = dpg.get_value(dpg_momentum)

    optimizer = dict_optimizer[dpg.get_value(dpg_optimizer)]
    if fixed_fe:
        optimizer = optimizer(model.fc.parameters(), lr=lr, momentum=momentum)
    else:
        optimizer = optimizer(model.parameters(), lr=lr)

    step_size = dpg.get_value(dpg_step_size)
    gamma = dpg.get_value(dpg_gamma)
    scheduler = dict_scheduler[dpg.get_value(dpg_scheduler)](optimizer, step_size=step_size, gamma=gamma)
    
    num_epochs = dpg.get_value(dpg_num_epochs)


    # train_model
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1)) 
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

dpg.create_context()

with dpg.window(width=300, pos=(0,0), autosize=True):

    # device
    with dpg.collapsing_header(label="Hardware Acceleration", default_open=True):
        with dpg.child_window(height=50, autosize_x=True):
            devices = ["cpu"]
            if torch.cuda.is_available():
                devices.append("cuda:0")
            dpg_device = dpg.add_combo(label="Device", items=devices, default_value=DEFAULT_DEVICE)


    # dataset parameters
    with dpg.collapsing_header(label="Datasets", default_open=True):
        with dpg.child_window(height=80, autosize_x=True):
            dpg_data_directory = dpg.add_input_text(label="Data Directory", default_value=DEFAULT_DATA_DIRECTORY)
            dpg_batch_size = dpg.add_input_int(label="Batch Size", default_value=DEFAULT_BATCH_SIZE)
            dpg_shuffle = dpg.add_checkbox(label="Shuffle", default_value=DEFAULT_SHUFFLE)


    with dpg.collapsing_header(label="Models", default_open=True):
        with dpg.child_window(height=80, autosize_x=True):
            print("models = ", dict_model.keys())
            dpg_model = dpg.add_combo(label="Model", items=list(dict_model.keys()), default_value=DEFAULT_MODEL)
            dpg_fixed_fe = dpg.add_checkbox(label="Fixed Feature Extractor", default_value=DEFAULT_FIXED_FE)


    with dpg.collapsing_header(label="Configurations", default_open=True):
        with dpg.child_window(height=200, autosize_x=True):
            # loss functions
            dpg_criterion = dpg.add_combo(label="Criterion", items=list(dict_criterion.keys()), default_value=DEFAULT_CRITERION)
            # parameter optimization algorithm
            dpg_optimizer = dpg.add_combo(label="Optimizer", items=list(dict_optimizer.keys()), default_value=DEFAULT_OPTIMIZER)
            dpg_learning_rate = dpg.add_input_float(label="Learning Rate", default_value=DEFAULT_LEARNING_RATE)
            dpg_momentum = dpg.add_input_float(label="Momentum", default_value=DEFAULT_MOMENTUM)
            # learning rate schedule
            dpg_scheduler = dpg.add_combo(label="Scheduler", items=list(dict_scheduler.keys()), default_value=DEFAULT_SCHEDULER)
            dpg_step_size = dpg.add_input_int(label="Step Size", default_value=DEFAULT_STEP_SIZE)
            dpg_gamma = dpg.add_input_float(label="Gamma", default_value=DEFAULT_GAMMA)

    with dpg.collapsing_header(label="Trains", default_open=True):
        with dpg.child_window(height=60, autosize_x=True):
            dpg_num_epochs = dpg.add_input_int(label="Num Epochs", default_value=DEFAULT_NUM_EPOCHS)            
            dpg_train_btn = dpg.add_button(label="Train", callback=train)

dpg.create_viewport(title='Custom Title', width=800, height=800)
dpg.setup_dearpygui()
dpg.show_viewport()

# below replaces, start_dearpygui()
while dpg.is_dearpygui_running():
    # insert here any code you would like to run in the render loop
    # you can manually stop by using stop_dearpygui()
    # print("this will run every frame")
    dpg.render_dearpygui_frame()

dpg.destroy_context()


