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

### --- DISPLAY FIRST IMAGE BATCH
# take grid of images and title as string and display using matplotlib
def imshow(inp, title):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()

### --- TRAIN MODEL
# Abstract model training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
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


if __name__ == "__main__":
    ### --- PARALLELISE TRAINING
    # set device as GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print ("DEVICE: ", device)


    ### --- DATASET CONFIGURATION AND SETUP
    DATA_DIR = './data/flower_photos'  # data directory

    # fixed arrays for normalisation
    MEAN = np.array([0.5, 0.5, 0.5])
    STD = np.array([0.25, 0.25, 0.25])

    data_transforms = {
        # transformation of images for training
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),  # random crop to grab different part of image
            transforms.RandomHorizontalFlip(),  # random flipping to vary image
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)     # normalises image for colour channels to be arround the centre
        ]),
        # transformation of images for validation
        'val': transforms.Compose([
            transforms.Resize(256),         # resize to standard
            transforms.CenterCrop(224),     # crop to object
            transforms.ToTensor(),          
            transforms.Normalize(MEAN, STD)
        ]),
    }

    # create dataset from folder structure
    image_loader = datasets.ImageFolder(DATA_DIR)

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

    # Get a batch (4) of training data
    inputs, classes = next(iter(dataloaders['train']))
    # Make image grid from batch
    out = torchvision.utils.make_grid(inputs)
    # display
    # imshow(out, title=[class_names[x] for x in classes])



    ### --- MODEL CONFIGURATION
    # https://pytorch.org/vision/stable/models.html
    # Load a pretrained model and reset final fully connected layer.
    # model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) 
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)   
    # model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    # model = models.GoogLeNet()

    #### ConvNet as fixed feature extractor ####
    # Here, we need to freeze all the network except the final layer.
    # We need to set requires_grad == False to freeze the parameters so that the gradients are not computed in backward()
    IS_FIXED_FEATURE_EXTRACTOR = True
    if IS_FIXED_FEATURE_EXTRACTOR:
        for param in model.parameters():   # Iterate through all layers of network
            param.requires_grad = False         # Freeze layer so it is not trainable
    # Replacing final Fully Connected (FC) output layer
    # As a new layer is created, requires_grad == True so it is retrained
    num_ftrs = model.fc.in_features     # get number of input features at FC layer
    model.fc = nn.Linear(num_ftrs, len(class_names)) # assign new layer with output resized to our number of classes
    model = model.to(device)    # send to device



    ### --- CRITERION CONFIGURATION
    # https://pytorch.org/docs/stable/nn.html#loss-functions
    # Criterion determines the loss function of model to determine current performance
    #
    # nn.L1Loss     - measures the mean absolute error (MAE) between each element in the input xx and target yy
    # nn.MSELoss    - measures the mean squared error (squared L2 norm) between each element in the input xx and target yy.
    # nn.CrossEntropyLoss   - computes the cross entropy loss between input logits and target.
    criterion = nn.CrossEntropyLoss()



    ### --- OPTIMIZER CONFIGURATION
    # https://pytorch.org/docs/stable/optim.html
    # Optimizer algorithm determines how the loss function is used to update the model's parameter
    #
    # Observe that all parameters are being optimized
    # using stochastic gradient descent (optionally with momentum).
    optimizer = optim.SGD(model.parameters(), lr=0.001) 
    if IS_FIXED_FEATURE_EXTRACTOR:
        optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)


    ### --- SCHEDULER CONFIGURATION
    # https://pytorch.org/docs/stable/optim.html
    # LR Scheduler can adjust the learning rate based on the number of epochs completed
    #
    # StepLR Decays the learning rate of each parameter group by gamma every step_size epochs
    # Decay LR by a factor of 0.1 every 7 epochs
    # Learning rate scheduling should be applied after optimizer’s update
    # e.g., you should write your code this way:
    # for epoch in range(100):
    #     train(...)
    #     validate(...)
    #     scheduler.step()
    # Decay LR by a factor of 0.1 every 7 epochs
    step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)



    ### --- TRAIN MODEL
    model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=25)

