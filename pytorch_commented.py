import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models



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
model = train_model(model, criterion, optimizer, step_lr_scheduler, hp=hy_params)

