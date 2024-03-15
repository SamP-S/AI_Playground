# AI Playground

## Info

### MODEL CONFIGURATION
https://pytorch.org/vision/stable/models.html
Load a pretrained model and reset final fully connected layer.
```
# model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) 
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)   
# model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
# model = models.GoogLeNet()
```

### ConvNet as fixed feature extractor ####
Here, we need to freeze all the network except the final layer.
We need to set requires_grad == False to freeze the parameters so that the gradients are not computed in backward()
```
IS_FIXED_FEATURE_EXTRACTOR = True
if IS_FIXED_FEATURE_EXTRACTOR:
    for param in model.parameters():   # Iterate through all layers of network
        param.requires_grad = False         # Freeze layer so it is not trainable
# Replacing final Fully Connected (FC) output layer
# As a new layer is created, requires_grad == True so it is retrained
num_ftrs = model.fc.in_features     # get number of input features at FC layer
model.fc = nn.Linear(num_ftrs, len(class_names)) # assign new layer with output resized to our number of classes
model = model.to(device)    # send to device
```

### CRITERION CONFIGURATION
 https://pytorch.org/docs/stable/nn.html#loss-functions
 Criterion determines the loss function of model to determine current performance
 nn.L1Loss     - measures the mean absolute error (MAE) between each element in the input xx and target yy
 nn.MSELoss    - measures the mean squared error (squared L2 norm) between each element in the input xx and target yy.
nn.CrossEntropyLoss   - computes the cross entropy loss between input logits and target.
```
criterion = nn.CrossEntropyLoss()
```

### OPTIMIZER CONFIGURATION
https://pytorch.org/docs/stable/optim.html
Optimizer algorithm determines how the loss function is used to update the model's parameter
Observe that all parameters are being optimized
using stochastic gradient descent (optionally with momentum).
```
optimizer = optim.SGD(model.parameters(), lr=0.001) 
if IS_FIXED_FEATURE_EXTRACTOR:
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)
```

### --- SCHEDULER CONFIGURATION
https://pytorch.org/docs/stable/optim.html
LR Scheduler can adjust the learning rate based on the number of epochs completed
StepLR Decays the learning rate of each parameter group by gamma every step_size epochs
Decay LR by a factor of 0.1 every 7 epochs
Learning rate scheduling should be applied after optimizer’s update
Decay LR by a factor of 0.1 every 7 epochs
```
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
```

## Goals

### Playground
[ ] - Implement generic ai fine tuning program
[ ] - Support pausing and continuing training when needed
[ ] - CLI only

### Training
[ ] - Fine tuning should involve transfer learning
[ ] - Hyperparameter selection will likely be required
[ ] - Support nvidia GPU parallelisation

### Models
[ ] - Resnet18 (2015)
[ ] - Resnet50/Resnet152 (2015)
[ ] - MobileNetv3 (2019)
[ ] - ViT (2021/2022)
[ ] - Dino ViT (2023)
[ ] - DeiT (2021/2023)

### Evaluation
[ ] - Models should be saved often for evaulation later, every epoch
[ ] - Loss & accuracy should be saved per batch to be able to recreate graphs



