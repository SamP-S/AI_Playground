import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("Assuming that we are on a CUDA machine, this should print a CUDA device:")
print(torch.cuda.is_available())
print(torch.version)
print(device)
