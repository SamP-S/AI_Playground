import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print("Check if CUDA is available:")
print(torch.cuda.is_available())
print(torch.version)
print(device)
