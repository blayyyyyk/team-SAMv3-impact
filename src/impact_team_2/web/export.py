import torch

if torch.cuda.is_available():
    print("CUDA is available, using GPU.")
    device = torch.device("cuda")
elif torch.mps.is_available():
    print("MPS is available, using GPU.")
    device = torch.device("mps")
else:
    print("Warning: CUDA and MPS not available, using CPU instead.")
    device = torch.device("cpu")
    
    
