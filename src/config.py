import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )
batch_size = 32 
input_dim = 28 * 28 
lr = 0.001
epochs = 50