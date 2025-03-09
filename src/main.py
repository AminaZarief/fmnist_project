import torch 
import torchvision
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt 

# Download FMNIST dataset if not already present 
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.FashionMNIST(root = './data', train=True, download=False, transform=transform)

# Display the first image 
img, label = trainset[0]
plt.imshow(img.squeeze(), cmap='gray')
plt.title(f'label:{label}')
plt.show()
