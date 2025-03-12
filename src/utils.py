import torch 
import torchvision 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
from config import batch_size

# Transform data into tensor
transform = transforms.Compose([transforms.ToTensor()])

def get_data(batch_size):
    """Load FashionMNIST dataset and return DataLoader. """

    train_data = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=batch_size)
    
    num_classes = len(train_data.classes)

    return train_loader, test_loader, num_classes


def show_img():
    """Display the first image from the dataset."""
    dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    img, label = dataset[0]
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f'label {label}')
    plt.show()
