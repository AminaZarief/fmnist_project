import torch 
import torch.nn as nn
import torch.optim as optim

from utils import get_data
from config import batch_size, lr, device, input_dim, epochs

# Get data 
train_loader, test_loader, num_classes = get_data(batch_size)

# Create model
class MnistClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024), 
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512), 
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 224), 
            nn.BatchNorm1d(224),
            nn.ReLU(),
            nn.Linear(224, 64), 
            nn.ReLU(),
            nn.Linear(64, num_classes), 
            
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)
    


model = MnistClassifier(input_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)




# Model training
model.train()
model.to(device)


for epoch in range(epochs):
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for idx, (X, labels) in enumerate(train_loader):
        X, labels = X.to(device), labels.to(device)

        prediction = model(X)
        loss = criterion(prediction, labels)
        
        # update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate loss and accuracy 
        total_loss += loss.item() * X.size(0)

        predicted = prediction.argmax(1) 
        total_correct += (predicted == labels).sum().item()
        total_samples += X.size(0) 

    avg_loss = total_loss / total_samples 
    avg_acc = total_correct / total_samples * 100 

    print(f'Epoch {epoch+1} / {epochs}')
    print(f'train loss: {avg_loss:.4f}, train Accuracy: {avg_acc:.2f}%')



# Evaluation 
val_loss = 0.0 
val_correct = 0 
val_samples = 0

model.eval()
with torch.no_grad():
    for idx, (X, labels ) in enumerate(test_loader):
        X, labels= X.to(device), labels.to(device)

        y_predict = model(X)
        loss = criterion(y_predict, labels)

        # Loss and accuracy calculation 
        val_loss += loss.item() * X.size(0)
        val_correct += (y_predict.argmax(1) == labels).sum().item()
        val_samples += X.size(0)

# Compute avg loss and avg acc 
avg_val_loss = val_loss / val_samples 
avg_acc = val_correct / val_samples * 100 

print(f'val loss: {avg_val_loss:.4f}, test Accuracy: {avg_acc:.2f}%')


        

        













