import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from sklearn.metrics import precision_recall_fscore_support

# Download and load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the full MNIST dataset
full_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Split the full dataset into training, validation, and test sets
train_size = int(0.8 * len(full_dataset))
val_size = int(0.1 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])

# Define data loaders for training, validation, and test sets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

# Define the improved neural network model with convolutional layers, batch normalization, and dropout
class ImprovedNet(nn.Module):
    def __init__(self):
        super(ImprovedNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = ImprovedNet()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Define the learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training the neural network
for epoch in range(20):  # loop over the dataset multiple times
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f, accuracy: %.2f' %
                  (epoch + 1, i + 1, running_loss / 2000, 100 * correct_train / total_train))
            running_loss = 0.0
    
    # Step the learning rate scheduler
    scheduler.step()

print('Finished Training')

# Save the trained model
torch.save(net.state_dict(), 'mnist_model.pth')

# Load the trained model
net = ImprovedNet()
net.load_state_dict(torch.load('mnist_model.pth'))

# Visualize random images from the test set with their ground truth and predicted labels
def visualize_random_images(data_loader):
    random_indices = random.sample(range(len(data_loader.dataset)), 20)
    images = []
    ground_truth_labels = []
    predicted_labels = []
    for idx in random_indices:
        image, label = data_loader.dataset[idx]
        images.append(image)
        ground_truth_labels.append(label)
        output = net(image.unsqueeze(0))
        _, predicted = torch.max(output.data, 1)
        predicted_labels.append(predicted.item())

    plt.figure(figsize=(10, 8))
    for i in range(len(images)):
        plt.subplot(4, 5, i + 1)
        plt.imshow(images[i].numpy().squeeze(), cmap='gray')
        plt.title(f'Ground Truth: {ground_truth_labels[i]}, Predicted: {predicted_labels[i]}')
        plt.axis('off')
    plt.show()

visualize_random_images(test_loader)

# Evaluate the model on the validation set and store ground truth and predicted labels
correct_val = 0
total_val = 0
predictions_val = []
labels_val = []
with torch.no_grad():
    for data in val_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total_val += labels.size(0)
        correct_val += (predicted == labels).sum().item()
        predictions_val.extend(predicted.numpy())
        labels_val.extend(labels.numpy())

print('Accuracy of the network on the validation set: %d %%' % (
    100 * correct_val / total_val))

# Evaluate the model on the test set and store ground truth and predicted labels
correct_test = 0
total_test = 0
predictions_test = []
labels_test = []
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()
        predictions_test.extend(predicted.numpy())
        labels_test.extend(labels.numpy())

print('Accuracy of the network on the test set: %d %%' % (
    100 * correct_test / total_test))

# Evaluate precision, recall, and F1 score on the validation set
precision_val, recall_val, f1_score_val, _ = precision_recall_fscore_support(labels_val, predictions_val, average='macro')
print('Validation Precision: %.2f' % precision_val)
print('Validation Recall: %.2f' % recall_val)
print('Validation F1 Score: %.2f' % f1_score_val)
