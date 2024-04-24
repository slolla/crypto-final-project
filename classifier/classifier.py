import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Load pre-trained model (ResNet50 in this example)
model = models.resnet50(pretrained=True)
# Remove the final classification layer
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
# Set the model to evaluation mode
feature_extractor.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),  # Resize the image to 256x256 pixels
    transforms.CenterCrop(224),  # Crop the image to 224x224 pixels around the center
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Load images from a directory
image_folder = "images"
dataset = datasets.ImageFolder(image_folder, transform=transform)

# Create a data loader to load images in batches
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

# Extract features from images
all_features = []
all_labels = []
for images, labels in data_loader:
    with torch.no_grad():  # Disable gradient computation to speed up inference
        # Extract features using the pre-trained model
        features = feature_extractor(images)
        # Flatten the features
        features = features.view(features.size(0), -1)
        # Append the features to the list
        all_features.append(features)
        all_labels.append(labels)

# Concatenate features from all batches
all_features = torch.cat(all_features)
all_labels = torch.cat(all_labels)

print(all_features.shape)
print(all_labels.shape)

# Create and run our classifier
X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.2, random_state=42)

# Define a simple fully connected neural network classifier
class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
        
    def forward(self, x):
        x = self.fc(x)
        return x

# Initialize the classifier
input_size = all_features.size(1)  # Size of the input features
num_classes = len(dataset.classes)  # Number of classes
classifier = Classifier(input_size, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

# Training loop
num_epochs = 10000
for epoch in range(num_epochs):
    classifier.train()  # Set the model to training mode
    running_loss = 0.0
    for features, labels in zip(X_train, y_train):
        # Forward pass
        outputs = classifier(features)
        loss = criterion(outputs.unsqueeze(0), labels.unsqueeze(0))
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Print average loss for each epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(X_train)}')

# Evaluate the classifier on the test set
classifier.eval()  # Set the model to evaluation mode
with torch.no_grad():
    outputs = classifier(X_test)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print(f'Accuracy on test set: {accuracy}')