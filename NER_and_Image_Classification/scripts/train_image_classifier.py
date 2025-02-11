import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load images
dataset = datasets.ImageFolder('NER_and_Image_Classification/data/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Load animal names from the text file
with open('NER_and_Image_Classification/data/animals.txt', 'r') as f:
    animal_names = [line.strip() for line in f.readlines()]

# Load pre-trained model
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, len(animal_names)) 

# Define loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(20):  
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Save the model
torch.save(model.state_dict(), 'NER_and_Image_Classification/models/image_classification_model.pth')