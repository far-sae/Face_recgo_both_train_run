import os
import ssl
from tkinter import Image

import certifi
import requests
import tarfile
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm


def download_and_extract_dataset(url, download_path, extract_to):
    os.makedirs(extract_to, exist_ok=True)
    local_filename = os.path.join(download_path, url.split('/')[-1])

    # Download the file with a progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(local_filename, 'wb') as file:
        for data in response.iter_content(block_size):
            t.update(len(data))
            file.write(data)
    t.close()

    # Extract the dataset
    if local_filename.endswith(".zip"):
        with zipfile.ZipFile(local_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif local_filename.endswith((".tar.gz", ".tar")):
        with tarfile.open(local_filename, 'r') as tar_ref:
            tar_ref.extractall(extract_to)
    print(f"Dataset downloaded and extracted to {extract_to}")


if __name__ == "__main__":
    # Step 1: Ensure SSL certificates are correctly handled by setting the SSL_CERT_FILE environment variable
    os.environ['SSL_CERT_FILE'] = certifi.where()

    # Configuration
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 2  # Set number of workers for DataLoader

    # Simplified Transformations to Reduce Load
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Debugging: Add Logging to Identify Bottleneck
    print("Loading training dataset...")
    train_dataset = CIFAR10(root="./dataset", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    print("Loading validation dataset...")
    val_dataset = CIFAR10(root="./dataset", train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    # Step 2: Model Definition using a smaller ResNet model for faster testing
    from torchvision.models import resnet18, ResNet18_Weights

    weights = ResNet18_Weights.IMAGENET1K_V1  # Use predefined weights for ResNet-18
    model = resnet18(weights=weights)  # Load the model with these weights
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=10)  # CIFAR-10 has 10 classes
    model = model.to(device)

    # Debugging: Confirm model setup
    print("Model defined and transferred to device.")

    # Step 3: Training Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Step 4: Training Loop with Progress Bar
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

        for batch_idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update progress bar
            progress_bar.set_description(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {running_loss / len(train_loader):.4f}")

    # Step 5: Validation
    print("Starting validation...")
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Validation Accuracy: {100 * correct / total:.2f}%")

    # Step 6: Save the Trained Model
    torch.save(model.state_dict(), "resnet18_model.pth")
    print("Model saved successfully.")


    # Step 7: Inference Function
    def infer(image_path, model, transform):
        model.eval()
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output.data, 1)

        return predicted.item()

    # Example Inference (Uncomment to test with an actual image)
    # pred_class = infer("path/to/image.jpg", model, transform)
    # print(f"Predicted Class: {pred_class}")
