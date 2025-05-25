import os
import shutil
import random
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from tqdm import tqdm
import requests
from io import BytesIO

# Visualize some random images with labels
def show_images(df, data_folder, n=6):
    samples = df.sample(n)
    plt.figure(figsize=(15, 8))
    for i, (_, row) in enumerate(samples.iterrows()):
        img_path = os.path.join(data_folder, 'train', row['id'] + '.jpg')
        img = Image.open(img_path)
        plt.subplot(2, 3, i+1)
        plt.imshow(img)
        plt.title(row['breed'])
        plt.axis('off')
    plt.show()

# Training function
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Prediction function for a PIL image
def predict(image, model, classes, val_transforms, device):
    image = val_transforms(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
    return classes[pred.item()], conf.item()

# Prediction from URL function
def predict_from_url(url, model, classes, val_transforms, device):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    return predict(img, model, classes, val_transforms, device)

# Interactive user input prediction
def predict_from_user(model, classes, val_transforms, device):
    choice = input("Choose input method:\n1 - Image URL\n2 - Local Image Path\nEnter 1 or 2: ").strip()

    if choice == "1":
        img_url = input("Enter the image URL: ").strip()
        if not img_url:
            print("No URL provided.")
            return
        try:
            response = requests.get(img_url)
            img = Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            print(f"Failed to load image from URL: {e}")
            return

    elif choice == "2":
        img_path = input("Enter local image file path: ").strip()
        if not os.path.isfile(img_path):
            print("File does not exist.")
            return
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Failed to load local image: {e}")
            return
    else:
        print("Invalid choice.")
        return

    pred_class, confidence = predict(img, model, classes, val_transforms, device)
    print(f"Predicted class: {pred_class}, Confidence: {confidence:.4f}")

def main():
    data_dir = 'dog_breed_data'
    print("Files in data directory:")
    print(os.listdir(data_dir))

    print("\nFiles in train folder:")
    print(os.listdir(os.path.join(data_dir, 'train'))[:5])

    # Load labels CSV
    labels_path = os.path.join(data_dir, 'labels.csv')
    labels_df = pd.read_csv(labels_path)
    print(labels_df.head())
    print(f"\nTotal images: {len(labels_df)}")
    print(f"Number of breeds: {labels_df['breed'].nunique()}")

    # Uncomment to visualize some images
    # show_images(labels_df, data_dir)

    # Organize images into breed folders (only run once)
    train_split_dir = os.path.join(data_dir, 'train_split')
    os.makedirs(train_split_dir, exist_ok=True)

    for _, row in labels_df.iterrows():
        breed_dir = os.path.join(train_split_dir, row['breed'])
        os.makedirs(breed_dir, exist_ok=True)
        src = os.path.join(data_dir, 'train', row['id'] + '.jpg')
        dst = os.path.join(breed_dir, row['id'] + '.jpg')
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)

    # Define transforms
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Load full dataset
    full_dataset = datasets.ImageFolder(train_split_dir, transform=train_transforms)

    # Split dataset into train and val sets
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Set val transform explicitly
    val_dataset.dataset.transform = val_transforms

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Number of classes
    num_classes = len(full_dataset.classes)
    print(f"Number of classes: {num_classes}")

    # Load pretrained model and replace final layer
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    num_epochs = 10

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    # Save model weights
    torch.save(model.state_dict(), 'best_model.pth')

    # Load model for inference
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()

    # Classes list
    classes = full_dataset.classes

    # Example: Predict from local image file
    img_path = 'dog.jpg'  # Change this to your image path
    if os.path.isfile(img_path):
        img = Image.open(img_path).convert('RGB')
        pred_class, confidence = predict(img, model, classes, val_transforms, device)
        print(f"Predicted class: {pred_class}, Confidence: {confidence:.4f}")

    # Uncomment to run interactive prediction in a script environment
    # predict_from_user(model, classes, val_transforms, device)

    # Optional: Jupyter notebook widget for file upload and prediction
    try:
        import ipywidgets as widgets
        from IPython.display import display
        import io

        def on_upload_change(change):
            for name, file_info in uploader.value.items():
                img = Image.open(io.BytesIO(file_info['content'])).convert('RGB')
                display(img)
                label, conf = predict(img, model, classes, val_transforms, device)
                print(f"Predicted Breed: {label} | Confidence: {conf:.4f}")

        uploader = widgets.FileUpload(accept='image/*', multiple=False)
        uploader.observe(on_upload_change, names='value')
        display(uploader)
    except ImportError:
        pass  # ipywidgets not installed or running outside Jupyter


if __name__ == '__main__':
    main()
