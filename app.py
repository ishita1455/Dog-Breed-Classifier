import streamlit as st
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

# Set page title and layout
st.set_page_config(page_title="Dog Breed Classifier", layout="centered")

# Title
st.title("🐶 Dog Breed Classifier")
st.markdown("Upload an image of a dog and this app will predict its breed!")

# Load model and classes
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 120)
    model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
    model.eval()
    return model

@st.cache_data
def get_classes():
    return sorted(os.listdir('dog_breed_data/train_split'))

model = load_model()
classes = get_classes()

# Image transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Prediction
def predict_image(image):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return classes[pred.item()], conf.item()

# File upload
uploaded_file = st.file_uploader("Upload a dog image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Classifying..."):
        label, confidence = predict_image(image)

    st.success(f"**Breed:** {label.replace('_', ' ').title()}  \n**Confidence:** {confidence:.2%}")
