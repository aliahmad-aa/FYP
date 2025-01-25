import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import timm
import json
import numpy as np

# Load class names from provided dictionary or JSON file
class_names = {'Alexandrite': 0, 'Almandine': 1, 'Amazonite': 2, 'Amber': 3, 'Amethyst': 4, 'Ametrine': 5, 'Andalusite': 6, 'Andradite': 7, 'Aquamarine': 8, 'Aventurine Green': 9, 'Aventurine Yellow': 10, 'Benitoite': 11, 'Beryl Golden': 12, 'Bixbite': 13, 'Bloodstone': 14, 'Blue Lace Agate': 15, 'Brazilianite': 16, 'Carnelian': 17, 'Cats Eye': 18, 'Chalcedony Blue': 19, 'Chalcedony': 20, 'Chrome Diopside': 21, 'Chrysoberyl': 22, 'Chrysocolla': 23, 'Chrysoprase': 24, 'Citrine': 25, 'Coral': 26, 'Danburite': 27, 'Diamond': 28, 'Diaspore': 29, 'Dumortierite': 30, 'Emerald': 31, 'Fluorite': 32, 'Garnet Red': 33, 'Goshenite': 34, 'Grossular': 35, 'Hessonite': 36, 'Hiddenite': 37, 'Iolite': 38, 'Jade': 39, 'Jasper': 40, 'Kunzite': 41, 'Kyanite': 42, 'Labradorite': 43, 'Lapis Lazuli': 44, 'Larimar': 45, 'Malachite': 46, 'Moonstone': 47, 'Morganite': 48, 'Onyx Black': 49, 'Onyx Green': 50, 'Onyx Red': 51, 'Opal': 52, 'Pearl': 53, 'Peridot': 54, 'Prehnite': 55, 'Pyrite': 56, 'Pyrope': 57, 'Quartz Beer': 58, 'Quartz Lemon': 59, 'Quartz Rose': 60, 'Quartz Rutilated': 61, 'Quartz Smoky': 62, 'Rhodochrosite': 63, 'Rhodolite': 64, 'Rhodonite': 65, 'Ruby': 66, 'Sapphire Blue': 67, 'Sapphire Pink': 68, 'Sapphire Purple': 69, 'Sapphire Yellow': 70, 'Scapolite': 71, 'Serpentine': 72, 'Sodalite': 73, 'Spessartite': 74, 'Sphene': 75, 'Spinel': 76, 'Spodumene': 77, 'Sunstone': 78, 'Tanzanite': 79, 'Tigers Eye': 80, 'Topaz': 81, 'Tourmaline': 82, 'Tsavorite': 83, 'Turquoise': 84, 'Variscite': 85, 'Zircon': 86, 'Zoisite': 87}

# Load the PyTorch model
@st.cache_resource
def load_model():
    model_path = "gem_best_model.pth"
    try:
        # Initialize ReXNet-150 model
        model = timm.create_model("rexnet_150", pretrained=False)
        num_classes = len(class_names)  # Adjust this to match the number of classes in your task
        model.head.fc = torch.nn.Linear(model.head.fc.in_features, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")), strict=False)
        model.eval()  # Set the model to evaluation mode
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_model()

# Define the preprocessing transformations
def preprocess_image(image):
    """
    Preprocess the uploaded image to match the model input format.
    """
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

# Predict the gemstone label
def predict_gemstone(image, model):
    """
    Predict the gemstone label using the model.
    """
    processed_image = preprocess_image(image)
    with torch.no_grad():  # Disable gradient calculation for inference
        prediction = model(processed_image)
        predicted_idx = torch.argmax(prediction, dim=1).item()
        predicted_label = [k for k, v in class_names.items() if v == predicted_idx][0]
    return predicted_label

# Streamlit App
st.title("Gemstone Classifier")

st.write("Upload an image of a gemstone to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")  # Convert image to RGB
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if model:
        # Predict gemstone
        with st.spinner("Classifying..."):
            try:
                label = predict_gemstone(image, model)
                st.success(f"The gemstone is: {label}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
    else:
        st.error("Model could not be loaded. Please check the model file.")
