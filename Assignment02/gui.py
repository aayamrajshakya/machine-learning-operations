import os
from PIL import Image
import mlflow.pytorch
import streamlit as st
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

DATASET_PATH = "sea_animal_dataset"
BEST_MODEL = "models:/mobilenet_model@production"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@st.cache_resource
def load_model():
    return mlflow.pytorch.load_model(BEST_MODEL).to(DEVICE).eval()


@st.cache_data
def load_class_names():
    if not os.path.isdir(DATASET_PATH):
        raise FileNotFoundError(f"Dataset path {DATASET_PATH} not found")
    return ImageFolder(root=DATASET_PATH).classes


st.title("Sea Animal Classifier")

try:
    with st.spinner("Loading production model..."):
        model = load_model()
        class_names = load_class_names()
except Exception as e:
    st.error(f"Could not load the production model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, width="content")

        tensor_img = transform(image).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():  # better than torch.no_grad
            outputs = model(tensor_img)
            probabilities = torch.softmax(outputs[0], dim=0)
            top_prob, top_class_idx = probabilities.max(0)

        st.write(f"Prediction: {class_names[top_class_idx.item()]}")
        st.write(f"Confidence: {top_prob.item() * 100:.2f}%")

    except Exception as e:
        st.error(f"Error: {e}")
