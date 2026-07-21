import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import joblib
import streamlit as st
from PIL import Image
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Fraud Currency Detection Using ResNet50", page_icon="💵", layout="wide")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#model building & loading
@st.cache_resource
def load_assets():
    checkpoint = torch.load("best_model9740.pth", map_location=device)
    model = models.resnet50(weights=None)

    model.fc = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 4)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    return model
model=load_assets()
model.to(device)
model.eval()

#label encoder loaded
label_encoder = joblib.load("label_encoder.pkl")

#transform function created
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#sidebar section
with st.sidebar:
    st.header("Developer Information")
    st.markdown("**Mohiuddin Mahady**")
    st.text("BSc in CSE")
    st.text("Mymensingh Engineering College(Affiliated with Dhaka University)")
    col3,col4=st.columns([1,1])
    with col3:
        st.link_button("LinkedIn","https://www.linkedin.com/in/mohiuddin-mahady/",use_container_width=True)
    with col4:
        st.link_button("Github",'https://www.github.com/mahady13',use_container_width=True)
    st.markdown("---")

    st.header("ℹ️ About App")
    st.write(
        "This application uses a fine-tuned **ResNet-50** deep learning model to detect genuine and fraud Bangladeshi currency notes (500 BDT & 1000 BDT).")
    st.markdown("---")

    st.header("Dataset & Citation")
    st.markdown("🔗 **[Main Dataset Source](https://doi.org/10.17632/gzzz5nrvbn.1)**")
    st.caption("Afif, Afif; Shad, Muhaiminul Rashid (2026), “Bangladeshi Counterfeit Currency Image Dataset”, Mendeley Data, V1")

#main page ui
st.title("Fraud Currency Detection Using ResNet50")
st.warning("⚠ **Note:** This model is trained specifically on **500 BDT and 1000 BDT** banknotes. Uploading other currency notes (e.g., 100, 200 BDT) or random images may produce inaccurate predictions.")

img = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
