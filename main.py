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