# 💵 Fraud Currency Detection Using ResNet50

A deep learning-based web application for detecting genuine and counterfeit Bangladeshi currency notes using a fine-tuned ResNet-50 model.

The application currently supports **500 BDT** and **1000 BDT** banknotes and provides the predicted class along with the model's confidence score through an interactive Streamlit interface.

---
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-ff4b4b)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- Detects genuine and counterfeit Bangladeshi banknotes
- Supports:
  - 500 BDT
  - 1000 BDT
- Fine-tuned ResNet-50 classifier
- Displays prediction confidence
- Handles low-confidence predictions
- Clean and responsive Streamlit interface
- GPU support (CUDA) when available

---

## Demo

### Main Interface

<p align="center">
<img src="assets/app.png" width="900">
</p>

---

## Project Structure

```
Fraud-Currency-Detection/
│
├── app.py
├── best_model9740.pth
├── label_encoder.pkl
├── requirements.txt
├── README.md
│
├── assets/
│   └── app.png
│
└── notebooks/
```

---

## Model

The classifier is built using **ResNet-50** from TorchVision.

The original classification head has been replaced with a custom fully connected classifier.

```python
Linear(2048 → 256)
ReLU
Dropout(0.5)
Linear(256 → 4)
```

The four output classes are:

- Genuine 500 BDT
- Counterfeit 500 BDT
- Genuine 1000 BDT
- Counterfeit 1000 BDT

---

## Technologies Used

- Python
- PyTorch
- TorchVision
- Streamlit
- Pillow
- scikit-learn
- Joblib

---

## Image Preprocessing

Each uploaded image undergoes the following preprocessing steps before inference:

- Resize to **224 × 224**
- Convert to Tensor
- Normalize using ImageNet mean and standard deviation

```python
Resize(224×224)
ToTensor()
Normalize(mean=[0.485,0.456,0.406],
          std=[0.229,0.224,0.225])
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/mahady13/Fraud-Currency-Detection.git
```

Move into the project directory

```bash
cd Fraud-Currency-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## Usage

1. Launch the Streamlit application.
2. Upload a **500 BDT** or **1000 BDT** banknote image.
3. Click **Predict**.
4. View:
   - Predicted class
   - Confidence score

If the confidence score is very low, the application warns that the uploaded image may not belong to the supported classes.

---

## Dataset

This project was trained using the **Bangladeshi Counterfeit Currency Image Dataset**.

**Citation**

> Afif, A., & Rashid, M. (2026). *Bangladeshi Counterfeit Currency Image Dataset*. Mendeley Data, V1.

DOI:

https://doi.org/10.17632/gzzz5nrvbn.1

---

## Limitations

This model has been trained **only** on:

- 500 BDT
- 1000 BDT

Uploading other denominations (100, 200, 50 BDT, etc.) or unrelated images may produce unreliable predictions.

---

## Future Improvements

- Support additional Bangladeshi banknote denominations
- Deploy the model on the cloud
- Real-time webcam detection
- Grad-CAM visualization for model interpretability
- Mobile-friendly interface
- Model optimization for faster inference

---

## Author

**Mohiuddin Mahady**

B.Sc. in Computer Science & Engineering

Mymensingh Engineering College  
(Affiliated with the University of Dhaka)

GitHub:
https://github.com/mahady13

LinkedIn:
https://www.linkedin.com/in/mohiuddin-mahady/

---

## License

This project is released under the MIT License.