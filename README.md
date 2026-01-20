# 🐶🐱 Cats vs Dogs Image Classifier  
**TensorFlow · CNN · Transfer Learning · Streamlit**

## 📌 Overview

This project is a **binary image classification system** that predicts whether an uploaded image is a **cat 🐱** or a **dog 🐶**.

The project was built **iteratively**, starting from a basic CNN and gradually evolving into a **high-accuracy transfer learning model**, demonstrating real-world deep learning engineering practices.

### Technologies Used
- **TensorFlow / Keras**
- **Custom CNN & Transfer Learning (MobileNetV2)**
- **Cleaned Cats vs Dogs dataset**
- **Streamlit web application** for deployment

---
## Screenshots

---
## 🗂 Dataset Structure

The dataset is organized as:

data/
├── cats/
└── dogs/


Before training, the dataset is **automatically scanned and cleaned** to remove:

- Corrupted images  
- Invalid JPEG files  
- Non-RGB images  

This ensures **stable training** and prevents runtime errors during both **training and inference**.

---

## 🧠 Model Evolution & Architecture

### 🔹 Phase 1: Baseline CNN
- Simple CNN trained from scratch  
- Basic convolution + pooling layers  
- **Validation Accuracy:** ~70%

---

### 🔹 Phase 2: Improved CNN + Data Augmentation

**Enhancements:**
- Deeper CNN architecture  
- Data augmentation (random flips, rotations, zoom)  
- Dropout for regularization  

**Result:**  
➡️ Validation accuracy improved to **~85%**

---

### 🔹 Phase 3: Transfer Learning (MobileNetV2)

Final upgrade using **MobileNetV2 pretrained on ImageNet**:

- Pretrained backbone used as a feature extractor  
- Model-specific preprocessing applied  
- Custom classifier head added  
- Lower learning rate for stable training  

**Result:**  
➡️ Validation accuracy improved to **~95–97%**

This approach significantly improved **generalization** and **training efficiency**.

---

## ⚙️ Training Pipeline

1. Load and clean dataset from directory  
2. Apply model-specific preprocessing  
3. Apply real-time data augmentation  
4. Train model (CNN → Transfer Learning)  
5. Save trained model as `.h5`

---

## 🚀 Streamlit App Features

- Upload an image (`.jpg`, `.jpeg`, `.png`)  
- Automatic preprocessing  
- Real-time prediction  
- Displays class label with confidence score  
- Uses the trained `.h5` model for inference  

---

## 🖥 How to Run

### 1️⃣ Install Requirements
```bash
pip install -r requirements.txt

2️⃣ Run the App

streamlit run app.py

📁 Project Structure

cats-dogs-classifier/
├── data/
│   ├── cats/
│   └── dogs/
├── models.py              # Training, preprocessing, model loading
├── app.py                 # Streamlit application
├── cat_dog_classifier.h5
├── requirements.txt
└── README.md

📊 Results Summary
Model Version	Validation Accuracy
Basic CNN	~70%
Improved CNN + Augmentation	~85%
Transfer Learning (MobileNetV2)	~95–97%
🧩 Future Improvements

    Fine-tuning deeper layers of MobileNet

    Model explainability (Grad-CAM)

    Better UI & confidence visualization

    Deployment to cloud platforms

🤝 Contributing

Feel free to fork this repository and experiment with:

    Different architectures

    Hyperparameter tuning

    Additional datasets

📜 License

This project is open-source and available under the MIT License.


If you want, I can also:
- Optimize this for **GitHub recruiters**
- Add **badges (accuracy, TensorFlow, Streamlit)**
- Rewrite it to match **FAANG-style ML project READMEs**
- Add **demo screenshots / GIF sections**

Just say the word.