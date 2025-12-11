# Skin Cancer Detection Using Deep Learning (ResNet50)

This project builds a deep learning model to classify skin lesions as **benign or malignant** using **transfer learning with ResNet50** and custom convolutional layers.  
The goal is to evaluate whether smartphone-quality images can support early detection of melanoma.

---

## 🔍 Project Overview

Skin cancer affects millions of people worldwide, and early detection dramatically improves survival rates.  
This project explores whether deep learning can help classify skin lesions using publicly available dermatoscopic images.

Using the **ISIC 2018 & 2024 datasets**, a transfer-learning pipeline was developed using:

- **ResNet50** (frozen feature extractor)  
- Custom **VGG-style convolutional layers**  
- **Data augmentation** & **balanced training**  
- Training on 224×224 RGB images  

📌 **Final performance (top model):**

- **Accuracy:** 86.8%  
- **Malignant Recall:** 36.8% (limited by class imbalance)  
- **Malignant Precision:** 71.6%  
- **Benign Precision/Recall:** strong and stable  

---

## 📁 Dataset

The model was trained on dermatoscopic lesion images from:

- **ISIC 2018 Challenge Dataset**  
- **ISIC Archive (2024)**  

Preprocessing included:

- Resizing to **224×224**  
- RGB normalization  
- Data augmentation (flip, rotation, zoom)  
- Training/validation split of 80/20  

---

## 🧪 Preprocessing & Augmentation

```python
height, width = 224, 224

train_set = tf.keras.preprocessing.image_dataset_from_directory(
    "content/image_data/xs_train",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(height, width),
    batch_size=10
)

validation_set = tf.keras.preprocessing.image_dataset_from_directory(
    "content/image_data/xs_train",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(height, width),
    batch_size=4
)
