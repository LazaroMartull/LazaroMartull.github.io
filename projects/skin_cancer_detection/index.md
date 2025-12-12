# Skin Cancer Detection Using Deep Learning (ResNet50)

This project builds a deep learning model to classify skin lesions as **benign or malignant** using **transfer learning with ResNet50** and custom convolutional layers.  
The goal is to evaluate whether smartphone-quality images can support early detection of melanoma.

---

## 🧩 My Role

I contributed to:

- Dataset organization & preprocessing  
- Transfer-learning model design  
- Hyperparameter tuning and training  
- Plotting accuracy/loss curves  
- Writing model evaluation & interpretation  
- Documenting the workflow  

---

## 🛠 Tools & Technologies

- Python, TensorFlow/Keras  
- ResNet50 pretrained weights  
- Matplotlib for visualization  
- ISIC dermatology datasets  
- Google Colab / Jupyter Notebook  

---

## 📄 Full Report & Code

This project folder contains:

- 📘 Final PDF Report  
- 🧠 Python training scripts  
- 🧪 Model checkpoints / metrics (optional)  
- 📁 Dataset preprocessing utilities  

---

## 🔍 Project Overview

Skin cancer affects millions worldwide, and early detection dramatically improves survival rates.  
This project explores whether deep learning can help classify skin lesions using dermatoscopic images.

Using the **ISIC 2018 & 2024 datasets**, a transfer-learning pipeline was built using:

- **ResNet50** as a frozen feature extractor  
- Custom VGG-style dense layers  
- Data augmentation  
- 224×224 RGB image preprocessing  

📌 **Final performance:**

- **Accuracy:** 86.8%  
- **Malignant Precision:** 71.6%  
- **Malignant Recall:** 36.8%  
- **Benign Recall:** strong and stable  

---

## 📁 Dataset

Images were gathered from:

- **ISIC 2018 Challenge Dataset**  
- **ISIC 2024 Archive**  

Preprocessing steps included:

- Resize to 224×224  
- Normalize and batch  
- Augment (flip, rotation, zoom)  
- 80/20 train-validation split  

---

## 🧪 Preprocessing, Architecture & Training Code

```python
import tensorflow as tf

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

# ------------------ Model Architecture ------------------

base_model = tf.keras.applications.ResNet50(
    include_top=False,
    input_shape=(224, 224, 3),
    weights="imagenet",
    pooling="avg"
)

for layer in base_model.layers:
    layer.trainable = False

from tensorflow.keras.layers import Dense, Flatten

model = tf.keras.Sequential([
    base_model,
    Flatten(),
    Dense(64, activation="relu"),
    Dense(2, activation="softmax")
])

# ------------------ Training Configuration ------------------

from tensorflow.keras.optimizers import AdamW
from tensorflow.keras import callbacks

model.compile(
    optimizer=AdamW(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

earlystopping = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_set,
    validation_data=validation_set,
    epochs=100,
    callbacks=[earlystopping]
)
```

---

## 📊 Results & Evaluation

Key observations:

- **Strong benign classification performance**  
- **Lower malignant recall**, likely due to class imbalance  
- Training stabilized early because ResNet50 was frozen  
- Adding more malignant samples or fine-tuning top ResNet layers could improve recall  

---

## 📈 Accuracy & Loss Curves

The training logs show:

- Validation accuracy plateaued early  
- Loss stabilized with little overfitting  
- Early stopping correctly restored best weights  

(Plots included in the project report.)

---

## 🚀 Future Improvements

To enhance model performance:

- Fine-tune upper ResNet50 layers  
- Add more malignant images  
- Test EfficientNet, MobileNet, or ViT models  
- Use metadata (age, lesion area) for multimodal learning  
- Convert to TensorFlow Lite for mobile deployment  

---

## 📦 Folder Structure

```
skin_cancer_detection/
│── index.md            # Project write-up (this page)
│── report.pdf          # Final project report
│── training_code.py    # Clean training script (ResNet50 training)
```

---

## 🎯 Summary

This project demonstrates the potential of deep learning to support dermatologists and mobile health apps by providing early classification of skin lesions.  
While not a substitute for clinical diagnosis, the model shows promising accuracy and provides a foundation for future medical AI research.
