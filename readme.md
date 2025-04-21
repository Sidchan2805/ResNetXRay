# 🩺 Pneumonia Detection from Chest X-rays using ResNet18

This project uses a deep learning model (**ResNet18**) to classify **chest X-ray images** into one of three categories:

- Pneumonia
- Other Lung Disease
- Normal

The model was trained using transfer learning on the NIH ChestX-ray14 dataset after careful data curation and preprocessing.

---

## 🧠 Project Overview

Pneumonia detection through chest X-rays is a common diagnostic task in medicine. However, interpreting X-rays manually is time-consuming and can be error-prone. This project automates that process using deep learning — helping reduce diagnostic overhead.

---

## 🧰 Tech Stack

| Tool/Library     | Role |
|------------------|------|
| `PyTorch`        | Deep learning framework |
| `Torchvision`    | Pretrained models (ResNet18), image transformations |
| `ResNet18`       | CNN backbone for classification |
| `Scikit-learn`   | Evaluation metrics |
| `Google Colab`   | Training and experimentation |
| `Pillow`         | Image loading for inference |
| `NIH ChestX-ray14` | Real chest X-ray dataset with 100k+ images |

---

## 🧪 Dataset Preprocessing & Curation

1. **Original Dataset**: NIH ChestX-ray14 contains over 112,000 images with 14 disease labels (multi-label format).
2. **Filtered Labels**:
   - Images labeled as **Pneumonia**
   - Images with **no finding** (used as Normal class)
   - Remaining images grouped under **Other diseases**
3. **Created 3-Class Classification Task**:
   - `Pneumonia`: Only pneumonia-labeled images
   - `Normal`: Only "No Finding" labeled images
   - `Other`: All other diseases grouped together
4. **Balanced Sample Counts**:
   - Stratified sampling to get ~700 images per class for training
   - ~150 per class for validation and testing
5. **Directory Structure After Preprocessing**:

   -data/ ├── train/ │ ├── class_0_pneumonia/ │ ├── class_1_other/ │ └── class_2_normal/ ├── val/ ├── test/

6. **Image Format**:
   - All images resized to 224×224
   - Normalized using ImageNet mean & std for compatibility with pretrained ResNet18