# Skin Disease Classification using Deep Learning (HAM10000)

A deep learning–based skin lesion classifier trained on the **HAM10000** dataset, wrapped in a Flask web app. It demonstrates CNNs and transfer learning for multi-class medical image classification.

> ⚠️ **Disclaimer:** This project is for **educational and research purposes only**. It must not be used for real-world medical diagnosis or clinical decisions.

## 📌 Project Overview
- Image-based classification of skin lesions
- Multi-class prediction across 7 dermatological categories
- Showcases model training, evaluation and a full ML → web workflow
- Handles class imbalance with class weighting

## 🗂 Dataset
- HAM10000 (Human Against Machine with 10,000 training images), publicly available
- 7 classes: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`
- 📁 Not included in this repo (size). Place it in `data/` (see Training).

## 🧠 Model
- Base: MobileNet (ImageNet weights), transfer learning
- Phase 1: train a custom head; Phase 2: fine-tune top layers
- Class weights for imbalance; lesion-level train/val/test split (no leakage)

## 📊 Evaluation (reported)
- Accuracy ~62%, weighted F1 ~0.66, macro F1 ~0.42
- Confusion matrix, training curves and per-class report in `outputs/`

## 🛠 Tech Stack
Python · Flask · TensorFlow/Keras · OpenCV · SQLite · FPDF · Matplotlib/Seaborn

## 🚀 Run the web app
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# optional: AI chat via OpenRouter (a basic offline fallback exists)
export OPENROUTER_API_KEY=your_key
export SECRET_KEY=something-random

python app.py                   # open http://127.0.0.1:5000
```
The site runs without a model; image analysis needs `model/skin_model.h5` (created by training).

## 🏋️ Train the model
1. Download HAM10000 from Kaggle.
2. Arrange:
   ```
   data/HAM10000_images/ISIC_xxxxxxx.jpg
   data/HAM10000_metadata.csv
   ```
3. Run `python training/train_cnn_improved.py`
4. It writes `model/skin_model.h5` and the four plots into `outputs/`.

## 📁 Structure
```
skin-disease-detection/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── static/        (style.css, script.js, diya_shukla.jpg, uploads/)
├── templates/     (base, index, result, reports, chat, appointments, clinics, tips, contact, about)
├── outputs/       (accuracy_curve, loss_curve, confusion_matrix, classification_report)
└── training/train_cnn_improved.py
```

## 👤 Author
Diya Shukla — Deep Learning & AI/ML Enthusiast

## 📄 License
Released strictly for educational and research use.
