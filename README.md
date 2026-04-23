# Sentiment Analysis of IMDB Movie Reviews  
### Baseline vs Transformer Models (DistilBERT)

## 📌 Project Overview
This project performs binary sentiment classification (positive/negative) on the IMDB 50K movie review dataset.  
We compare a classical baseline model (TF‑IDF + Logistic Regression) with a modern transformer model (DistilBERT).

## 📂 Dataset
- Source: Kaggle IMDB 50K Movie Reviews  
- Labels: `positive` / `negative`  
- Preprocessing: text cleaning + label encoding  
- Split: **70% train / 15% validation / 15% test**

## 🧠 Models Implemented
### Baseline
- TF‑IDF Vectorizer (20k features, 1–2 grams)
- Logistic Regression (max_iter=1000)

### Transformer Model
- DistilBERT (HuggingFace Transformers)
- Fine‑tuned for sequence classification

## 🏋️ Training
- Optimizer: AdamW  
- Learning rate: 2e‑5  
- Epochs: 1 (for Colab efficiency)  
- Batch size: 8  
- Max sequence length: 128  

## 📊 Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1‑macro  
- ROC‑AUC  
- Confusion Matrix  

## 📈 Results Summary
| Model | Accuracy | F1‑macro | ROC‑AUC |
|-------|----------|----------|---------|
| TF‑IDF + Logistic Regression | ~0.86–0.88 | ~0.86 | ~0.90 |
| DistilBERT | ~0.90–0.93 | ~0.91 | ~0.95 |

## ▶️ How to Run
1. Upload `IMDB Dataset.csv` to the notebook directory  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
