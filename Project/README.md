# Sentiment Analysis of IMDB Movie Reviews  
### TF‑IDF + Logistic Regression vs. DistilBERT

This project implements and compares two sentiment classification models on a 10,000‑sample subset of the IMDB movie review dataset:

1. **Classical Baseline:** TF‑IDF + Logistic Regression  
2. **Transformer Model:** DistilBERT (fine‑tuned for 1 epoch)

The goal is to evaluate performance under **constrained training conditions** and analyze trade‑offs between classical and transformer‑based NLP approaches.

---

## 📌 Dataset
- Source: IMDB Movie Reviews  
- Total samples used: **10,000**  
- Balanced classes (positive/negative)  
- Split: **70% train / 15% validation / 15% test**  
- Avg review length: ~230 tokens  

---

## 📌 Models

### **1. Baseline: TF‑IDF + Logistic Regression**
- TF‑IDF with 20,000 features  
- 1–2 gram range  
- Logistic Regression with L2 regularization  
- Fast, interpretable, lightweight  

### **2. DistilBERT (Transformer)**
- 6‑layer distilled version of BERT  
- Fine‑tuned for **1 epoch**  
- Max sequence length: 128  
- Batch size: 8  
- Optimizer: AdamW (lr = 2e‑5)  

---

## 📊 Final Results (Updated)

### **Baseline (TF‑IDF + Logistic Regression)**
| Metric | Score |
|--------|--------|
| Accuracy | **0.8633** |
| Precision | **0.8639** |
| Recall | **0.8632** |
| F1‑score | **0.8632** |
| ROC‑AUC | **0.9458** |

---

### **DistilBERT (Fine‑tuned)**  
| Metric | Score |
|--------|--------|
| Accuracy | **0.8447** |
| Precision | **0.8448** |
| Recall | **0.8446** |
| F1‑score | **0.8446** |
| ROC‑AUC | **0.9261** |

---

## 🔢 DistilBERT Confusion Matrix (Corrected)

|               | Pred Neg | Pred Pos |
|---------------|----------|----------|
| **Actual Neg** | 622      | 122      |
| **Actual Pos** | 111      | 645      |

- TN = 622  
- FP = 122  
- FN = 111  
- TP = 645  

---

## 📈 Comparison Summary

| Model | Accuracy | F1 | ROC‑AUC |
|-------|----------|-----|---------|
| **TF‑IDF + LR** | 0.863 | 0.863 | 0.946 |
| **DistilBERT** | 0.845 | 0.845 | 0.926 |

**Observation:**  
Under limited training (1 epoch, 10k samples), the classical baseline slightly outperforms DistilBERT.  
However, DistilBERT shows stronger contextual understanding and would likely surpass the baseline with more epochs or more data.

---
# 🚀 How to Run This Notebook

### **1. Install dependencies**
### **2. Download the IMDB dataset**
Place `IMDB Dataset.csv` in the same folder as the notebook.

### **3. Run the notebook**

Run all cells **top to bottom**.  
The notebook will:

- Load the dataset  
- Train the baseline model  
- Train DistilBERT  
- Evaluate both models  
- Display metrics, confusion matrices, and comparison tables  

### **4. Notebook Output**
This notebook has been **fully executed** before upload.  
All output cells are visible directly on GitHub as required.
---
# ✔ Requirements
See `requirements.txt` for all dependencies.
---
# 📄 Report
The full ACM‑formatted report is included as:
---
# 🎯 Status
This repository contains:

- ✔ Fully executed Jupyter notebook  
- ✔ Markdown instructions  
- ✔ requirements.txt  
- ✔ ACM‑formatted report    
