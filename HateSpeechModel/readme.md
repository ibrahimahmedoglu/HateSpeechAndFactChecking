# Hate Speech Detection Model (RoBERTa)

This repository contains a fine-tuned transformer-based model for detecting hate speech, offensive language, and neutral (normal) content. The model is trained on a combined dataset aggregated from Davidson, HateXplain, and Gab Hate Corpus sources.

---

## 🔍 Problem Statement

The aim of this model is to accurately classify a piece of text into one of three categories:

- `hate`: Explicit hate speech targeting a group or individual.
- `offensive`: Rude or aggressive language without explicit hate.
- `normal`: Neutral, non-offensive text.

---

## 📁 Dataset

The training dataset is a **combined version** of three open-source hate speech datasets:

- **Davidson Dataset**
- **HateXplain**
- **Gab Hate Corpus**

A unified label schema was used: `hate`, `offensive`, and `normal`. All samples were normalized, and labels encoded to integer classes (`0`, `1`, `2`).

For performance testing, only **10% of the full dataset** was used to reduce computation and fit memory constraints (especially on M1 Macs).

---

## 🧠 Model Architecture

- **Base model:** `roberta-base` (by Hugging Face)
- **Classification head:** Linear layer for 3-way classification
- **Tokenizer:** `RobertaTokenizer` with truncation (`max_length=128`)

---

## 🏋️ Training Configuration

| Setting                     | Value               |
|----------------------------|---------------------|
| Model                      | RoBERTa (base)      |
| Batch Size                 | 2                   |
| Max Sequence Length        | 128 tokens          |
| Epochs                     | 3                   |
| Optimizer                  | AdamW               |
| Learning Rate              | 2e-5                |
| Evaluation Strategy        | Epoch               |
| Dataset Used               | 10% sampled dataset |
| Hardware                   | MacBook M1 / Colab  |

---

## 📊 Results (10% of Dataset)

After training on 10% of the combined dataset (667 samples in the test set), the model achieved:

### **Classification Report**

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Hate (0)    | 0.59      | 0.46   | 0.52     | 93      |
| Offensive (1)| 0.84     | 0.89   | 0.86     | 256     |
| Normal (2)  | 0.81      | 0.82   | 0.81     | 318     |

**Overall Accuracy:** 0.80  
**Macro F1 Score:** 0.73  
**Weighted F1 Score:** 0.79

✅ The model performs strongly on `offensive` and `normal` classes. The `hate` class shows room for improvement — possibly due to class imbalance in the reduced dataset.

---

## 🧪 How to Run

### 1. Install dependencies

```bash
pip install transformers datasets scikit-learn pandas
