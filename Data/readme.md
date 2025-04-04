# Combined Hate Speech Dataset

This dataset is a unified collection of three well-known hate speech detection datasets:
- **Davidson Dataset** (CSV)
- **HateXplain** (JSON)
- **Gab Hate Corpus** (TSV)

It is intended to support research in hate speech detection, misinformation analysis, and explainability.

## 📁 File Structure

**File:** `combined_hate_speech_dataset.csv`

| Column   | Description                                                                 |
|----------|-----------------------------------------------------------------------------|
| `text`   | The textual content of the post, tweet, or comment.                         |
| `label`  | The normalized class label: `"hate"`, `"offensive"`, or `"normal"`.        |
| `source` | Origin of the data: `"davidson"`, `"hatexplain"`, or `"gab"`.              |

---

## 🧠 Label Definitions

| Label       | Description                                                                 |
|-------------|-----------------------------------------------------------------------------|
| `hate`      | Contains explicit hate speech targeting individuals or groups.              |
| `offensive` | Contains offensive language, insults, or rude speech without explicit hate. |
| `normal`    | Neutral or non-harmful speech.                                              |

Note: Not all datasets included all three labels. Gab dataset, for instance, only distinguishes between hate and normal.

---

## 📦 Source Dataset Descriptions

### 1. Davidson et al. (2017)
- Format: CSV
- Labels: Hate speech (0), Offensive (1), Normal (2)
- Used majority-vote from multiple annotators.

### 2. HateXplain (2021)
- Format: JSON
- Labels: `hatespeech`, `offensive`, `normal`
- Includes multi-annotator rationales, targets, and explanations.

### 3. Gab Hate Corpus
- Format: TSV
- Annotated with multiple binary hate categories.
- Aggregated hate label extracted for consistency.

---

## 🛠 How It Was Built

1. Loaded each dataset into a structured format.
2. Normalized all label sets to a unified schema (`hate`, `offensive`, `normal`).
3. Aggregated annotations if multiple annotators existed.
4. Retained only the `text`, `label`, and `source` columns for simplicity.

---

## 💡 Usage

This dataset is suitable for:
- Binary or multiclass hate speech detection tasks.
- Fine-tuning transformer models (e.g., BERT, RoBERTa).
- Evaluating generalization across platforms and annotation schemes.
- Research in fairness, explainability, and misinformation in NLP.

---

## ⚠️ Disclaimer

This dataset contains explicit and offensive language. It is intended strictly for academic research and must be handled responsibly.

---

## 📜 Citation

If you use this dataset, please cite the original sources:
- Davidson et al., 2017
- Mathew et al., 2021 (HateXplain)
- Kennedy et al., 2020 (Gab Hate Corpus)
