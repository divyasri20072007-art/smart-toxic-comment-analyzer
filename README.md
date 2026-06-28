# 🛡️ ToxiShield AI
### Intelligent Toxic Comment Detection & Content Moderation System

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?logo=streamlit&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.8+-154F3C)
![License](https://img.shields.io/badge/License-MIT-06D6A0)

---

## 📖 Project Overview

**ToxiShield AI** is a production-ready NLP system that automatically detects and classifies toxic
comments into **7 categories** using machine learning. Built for real-world content moderation
pipelines, it combines TF-IDF text vectorisation with Logistic Regression and Multinomial
Naïve Bayes classifiers, evaluated side-by-side to select the best performer.

The interactive **Streamlit dashboard** provides real-time predictions, batch CSV moderation,
rich analytics, WordClouds, and downloadable Excel reports — all in a professional dark UI.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Real-Time Prediction** | Classify any comment with confidence score & risk level |
| 📁 **Batch CSV Moderation** | Upload thousands of comments and get predictions for every row |
| 🧹 **Preprocessing Demo** | Visualise every text-cleaning step interactively |
| 📊 **Dataset Analytics** | Class distributions, length histograms, word frequencies |
| ☁ **WordCloud** | Visualise frequent words in toxic vs non-toxic comments |
| 📈 **Model Evaluation** | Accuracy, F1, confusion matrix, CV score, side-by-side comparison |
| ⬇ **Export Reports** | Download predictions as CSV or colour-coded Excel workbook |

---

## 🏷️ Toxicity Categories

| # | Category | Risk |
|---|---|---|
| 0 | ✅ Non-Toxic | Low |
| 1 | ⚡ Toxic | High |
| 2 | 💀 Severe Toxic | Critical |
| 3 | 🔞 Obscene | High |
| 4 | ⚠️ Threat | Critical |
| 5 | 😠 Insult | Medium |
| 6 | 🚫 Identity Hate | Critical |

---

## 🧠 Tech Stack

```
Python 3.11+
├── Machine Learning
│   ├── scikit-learn  – Logistic Regression, Multinomial NB, TF-IDF
│   └── joblib        – Model persistence
├── NLP
│   └── NLTK          – Tokenisation, stopword removal, lemmatisation
├── Visualisation
│   ├── matplotlib / seaborn
│   └── wordcloud
├── Dashboard
│   └── Streamlit
└── Export
    ├── pandas
    └── openpyxl
```

---

## 📂 Dataset

**Jigsaw Toxic Comment Classification Challenge** (Kaggle)

- ~159,000 Wikipedia talk-page comments
- Six binary toxicity labels: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- Download: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

---

## 📁 Folder Structure

```
toxishield-ai/
├── app.py                # Streamlit dashboard (main entry point)
├── preprocessing.py      # Full text-cleaning pipeline
├── train_model.py        # Model training & evaluation
├── predict.py            # Single & batch inference
├── evaluation.py         # Metrics, charts, confusion matrix
├── utils.py              # EDA charts, WordCloud, export helpers
├── requirements.txt
├── README.md
├── LICENSE
├── data/
│   └── train.csv         # ← Place Jigsaw dataset here
└── models/               # ← Auto-created on training
    ├── model.pkl
    ├── tfidf.pkl
    └── model_meta.pkl
```

---

## 🚀 Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/toxishield-ai.git
cd toxishield-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download `train.csv` from Kaggle (link above) and place it in the `data/` folder.

### 4. Train the model

```bash
python train_model.py --data data/train.csv
```

This will:
- Preprocess the dataset
- Train Logistic Regression and Multinomial Naïve Bayes
- Compare both models and select the best
- Save `models/model.pkl`, `models/tfidf.pkl`, `models/model_meta.pkl`

### 5. Launch the dashboard

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | ~0.92 | ~0.91 | ~0.92 | ~0.91 |
| Multinomial Naïve Bayes | ~0.88 | ~0.87 | ~0.88 | ~0.87 |

*Exact figures depend on train/test split and dataset version.*

---

## 🔮 Future Improvements

- [ ] Fine-tuned BERT / DistilBERT for higher accuracy
- [ ] Multilingual toxicity detection
- [ ] FastAPI REST endpoint for integration
- [ ] Docker containerisation
- [ ] Real-time WebSocket moderation feed
- [ ] Active learning feedback loop from moderators
- [ ] Explainability with SHAP / LIME

---

## 📄 License

This project is licensed under the **MIT License** – see [LICENSE](LICENSE) for details.

---

