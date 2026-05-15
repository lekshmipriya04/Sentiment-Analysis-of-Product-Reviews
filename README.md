# 💬 Sentiment Analysis of Product Reviews

> An end-to-end NLP pipeline for three-class sentiment classification (Positive · Neutral · Negative) — trained on Amazon Product Reviews and deployed as a YouTube Comment Sentiment Analyzer via Streamlit.

---

## 🧾 Project Overview

This project was developed as part of the **Predictive Analytics** course at **Digital University Kerala**. It implements a complete end-to-end sentiment analysis pipeline — from raw text preprocessing through feature engineering, model training, and multi-model evaluation, to a fully interactive Streamlit web application for real-time sentiment analysis of YouTube video comments.

The classifier assigns one of three sentiment labels — **Positive**, **Neutral**, or **Negative** — to input text using a hybrid TF-IDF + linguistic feature pipeline combined with a high-performance machine learning classifier.

---

## 🎯 Problem Statement & Motivation

In today's digital economy, customer opinions expressed through product reviews and social media comments are a goldmine of insight for businesses, creators, and consumers. Manually reading thousands of reviews or video comments to gauge public opinion is impractical. Automated sentiment analysis enables brands to monitor reception in real time, creators to understand audience reactions, and researchers to study opinion trends at scale.

This project addresses the problem of **automated three-class sentiment classification** on product reviews, then extends the trained model to classify live YouTube comments — bridging traditional NLP with a real-world social media use case.

---

## 🎯 Objectives

- Build a robust three-class (Positive / Neutral / Negative) sentiment classification pipeline using TF-IDF and linguistic features
- Compare four classifiers: Logistic Regression, Multinomial Naive Bayes, Linear SVM, and Gradient Boosting
- Select the best model by Macro F1-Score for deployment
- Deploy the model as an interactive Streamlit app that fetches and classifies live YouTube comments via the YouTube Data API
- Provide rich visualisation of sentiment distribution, comment length patterns, and top keywords

---

## 📁 Repository Structure

```
├── app.py                  # Streamlit web application (YouTube Comment Analyzer)
├── train.py                # Model training and evaluation script
├── model.py                # ModelTrainer class — all 4 classifiers + evaluation
├── features.py             # Feature engineering — TF-IDF + linguistic features
├── preprocessing.py        # Text cleaning and normalisation pipeline
├── predict.py              # SentimentPredictor class for inference
├── youtube_api.py          # YouTube Data API integration (comment fetching)
├── pipeline.pkl            # Trained and saved final model pipeline
├── requirements.txt        # Project dependencies
├── visuals/
│   ├── class_dist.png          # Class distribution chart
│   ├── review_len_dist.png     # Review length distribution
│   ├── top_features.png        # Top 20 TF-IDF features
│   ├── model_comparison.png    # Macro F1 comparison across models
│   └── confusion_matrices.png  # Side-by-side confusion matrices
├── individual_files/       # Team member individual contribution files
└── README.md
```

---

## 📊 Dataset

| Property | Detail |
|---|---|
| **Source** | [Amazon Product Reviews — Kaggle](https://www.kaggle.com/) |
| **Task** | Three-class sentiment classification |
| **Classes** | Positive · Neutral · Negative |
| **Features** | TF-IDF vectors + Linguistic features (hybrid pipeline) |
| **Format** | Text reviews with star ratings mapped to sentiment labels |
| **Inference Input** | Live YouTube comments via YouTube Data API (up to 200 per video) |

---

## 🔧 Methodology

### 1. Text Preprocessing (`preprocessing.py`)

Each raw review is processed through a multi-step cleaning pipeline:

- **Lowercasing** — normalises case
- **URL and mention removal** — strips hyperlinks and @mentions
- **Punctuation removal**
- **Stopword removal** — using NLTK English stopword list
- **Lemmatisation** — reduces words to their base form

### 2. Feature Engineering (`features.py`)

A **hybrid feature pipeline** combines two feature types:

- **TF-IDF Vectoriser** — captures term frequency-inverse document frequency weights for vocabulary-based features
- **Linguistic Features** — handcrafted features such as review length, exclamation/question mark counts, and other stylistic signals
- Both feature sets are concatenated via `FeatureUnion` into a single input matrix

### 3. Model Training & Comparison (`model.py`)

Four classifiers were trained and evaluated using **Macro F1-Score** as the selection criterion:

| Model | Notes |
|---|---|
| **Logistic Regression ★** | Best overall — selected for deployment |
| Multinomial Naive Bayes | Fast, strong baseline for text tasks |
| Linear SVM (LinearSVC) | High-precision linear classifier |
| Gradient Boosting | Tree-based ensemble, strong on tabular features |

The best model by Macro F1-Score is automatically saved as `pipeline.pkl` for deployment.

### 4. Evaluation (`visuals/`)

The training script generates the following evaluation artefacts:

- **Class distribution chart** — detects class imbalance
- **Review length distribution** — analyses text variability across sentiment classes
- **Top 20 TF-IDF features** — proves the model learns semantic meaning
- **Model comparison chart** — Macro F1 comparison across all 4 models
- **Confusion matrices** — side-by-side heatmaps showing per-class errors for all models

---

## 📈 Results Summary

| Model | Evaluation Metric | Notes |
|---|---|---|
| **Logistic Regression ★** | **Best Macro F1** | Selected for production deployment |
| Linear SVM | Strong F1 | Close competitor to Logistic Regression |
| Multinomial Naive Bayes | Good baseline | Slightly lower on Neutral class |
| Gradient Boosting | Competitive | Slower to train |

> 📌 *Detailed per-class precision, recall, F1 scores, and confusion matrices are available in the `visuals/` folder and the Training Metrics tab of the Streamlit app.*

---

## 🖥️ Web Application

The project ships with a **Streamlit** web app (`app.py`) that goes beyond static review classification — it connects to the **YouTube Data API** to fetch and analyse live comments from any public YouTube video in real time.

### Features

- **Live YouTube comment analysis** — paste any YouTube URL to fetch up to 200 comments and classify them instantly
- **Overall opinion summary** — auto-generates a verdict (Highly Positive / Mixed / Negative Feedback Trend)
- **Sentiment distribution pie chart** — visualises Positive / Neutral / Negative breakdown
- **Comment length vs sentiment boxplot** — reveals whether negative commenters write longer rants
- **Top keyword bar charts** — most frequent words in positive vs negative comments side by side
- **Sample comments viewer** — tabbed view of representative Positive, Neutral, and Negative comments
- **Training Metrics tab** — embedded visualisations from the training phase (class distribution, model comparison, confusion matrices)

### Screenshots

**Home — Input Interface**
![App Home](https://github.com/user-attachments/assets/8c595694-faab-4bc0-8918-5b4145dce126)


**Sentiment Results**
![Sentiment Results](https://github.com/user-attachments/assets/901aaab6-7da7-4b1f-b3be-d8d76f16145b)


**Positive  Comments Detection**
![Positive  Comments Detection](https://github.com/user-attachments/assets/832a9f78-f749-453a-be38-7724cba9cdea)


**There were 0 Neutral Comments**

**Negative Comments Detection**
![Negative Comments Detection](https://github.com/user-attachments/assets/beef0dd5-4835-4ed5-814b-025dfcf56f99)


> 🔗 **Live Demo:** 

https://sentiment-analysis-of-appuct-reviews-wcvcg2zjkk3o8vb65ilejy.streamlit.app/

---

## ⚙️ Setup & Running Locally

**1. Clone the repository**

```bash
git clone https://github.com/lekshmipriya04/Sentiment-Analysis-of-Product-Reviews.git
cd Sentiment-Analysis-of-Product-Reviews
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Train the model**

```bash
python train.py
```

This generates `pipeline.pkl` and all charts under `visuals/`.

**4. Launch the Streamlit app**

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

**5. Using the app**

- Enter your **YouTube Data API key** in the sidebar (get one free from [Google Cloud Console](https://console.cloud.google.com/))
- Paste any public **YouTube video URL**
- Click **Analyze Sentiments**

---

## 📦 Dependencies

```
streamlit
scikit-learn
pandas
numpy
matplotlib
seaborn
nltk
Pillow
google-api-python-client
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 🏫 Academic Context

| Field | Detail |
|---|---|
| **Institution** | Digital University Kerala |
| **Course** | Predictive Analytics |
| **Topic** | Sentiment Analysis of Product Reviews Using NLP and Machine Learning |

---

## 👥 Team Members

| Name |
|---|
| *M. R. Lekshmi Priya* |
| *Mohammed Yazin N* |
| *Rehan Biju* |

---

## 📄 License

This project is intended for academic and educational purposes.
