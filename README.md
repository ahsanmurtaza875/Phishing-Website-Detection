# Phishing Website Detection

A machine learning project that detects phishing websites from URL and web page features using Python and scikit-learn.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ahsanmurtaza875/Phishing-Website-Detection/blob/main/phishing_website_detection.ipynb)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

Phishing is a method of trying to gather personal information using deceptive e-mails and websites. This project trains classification models to tell whether a website is **phishing** or **legitimate**, based on features extracted from its URL and page content. The notebook also includes a simple URL checker.

## Project Workflow

```mermaid
flowchart LR
    A[Load CSV Data] --> B[Explore and Visualize]
    B --> C[Clean and Select Features]
    C --> D[Train / Test Split]
    D --> E[Train Models]
    E --> F[Evaluate Models]
    F --> G[URL Checker]
```

## Dataset

- **File:** `Phishing_Legitimate_full.csv`
- **Source:** Kaggle (add the link to the dataset page here)
- **Content:** 10,000 websites (5,000 phishing, 5,000 legitimate) with 48 extracted features
- **Target column:** `CLASS_LABEL` (1 = phishing, 0 = legitimate)

### Class Distribution

The dataset is balanced, so accuracy is a fair metric.

![Class Distribution](images/class_distribution.png)

## Exploratory Analysis

### Features Most Related to the Target

These features have the strongest correlation with whether a site is phishing.

![Top Correlated Features](images/top_correlations.png)

### Correlation Heatmap

![Correlation Heatmap](images/correlation_heatmap.png)

## Models and Results

Models trained: Random Forest, Decision Tree, and Logistic Regression.

![Model Comparison](images/model_comparison.png)

| Model | Accuracy |
|-------|----------|
| Random Forest | XX% |
| Decision Tree | XX% |
| Logistic Regression | XX% |

### Confusion Matrix (Best Model)

![Confusion Matrix](images/confusion_matrix.png)

### Most Important Features

![Feature Importance](images/feature_importance.png)

## How to Run

### Option 1: Google Colab (easiest)

1. Click the **Open in Colab** badge at the top of this page.
2. Click **Runtime → Run all**.

The notebook loads the dataset directly from this repository, so no manual upload is needed.

### Option 2: On your computer

```bash
git clone https://github.com/ahsanmurtaza875/Phishing-Website-Detection.git
cd Phishing-Website-Detection
pip install -r requirements.txt
jupyter notebook phishing_website_detection.ipynb
```

If running locally, change the data loading line in the notebook to:

```python
data = pd.read_csv("Phishing_Legitimate_full.csv")
```

## Repository Structure

```
Phishing-Website-Detection/
├── images/                            # Charts used in this README
├── phishing_website_detection.ipynb   # Main notebook
├── Phishing_Legitimate_full.csv       # Dataset
├── requirements.txt                   # Python libraries
├── README.md                          # Project description
├── LICENSE                            # MIT License
└── .gitignore
```

## Tech Stack

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn

## Author

**Ahsan Murtaza**
GitHub: [@ahsanmurtaza875](https://github.com/ahsanmurtaza875)
