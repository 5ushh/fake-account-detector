# Fake Account Detector

NLP-based machine learning system to classify fake social media accounts, developed as part of a KSCST-funded research initiative.

## Problem

Fake accounts on social media platforms undermine trust, spread misinformation, and enable fraud. Manual detection doesn't scale to millions of accounts.

## Approach

- Analyzed 10,000+ social media account records
- Extracted NLP-based features from user bio, post patterns, and account metadata
- Trained and benchmarked multiple classification models
- Validated results using precision, recall, and ROC-AUC to handle class imbalance

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 84% |
| Evaluation | Precision, Recall, ROC-AUC |
| Dataset size | 10,000+ accounts |

## Tech Stack

Python · Scikit-learn · Pandas · NLP · REST APIs

## Project Structure

```
fake-account-detector/
├── data/               # Dataset (fake_accounts.csv)
├── scripts/
│   ├── train.py        # Model training pipeline
│   └── predict.py      # Inference script
└── models/             # Saved model (fake_acc_model.pkl)
```

## Setup

```bash
git clone https://github.com/5ushh/fake-account-detector
cd fake-account-detector
pip install pandas scikit-learn
python scripts/train.py
```

## Background

This project was funded by KSCST (Karnataka State Council for Science and Technology) as part of undergraduate research at New Horizon College of Engineering (2023).
