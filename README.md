# RetentionIQ
### Customer Retention Intelligence Platform

A machine learning powered customer churn prediction dashboard built using Python, Streamlit, Scikit-Learn and Plotly.

---

## Live Demo

🔗 https://retentioniq-cripproj1.streamlit.app/

---

## 📌 Problem Statement

Customer churn is one of the biggest challenges for subscription-based businesses.

RetentionIQ helps companies:

- Predict customers likely to churn
- Understand churn behavior
- Visualize customer insights
- Generate retention recommendations
- Compare model performance

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Plotly
- Joblib

---

## 📊 Features

### Dashboard
- Customer KPIs
- Churn Rate
- Revenue Analytics
- Average Tenure

### Analytics
- Churn Distribution
- Contract Analysis
- Monthly Charges Analysis
- Tenure Analysis

### Churn Predictor
- Real-time predictions
- Churn probability score
- Risk classification

### Retention Insights
- AI-inspired retention recommendations
- Customer engagement strategies

### Model Performance
- Accuracy comparison
- Precision / Recall metrics
- Model benchmarking

---

## 📈 Machine Learning Model

Model Used:

- Random Forest Classifier

Performance:

| Metric | Score |
|----------|----------|
| Train Accuracy | 87.48% |
| Test Accuracy | 79.46% |
| Precision | 79% |
| Recall | 79% |
| F1 Score | 79% |

---

## 📂 Dataset

Telco Customer Churn Dataset

Contains:

- Customer demographics
- Services subscribed
- Contract information
- Monthly charges
- Tenure
- Churn status

---

## 🏗 Project Structure

```text
CRIP/
│
├── app.py
├── train.py
├── requirements.txt
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── models/
│   ├── churn_model.joblib
│   └── feature_columns.joblib
│
├── src/
│   ├── recommendations.py
│   └── feature_importance.py
```

---

## ⚙ Installation

Clone repository

```bash
git clone https://github.com/YOUR_USERNAME/RetentionIQ.git
```

Move into project

```bash
cd RetentionIQ
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run app

```bash
streamlit run app.py
```

---

## 👨‍💻 Author

Built by Vikanshu Sharma(@r4gefr)

Aspiring Data Scientist | Machine Learning Engineer | Python Developer
