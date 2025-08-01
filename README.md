# 📈 Customer Churn Prediction System (Streamlit)

This project is an end-to-end **Customer Churn Prediction System** built with **Streamlit** and pre-loaded with the **Telco Customer Churn** CSV so you can explore insights out-of-the-box—no extra data required.

## 🚀 Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Train the models** (one-time step)
   ```bash
   python model_training.py
   ```
3. **Launch the dashboard**
   ```bash
   streamlit run app.py
   ```
   The app opens automatically at `http://localhost:8501`.

## 📂 Project Layout

```
├── Telco-Customer-Churn.csv   # Ready-to-use dataset (7,043 rows, 21 cols)
├── model_training.py          # ML preprocessing + training pipeline
├── app.py                     # Streamlit dashboard
├── requirements.txt           # All Python dependencies
└── README.md                  # Project guide (this file)
```

## 🔍 What’s Inside the Dashboard

| Section | What You Get |
| ------- | ------------ |
| Overview | KPIs: total customers, churned customers, churn rate |
| Segmentation | Churn by contract type, tenure, internet service |
| Charges Analysis | Monthly charges vs tenure scatter, interactive filters |
| Model Predictions | Select any customer ➜ churn probability & risk level |
| Feature Importance | Top predictive drivers for tree-based models |

## 🧠 Machine-Learning Pipeline

* **Preprocessing**: missing values, label-encoding, engineered features (`tenure_group`, `avg_monthly_charges`, `charges_per_service`).
* **Models**: Logistic Regression (scaled), Random Forest, XGBoost.
* **Evaluation**: accuracy & ROC-AUC; best model chosen automatically.
* **Artifacts saved**: `*_model.pkl`, `scaler.pkl`, `label_encoders.pkl`, `feature_names.pkl` (all reused by the app).

## 🔗 Dataset Reference

Synthetic recreation of IBM Telco Customer Churn: 7,043 customers, 21 columns, 32.7% churn rate. (Generated locally—no external download required.)

## 💡 Business Value

Identifies high-risk customers, uncovers churn drivers, and arms retention teams with actionable insights—all via a single, no-code dashboard.

Enjoy exploring! 🎉
