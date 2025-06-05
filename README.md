# 💳 Credit Card Fraud Detection using Scikit-Learn & Snap ML

This project demonstrates how machine learning techniques can be applied to detect fraudulent credit card transactions. We explore the use of **Decision Tree** and **Support Vector Machine (SVM)** models, with steps covering data preprocessing, exploratory data analysis (EDA), class imbalance handling, model training, evaluation, and interpretation.

---

## 📁 Project Overview

- **Author**: Joseph Maina  
- **Dataset Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Tools & Libraries**: Python, Scikit-Learn, Snap ML, Pandas, NumPy, Seaborn, Matplotlib  
- **Techniques**: 
  - Data Cleaning & Preprocessing  
  - Exploratory Data Analysis (EDA)  
  - SMOTE for Class Imbalance  
  - Supervised ML (Decision Tree & SVM)  
  - Evaluation Metrics (Precision, Recall, F1-Score, ROC AUC)

---

## 📊 Dataset Information

- **Rows**: ~37,000  
- **Features**: 31 (28 anonymized PCA features, `Time`, `Amount`, and `Class`)  
- **Target**: `Class` → 0 (Normal), 1 (Fraud)  
- The dataset is highly **imbalanced**, with less than 1% fraudulent transactions.

---

## 🔍 Project Workflow

### Step 1: Setup
- Upload CSV file manually to Google Colab
- Import required libraries

### Step 2: Data Exploration
- Shape & structure of the dataset
- Null value analysis
- Column data types and memory usage
- Class distribution
- Transaction amount distribution visualization

### Step 3: Preprocessing
- Drop null values
- Normalize skewed `Amount` using log transformation
- Feature scaling using StandardScaler

### Step 4: Train-Test Split & Imbalance Handling
- Split dataset (70% train, 30% test)
- Apply **SMOTE** to balance training data

### Step 5: Model Training & Evaluation
- Train **Decision Tree** and **SVM** on SMOTE-balanced data
- Evaluate models with:
  - Accuracy, Precision, Recall, F1-Score
  - ROC Curve & AUC

### Step 6: Interpretation & Conclusion
- Compare model metrics
- Decision Tree slightly outperformed SVM in recall and AUC
- Suggestions for improvement: Ensemble models, hyperparameter tuning, deployment

---

## 📈 Results Summary

| Metric              | Decision Tree         | Support Vector Machine (SVM) |
|---------------------|------------------------|-------------------------------|
| Accuracy            | ✅ High                | ✅ High                        |
| Precision (fraud)   | 🔼 Better              | Good                          |
| Recall (fraud)      | ✅ Higher              | Slightly lower                |
| F1-Score (fraud)    | ✅ Balanced            | Balanced                      |
| ROC AUC             | **0.93**               | *~0.91*                       |

> 📌 *Recall is crucial in fraud detection — the Decision Tree had a slight edge.*

---

## 📌 Key Insights

- The dataset’s PCA-transformed features make it easier to apply models without worrying about multicollinearity.
- Handling class imbalance with SMOTE significantly improves model performance.
- Decision Trees are interpretable and quick to train, making them a good first model choice for fraud detection.

---

## 🚀 Future Improvements

- Try **Random Forest**, **XGBoost**, or **LightGBM**
- Apply **GridSearchCV** or **Optuna** for tuning
- Incorporate **real-time fraud detection** with streaming tools
- Build a **dashboard** for business insights

---

## 🧠 Project Motivation

Financial fraud is a growing concern. With millions of transactions happening every day, manually flagging fraud is impossible. This project showcases how **machine learning** can help solve this real-world problem efficiently and at scale.

---

## 📎 License

This project is for educational purposes and uses an open-source dataset.  
Dataset Credit: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 🙌 Acknowledgements

- Snap ML by IBM for fast model training
- Scikit-Learn for robust ML tooling
- Kaggle community for accessible datasets
