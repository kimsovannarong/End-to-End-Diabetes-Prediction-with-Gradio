# 🩺 Diabetes Risk Prediction & Interactive ML App

An end-to-end Machine Learning pipeline and interactive web interface that cleans diagnostic health data, performs exploratory data analysis (EDA), evaluates predictive models (including Logistic Regression and ensemble classifiers), and serves real-time risk predictions via an interactive **Gradio** UI.

---

## 📌 Project Overview

Early detection of diabetes can significantly improve patient outcomes and healthcare management. This project delivers a complete data science workflow:
1. **Data Preprocessing & Cleaning:** Handling missing values, outlier detection, and feature scaling.
2. **Exploratory Data Analysis (EDA):** Visualizing correlations between clinical factors (e.g., Glucose, BMI, Age, Insulin) and diabetes onset.
3. **Model Training & Evaluation:** Comparing multiple classification models (Logistic Regression, Random Forest, SVM) to select the best performer based on Accuracy, Precision, Recall, and ROC-AUC metrics.
4. **Interactive Web Deployment:** Building a user-friendly Gradio app allowing users and clinical workers to input patient parameters and view instant risk predictions.

---
## 📊 Dataset Overview

This project uses the **Pima Indians Diabetes Dataset** provided by the *National Institute of Diabetes and Digestive and Kidney Diseases*.

* **Objective:** Predict whether a patient will develop diabetes using binary classification.
* **Target Population:** Females aged 21 and older of Pima Indian heritage.
* **Target Variable:** `Outcome` — `1` (Diabetic) or `0` (Non-Diabetic).

### Key Features & Medical Metrics

| Feature | Description | Metric / Unit |
| :--- | :--- | :--- |
| **Pregnancies** | Number of times pregnant | Count |
| **Glucose** | Plasma glucose concentration (2-hour oral glucose tolerance test) | mg/dL |
| **Blood Pressure** | Diastolic blood pressure | mm Hg |
| **Skin Thickness** | Triceps skinfold thickness | mm |
| **Insulin** | 2-Hour serum insulin | mu U/ml |
| **BMI** | Body mass index | $\text{kg/m}^2$ |
| **Diabetes Pedigree Function** | Genetic likelihood score based on family history | Numerical Score |
| **Age** | Age of the patient | Years |

## 🚀 Key Features

* **🧹 Comprehensive Data Wrangling:** Handled missing/invalid zero values across clinical metrics and standardized feature ranges using `StandardScaler`.
* **📊 Visual Exploratory Analysis:** Plotted distribution curves, feature importance charts, and correlation heatmaps to extract key biological indicators.
* **🤖 Multi-Model Evaluation:** Trained and evaluated several supervised classification algorithms (Logistic Regression, Decision Trees, Random Forest, etc.) to optimize for high recall and precision.
* **🌐 Web User Interface (Gradio):** Built a web interface enabling real-time input sliders/fields and generating probability scores alongside instant diagnostic output.

---

## 🛠️ Data Preparation Methodology

To ensure high model performance and reliable real-world predictions, the raw dataset underwent a rigorous data preprocessing pipeline:

* **🧹 Duplicate Verification:** Verified the dataset to confirm zero duplicate records were present, avoiding data leakage during model training.
  ```python
  # Check and drop duplicate rows
  df = df.drop_duplicates()

* **🔍 Invalid & Null Value Handling: Identified invalid zero values in biological columns where zeros are physiologically impossible (e.g., Glucose, Blood Pressure, BMI, Insulin, Skin Thickness) and imputed them using statistical medians.
   ```python
  # Replace invalid 0s with column medians
  zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
  for col in zero_cols:
  df[col] = df[col].replace(0, df[col].median())

* **🎯 Feature Selection: Evaluated feature importance and correlation matrices to select key diagnostic predictors, eliminating non-informative features to reduce computational complexity and prevent overfitting.
  ```python
  # Correlation-based feature selection
  correlation = df.corr()['Outcome'].abs().sort_values(ascending=False)
  selected_features = correlation[correlation > 0.1].index

  X = df[selected_features].drop('Outcome', axis=1)
  y = df['Outcome']# Replace invalid 0s with column medians

* **⚖️ Data Splitting & Class Balancing: Performed stratified train-test splitting (train_test_split with stratify=y) to maintain identical target class proportions (diabetic vs. non-diabetic) across both training and testing subsets.
  ```python
  from sklearn.model_selection import train_test_split

  # Stratified split to preserve class distribution
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
  )
## 🛠️ Tech Stack & Libraries

* **Language:** Python
* **Data Processing & EDA:** Pandas, NumPy, Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn
* **Web Interface:** Gradio
* **Environment:** Jupyter Notebook / Python Scripts

---

## 💡 Machine Learning Workflow & Code Snippets

### 1. Data Cleaning & Feature Scaling
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Handle zero values in clinical columns where 0 is invalid
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    df[col] = df[col].replace(0, df[col].median())

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop('Outcome', axis=1))
