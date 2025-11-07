# Students-Performance-Prediction Using Machine Learning
The goal of this project is to predict if a student will Pass or Fail based on academic and behavioral data such as hours studied, attendance, sleep hours, parental education, and sports participation. 

##  Overview
This project predicts whether a student will **Pass or Fail** based on academic, behavioral, and socio-economic factors.  
It leverages **Data Science & Machine Learning techniques** such as data cleaning, feature engineering, model tuning, and evaluation.

The project also includes an **interactive Streamlit dashboard** that visualizes data insights and allows users to test predictions dynamically.

---

##  Problem Statement
Students’ performance depends on several factors including:
- Study habits
- Attendance
- Parental education level
- Internet access
- Stress levels
- Sleep hours

Manually identifying at-risk students is challenging.  
This project aims to automate this prediction and help educators identify students needing support.

---

##  Key Features
-- Data Cleaning and Preprocessing  
-- Outlier Detection and Removal  
-- Feature Engineering  
-- Label Encoding and Scaling  
* Model Building using **Logistic Regression** and **Random Forest**  
✅ Hyperparameter Tuning using **GridSearchCV**  
✅ Model Evaluation with ROC-AUC and Cross-Validation  
✅ Feature Importance Visualization  
✅ (Bonus) Interactive **Streamlit App** for real-time prediction  

---

## 🧠 Machine Learning Workflow

### 1️⃣ Data Understanding
- Load dataset and explore structure (`info()`, `describe()`)
- Handle missing values
- Fix inconsistent categorical entries (e.g., `male → Male`)

### 2️⃣ Data Cleaning
- Replace missing education levels with “No High School”
- Drop duplicate entries
- Remove irrelevant columns like `Email`, `First_Name`, etc.

### 3️⃣ Outlier Detection
- Boxplots for numeric columns
- IQR method to remove extreme values

### 4️⃣ Feature Engineering
Created new informative features:
- **Study_Efficiency** = Total_Score / (Stud_**_**
