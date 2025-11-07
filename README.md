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
- Data Cleaning and Preprocessing  
- Outlier Detection and Removal  
- Feature Engineering
- Label Encoding and Scaling  
- Model Building using **Logistic Regression** and **Random Forest**  
- Hyperparameter Tuning using **GridSearchCV**  
- Model Evaluation with ROC-AUC and Cross-Validation  
- Feature Importance Visualization  
- (Bonus) Interactive **Streamlit App** for real-time prediction  

---

##  Machine Learning Workflow

###  Data Understanding
- Load dataset and explore structure (`info()`, `describe()`)
- Handle missing values
- Fix inconsistent categorical entries (e.g., `male → Male`)

###  Data Cleaning
- Replace missing education levels with “No High School”
- Drop duplicate entries
- Remove irrelevant columns like `Email`, `First_Name`, etc.

###  Outlier Detection
- Boxplots for numeric columns
- IQR method to remove extreme values

###  Feature Engineering
Created new informative features:
- **Study_Efficiency** = Total_Score / (Study_Hours * Attendance)
- **Parent_Support_Score** = 0.7 × Parent_Education + 0.3 × Internet_Access

###  Encoding and Scaling
- Used `LabelEncoder()` for categorical variables
- Scaled numerical features using `StandardScaler`

###  Model Building
Trained two classification models:
- **Logistic Regression**
- **Random Forest Classifier**

###  Model Evaluation
Metrics used:
- Accuracy
- ROC-AUC Score
- Classification Report
- Confusion Matrix
- ROC Curve Visualization

###  Hyperparameter Tuning
Optimized Random Forest parameters (`n_estimators`, `max_depth`, etc.) using `GridSearchCV`.

###  Feature Importance
Identified the top contributing features to the model’s predictions:
- Total_Score  
- Study_Hours_per_Week  
- Attendance (%)  
- Stress_Level (1-10)  
- Sleep_Hours_per_Night  

---

##  Technologies Used

| Category | Tools & Libraries |
|-----------|------------------|
| Programming | Python  |
| Data Handling | pandas, numpy |
| Visualization | matplotlib, seaborn, plotly |
| Machine Learning | scikit-learn |
| Model Persistence | joblib |
| Web App (optional) | Streamlit |

---

##  Dataset Information
The dataset `Students_Performance_Dataset.csv` includes:
- Academic details (scores, attendance)
- Behavioral metrics (study hours, stress level)
- Parental and socio-economic details
- Target column: **Result** (Pass/Fail)

---

##  Model Training Steps

```python
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
model_l = LogisticRegression(max_iter=1000)
model_r = RandomForestClassifier(random_state=42)

model_l.fit(X_train, y_train)
model_r.fit(X_train, y_train)

# Evaluate
y_pred = model_r.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```
## Model Saving

The trained model and preprocessing pipeline are saved using joblib:
```python
joblib.dump(pipeline, "pipeline.pkl")
```

This allows easy reuse in prediction apps or Streamlit dashboards.

## Installation & Setup
### Clone the Repository
```bash
git clone https://github.com/<your-username>/Students_Performance.git
cd Students_Performance
```

### Create a Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate        # On Windows
# OR
source venv/bin/activate     # On macOS/Linux
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Run the Streamlit App
```bash
streamlit run Students_Performance.py
```

Then open the displayed local URL, usually:
http://localhost:8501

## Sample Prediction
```
🎉 The student is likely to PASS (Confidence: 89.45%)
```

## Acknowledgements

Special thanks to:

- Scikit-learn for ML utilities

- Streamlit for fast web visualization

- Plotly for interactive plots

#### “Predicting success is not about data alone — it’s about enabling every student to reach their potential.”
