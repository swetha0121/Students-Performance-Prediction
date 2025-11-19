import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import streamlit as st
import joblib
import os
import plotly.express as px

data = pd.read_csv("Students_ Performance_Dataset.csv")
df = pd.DataFrame(data)

# ------Data Understanding---------
print(df.head())
print (df.info())
print (df.describe())
print("Before handling missing values:\n", df.isnull().sum()) # Missing values

# ------Data Cleaning-------
# Handling missing values
df['Parent_Education_Level'] = df["Parent_Education_Level"].fillna('None')
df.fillna(df.mean(numeric_only=True),inplace=True)
print("After handling missing values:\n", df.isnull().sum())
print("\nDuplicates:", df.drop_duplicates(inplace=True)) #Dropping duplicates
# Fixing inconsistent
print("\nBefore fixing inconsistent:",df["Gender"].unique())
df['Gender'] = df["Gender"].str.strip().str.title() # male -> Male
print("After fixing inconsistent:",df["Gender"].unique())
print("\nBefore fixing inconsistent:",df['Parent_Education_Level'].unique())
df['Parent_Education_Level'] = df["Parent_Education_Level"].replace({'None': 'No High School'})  # None -> No High School
print("After fixing inconsistent:",df['Parent_Education_Level'].unique())

# --------Outlier Detection---------
# Visualizing outliers
plt.figure(figsize=(10,4))
sns.boxplot(data=df[['Study_Hours_per_Week', 'Attendance (%)']])
plt.title("Outliers in Study_Hours_per_Week & Attendance (%)")
plt.show()

numeric_cols = list(df.select_dtypes(include=['float64', 'int64']).columns)
# numeric_cols = ['Age', 'Attendance (%)', 'Midterm_Score', 'Final_Score',
#        'Assignments_Avg', 'Quizzes_Avg', 'Participation_Score',
#        'Projects_Score', 'Total_Score', 'Study_Hours_per_Week',
#        'Stress_Level (1-10)', 'Sleep_Hours_per_Night']
print("\nNumeric columns:\n",numeric_cols)
plt.figure(figsize=(15, 6))
df.boxplot(column=(numeric_cols))
plt.title("Boxplot for Numeric Columns (Outlier Visualization)")
plt.xticks(rotation=45)
plt.show()
#Outliers detection
df_no = df.copy()
drop_cols = ["Email", "First_Name", "Last_Name", "Student_ID"]
for col in drop_cols:
    if col in df_no.columns:
        df_no = df_no.drop(columns=col)
# df_no = df.drop(columns=["Email", "First_Name", "Last_Name", "Student_ID"], errors="ignore")

for col in numeric_cols:
    Q1 = df_no[col].quantile(0.25)
    Q3 = df_no[col].quantile(0.75)
    IQR = Q3 - Q1
    Lower_Bound = Q1 - 1.5 * IQR
    Upper_Bound = Q3 + 1.5 * IQR
    df_no = df_no[(df_no[col] >= Lower_Bound) & (df_no[col] <= Upper_Bound)]
print("\nBefore removing outliers:",df.shape)
print("After removing outliers:",df_no.shape)

# ----------Feature Engineering-----------
df_no['Study_Efficiency'] = ((df_no['Total_Score']) / (df_no['Study_Hours_per_Week']*df_no['Attendance (%)']+1))

Parent_edu_score = {'No High School':0, 'High School':1, "Bachelor's":2, "Master's":3, 'PhD':4}
df_no['Parent_Education_Score'] = df_no['Parent_Education_Level'].map(Parent_edu_score)

df_no['Internet_Score'] = df_no['Internet_Access_at_Home'].map({'Yes':1, 'No':0})
# combining both
df_no['Parent_Support_Score'] = (df_no['Parent_Education_Score']*0.7 + df_no['Internet_Score']*0.3)
print(df_no[['Study_Efficiency', 'Parent_Support_Score']].head())

# ---------Encoding-----------
categorical_cols = list(df_no.select_dtypes(include=['object']).columns)
# categorical_cols = [col for col in categorical_cols if col in df_no.columns] 
encoders = {}

for col in categorical_cols:
    le = LabelEncoder() 
    df_no[col] = le.fit_transform(df_no[col].astype(str))
    encoders[col] = le
# ----------Feature Scaling----------

# Split data
X = df_no.drop(columns=['Result'], errors = "ignore")
y = df_no['Result']

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

df_clean = df_no.copy()
Scaler = StandardScaler()
X[numeric_cols] = Scaler.fit_transform(X[numeric_cols])

# Handle imbalance using SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)


#  Check class balance after SMOTE
print("\nBalanced Target Counts:\n", y_res.value_counts())

# split + train test model
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
print("\nTrain size:", X_train.shape, "Test size:", X_test.shape)


# ---------Model Building-------------
# df_no_small = df_no.sample(10, random_state=42)
model_l = LogisticRegression(max_iter=1000)
model_r = RandomForestClassifier(random_state=42)

model_l.fit(X_train, y_train)
model_r.fit(X_train, y_train)

# -----------Model Evaluation------------
models = {'LogisticRegression' : model_l, 'RandomForestClassifier' : model_r}

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"\n{name} Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    # confusion matrix
    print("Confusion Matrix:", confusion_matrix(y_test, y_pred))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='coolwarm')
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, y_prob):.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"ROC Curve - {name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

# ----------Hyperparameter Tuning--------------
# Cross validation comparison
print("\n Performing 5-Fold Cross Validation\n")
for name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"{name} Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 5, 10, 15],'min_samples_split': [2, 5, 10]}
grid = GridSearchCV(model_r, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print("\n Best Random Forest Parameters:", grid.best_params_)
print("Best Accuracy from GridSearchCV:", grid.best_score_)

# -------------Model Interpretation (Feature Importance)-----------
best_rf = grid.best_estimator_
# If we want slightly smoother results, you can also:
best_rf.set_params(class_weight='balanced', random_state=42)

feat_imp = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 5))
feat_imp[:10].plot(kind='bar', color='teal')
plt.title("Top 10 Important Features(Random Forest)")
plt.show()

print("\nTop 10 Important Features:\n", feat_imp.head(10))

pipeline = {
    "model": best_rf,
    "scaler": Scaler,
    "encoders": encoders,
    "features": X.columns.tolist(),
    "numeric_cols": numeric_cols
}

joblib.dump(pipeline, "pipeline.pkl")
print("Saved model and preprocessing pipeline as pipeline.pkl")

joblib.dump(encoders, "encoders.pkl")

# 1. LOAD PIPELINE
pipeline = joblib.load("pipeline.pkl")
model = pipeline["model"]
scaler = pipeline["scaler"]
encoders = pipeline["encoders"]
FEATURES = pipeline["features"]
NUMERIC_COLS = pipeline["numeric_cols"]

# STREAMLIT UI

st.set_page_config(page_title="🎓 Student Performance Predictor", layout="centered")

st.title("Student Performance Predictor")

st.write("Enter student details to predict Pass/Fail:")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Overview", "📊 Data Insights", "📁 Dataset", "⚙️ Feature Importance", "🎯 Prediction"])

# ----- Tab 1: Overview -----
with tab1:
    st.title(" Overview Dashboard")
    st.metric("Total Students", 150)
    st.metric("Average Score", "82%")
    st.success("All systems running smoothly ✅")
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Study Hours", round(df_clean["Study_Hours_per_Week"].mean(),1))
    col2.metric("Pass Rate", f"{(df_no['Result'].mean()*100):.1f}%")
    col3.metric("Avg Stress Level", round(df_clean["Stress_Level (1-10)"].mean(),1))

# ----- Tab 2: Data Insights ----

with tab2:
    st.header(" Data Insights")
    fig = px.scatter(df, x="Study_Hours_per_Week", y="Total_Score", color="Result", size="Attendance (%)", hover_data=["Department", "Grade"], title="📈 Study Hours vs Total Score (by Result)")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(df, x="Sleep_Hours_per_Night", y="Stress_Level (1-10)", color="Result", symbol="Gender", title="😴 Sleep Hours vs Stress Level (by Result)")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(df, x="Department", y="Total_Score", color="Result", barmode="group", title="🏫 Department-wise Average Score")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header(" Dataset Viewer")
    st.write("Here’s the student performance dataset used for analysis and prediction:")
    st.dataframe(df)  #  displays the full dataset interactively
    st.markdown("### 🔍 Dataset Info")
    st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
    st.write("**Columns:**", list(df.columns))

# ----- Tab 3: Feature Importance -----
with tab4:
    st.header(" Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # Plotly bar chart
    fig = px.bar(feature_importance,
                 x='Importance',
                 y='Feature',
                 orientation='h',
                 title="Top Feature Importances in Student Performance Model")
    st.plotly_chart(fig, use_container_width=True)
    


st.set_page_config(page_title="Student Performance Dashboard", layout="wide")


# Use same mappings as in training
gender_map = {'Male': 1, 'Female': 0}  # because you used LabelEncoder
dept_map = {'Mathematics': 2, 'Business': 0, 'Engineering': 1, 'CS': 3}  # example
grade_map = {'A':4, 'B':3, 'C':2, 'D':1, 'F':0}
yes_no_map = {'Yes': 1, 'No': 0}
family_income_map = {'Low': 0, 'Medium': 1, 'High': 2}

def get_grade(total):
    if total < 60:
        return "F"
    elif total < 70:
        return "D"
    elif total < 80:
        return "C"
    elif total < 90:
        return "B"
    else:
        return "A"
df['Grade'] = df['Total_Score'].apply(get_grade)   # Example usage
print(df)


# ----- 🎯 Tab 5: Prediction Form -----
with tab5:
    st.header("🎯 Student Performance Prediction")
    with st.form("Prediction_form"):
        # gender = st.selectbox("Gender", ["Male", "Female"])
        # age = st.number_input("Age", min_value=10, max_value=25, value=18)
        # department = st.selectbox("Department", ['Mathematics', 'Business', 'Engineering', 'CS'])
        attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=90)
        midterm_score = st.number_input("Midterm Score", min_value=0, max_value=100, value=75)
        final_score = st.number_input("Final Score", min_value=0, max_value=100, value=80)
        assignments_avg = st.number_input("Assignments Avg", min_value=0, max_value=100, value=85)
        quizzes_avg = st.number_input("Quizzes Avg", min_value=0, max_value=100, value=80)
        participation_score = st.number_input("Participation Score", min_value=0, max_value=10, value=8)
        projects_score = st.number_input("Projects Score", min_value=0, max_value=100, value=90)
        total_score = st.number_input("Total Score", min_value=0, max_value=500, value=330)
        # grade = st.selectbox("Grade", ["A", "B", "C", "D", "F"])
        # study_hours_per_week = st.number_input("Study Hours per Week", min_value=0, max_value=100, value=15)
        # extracurricular_activities = st.selectbox("Extracurricular Activities", ["Yes", "No"])
        # internet_access_at_home = st.selectbox("Internet Access at Home", ["Yes", "No"])
        # parent_education_level = st.selectbox("Parent Education Level", ["No High School","High School", "Bachelor's", "Master's", "PhD"])
        # family_income_level = st.selectbox("Family Income Level", ["Low", "Medium", "High"])
        # stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5)
        # sleep_hours_per_night = st.number_input("Sleep Hours per Night", min_value=0, max_value=12, value=7)
    
    
        submit = st.form_submit_button("Predict")

        # Create input DataFrame with **exact column names** as in your training data
    if submit:
        total_score = np.mean([midterm_score, final_score, assignments_avg, quizzes_avg, participation_score, projects_score])
        study_efficiency = total_score / (midterm_score + final_score + assignments_avg + quizzes_avg + paticipation_score + projects_score + 1)
        grade = get_grade(total_score)
        parent_support_score_val = 0.5   # default fixed value
        internet_score_val = 0.5
        parent_edu_score_map = 2
        # parent_edu_score_map = {'No High School':0, 'High School':1, "Bachelor's":2, "Master's":3, 'PhD':4}
        # parent_edu_score_val = parent_edu_score_map[parent_education_level]
        # internet_score_val = 1 if internet_access_at_home == "Yes" else 0
        # parent_support_score_val = parent_edu_score_val * 0.7 + internet_score_val * 0.3
        # study_efficiency_val = total_score / (study_hours_per_week * attendance + 1)
        # Encode inputs
        raw_input = pd.DataFrame([{
            # "Gender": gender,
            # "Age": age,
            # "Department": department,
            "Attendance (%)": attendance,
            "Midterm_Score": midterm_score,
            "Final_Score": final_score,
            "Assignments_Avg": assignments_avg,
            "Quizzes_Avg": quizzes_avg,
            "Participation_Score": participation_score,
            "Projects_Score": projects_score,
            "Total_Score": total_score,
            "Grade": grade,
            # "Study_Hours_per_Week": study_hours_per_week,
            # "Extracurricular_Activities": extracurricular_activities,
            # "Internet_Access_at_Home": internet_access_at_home,
            "Parent_Education_Level": parent_education_level,
            "Family_Income_Level": family_income_level,
            "Stress_Level (1-10)": stress_level,
            # "Sleep_Hours_per_Night": sleep_hours_per_night,
            "Study_Efficiency": study_efficiency_val,
            "Parent_Education_Score": parent_edu_score_val,
            "Internet_Score": internet_score_val,
            "Parent_Support_Score": parent_support_score_val
        }])
        # Encode categorical columns using saved encoders
        input_df = raw_input.copy()
        for col, le in encoders.items():
            if col in input_df.columns:
                val = input_df.loc[0, col]
                if val in le.classes_:
                    input_df[col] = le.transform(input_df[col])
                else:
                    st.warning(f"⚠️ '{val}' not seen in training for '{col}', using fallback encoding.")
                    input_df[col] = 0
        # Ensure numeric types
        for col in NUMERIC_COLS:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0)
        # Reorder and scale numeric columns
        input_df = input_df[FEATURES]
        scaled_values = scaler.transform(input_df[NUMERIC_COLS])
        input_df.loc[:, NUMERIC_COLS] = scaled_values
    
        # ensure feature order
        input_df = input_df.reindex(columns=FEATURES, fill_value=0)

        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]  # class 1 = PASS
        
        # Display Result
        st.markdown("---")
        if prediction == 1:
            st.success(f"🎉 The student is likely to **PASS** (Confidence: {probability*100:.2f}%)")
        else:
            st.error(f"❌ The student is likely to **FAIL** (Confidence: {(1-probability)*100:.2f}%)")
        # Probability chart
        st.subheader("Prediction Probability") 
        labels = ["Fail", "Pass"] 
        probabilities = [1-probability, probability] 
    
        fig, ax = plt.subplots(figsize=(6, 2.5)) 
        ax.barh(labels, probabilities, color=['red','green']) 
        for i, v in enumerate(probabilities): 
            ax.text(v + 0.01, i, f"{v:.1%}", va='center')  
        ax.set_xlim(0,1) 
        st.pyplot(fig, use_container_width=False)
       
        
        # Scatter plot
        st.subheader("Student Context")
        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.scatter(df['Study_Hours_per_Week'], df['Total_Score'], alpha=0.5)
        ax.scatter(study_hours_per_week, total_score, color='red', label="Current Student")
        ax.set_xlabel("Study Hours per Week")
        ax.set_ylabel("Total Score")
        ax.legend()
        st.pyplot(fig, use_container_width=False)
        





