import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------------------
# 🎨 Page Config
# ------------------------------
st.set_page_config(page_title="Telecom Churn Analyzer", layout="wide")

st.title("📡 Telecom Customer Churn Analysis System")
st.markdown("Predict customer churn using Machine Learning")

# ------------------------------
# 📂 Sidebar Upload
# ------------------------------
st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"]
   if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload a dataset to continue")
    st.stop()

# ------------------------------
# ✅ Column Validation
# ------------------------------
required_cols = ["gender","SeniorCitizen","tenure","MonthlyCharges",
                 "InternetService","Contract","TechSupport","Churn"]

if not all(col in df.columns for col in required_cols):
    st.error("Dataset must contain required telecom columns")
    st.stop()

# ------------------------------
# 📄 Data Overview
# ------------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

st.write("Dataset Shape:", df.shape)

# ------------------------------
# 🧹 Feature Engineering
# ------------------------------
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['Charge_per_Tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)

# ------------------------------
# 📊 EDA
# ------------------------------
st.subheader("📊 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.write("Churn Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="Churn", data=df, ax=ax1)
    st.pyplot(fig1)

with col2:
    st.write("Contract vs Churn")
    fig2, ax2 = plt.subplots()
    sns.countplot(x="Contract", hue="Churn", data=df, ax=ax2)
    st.pyplot(fig2)

col3, col4 = st.columns(2)

with col3:
    st.write("Monthly Charges vs Churn")
    fig3, ax3 = plt.subplots()
    sns.boxplot(x="Churn", y="MonthlyCharges", data=df, ax=ax3)
    st.pyplot(fig3)

with col4:
    st.write("Tenure vs Churn")
    fig4, ax4 = plt.subplots()
    sns.boxplot(x="Churn", y="tenure", data=df, ax=ax4)
    st.pyplot(fig4)

# ------------------------------
# 🤖 Model Training
# ------------------------------
X = pd.get_dummies(df.drop("Churn", axis=1))
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# ------------------------------
# 📈 Evaluation
# ------------------------------
y_pred = model.predict(X_test)

st.subheader("📈 Model Performance")

st.write("Accuracy:", accuracy_score(y_test, y_pred))

st.write("Confusion Matrix")
st.write(confusion_matrix(y_test, y_pred))

st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

# ------------------------------
# 🔮 Prediction
# ------------------------------
st.subheader("🔮 Predict Customer Churn")

col5, col6 = st.columns(2)

with col5:
    gender = st.selectbox("Gender", ["Male", "Female"])
    tenure = st.slider("Tenure", 0, 72, 12)
    monthly = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)

with col6:
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    tech = st.selectbox("Tech Support", ["Yes", "No"])

if st.button("Predict"):
    sample = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [0],
        "tenure": [tenure],
        "MonthlyCharges": [monthly],
        "InternetService": [internet],
        "Contract": [contract],
        "TechSupport": [tech]
    })

    sample['Charge_per_Tenure'] = sample['MonthlyCharges'] / (sample['tenure'] + 1)

    sample = pd.get_dummies(sample)
    sample = sample.reindex(columns=X.columns, fill_value=0)

    prediction = model.predict(sample)

    if prediction[0] == 1:
        st.error("⚠️ High Risk: Customer may churn")
    else:
        st.success("✅ Low Risk: Customer likely to stay")
