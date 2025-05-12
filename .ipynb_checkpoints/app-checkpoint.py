import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Soybean Disease Classifier", layout="centered")
st.title("🌱 Soybean Disease Prediction App")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Soybean.csv")
    return df

df = load_data()
st.subheader("Raw Dataset")
st.write(df.head())

# Handle missing values
imputer = SimpleImputer(strategy='most_frequent')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Encode categorical features
label_encoders = {}
for col in df_imputed.columns:
    le = LabelEncoder()
    df_imputed[col] = le.fit_transform(df_imputed[col])
    label_encoders[col] = le

# Split data
X = df_imputed.drop('Class', axis=1)
y = df_imputed['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
st.subheader("Model Performance")
st.write("Accuracy:", accuracy_score(y_test, y_pred))

fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", ax=ax)
ax.set_title("Confusion Matrix")
st.pyplot(fig)

st.subheader("Predict a New Sample")

# Get user input
def user_input_features():
    user_data = {}
    for col in X.columns:
        options = label_encoders[col].classes_.tolist()
        user_data[col] = st.selectbox(f"{col}", options)
    input_df = pd.DataFrame([user_data])
    for col in input_df.columns:
        input_df[col] = label_encoders[col].transform(input_df[col])
    return input_df

input_df = user_input_features()

if st.button("Predict"):
    prediction = model.predict(input_df)
    class_label = label_encoders['Class'].inverse_transform(prediction)[0]
    st.success(f"Predicted Class: {class_label}")

