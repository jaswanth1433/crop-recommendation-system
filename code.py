#Crop Recommendation System

# Import libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("C:/Users/JASWATH/Downloads/archive/Crop_recommendation.csv")

# Data preprocessing

# Feature Selection

# Train Test Split

@st.cache_resource
def train_model(df):
    X = df.drop("label", axis=1)
    y = df["label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Tuned Random Forest Model
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, acc, X.columns

# Model training

df = load_data()
model, accuracy, feature_names = train_model(df)

# Prediction Logic

# Title

st.title("Crop Recommendation System")
st.write("Predict suitable crops based on soil nutrients and weather conditions")



