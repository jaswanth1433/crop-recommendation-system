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
