# 🚢 Titanic Survival Predictor (End-to-End ML App)

### 🔴 Live Demo: [Click Here to View App](https://titanicappgit-nlpmdx5kk7nvndtyk7mklj.streamlit.app/)

## 📋 Project Overview
This project is a Machine Learning web application that predicts the survival probability of Titanic passengers based on demographic and ticket data. It demonstrates the full Data Science lifecycle: from data cleaning and feature engineering to model training and cloud deployment.

## 🛠️ Tech Stack
* **Python 3.9**
* **Machine Learning:** Scikit-Learn (Random Forest Classifier), Pandas, NumPy
* **Web Framework:** Streamlit
* **Deployment:** Streamlit Cloud

## 🧠 Key Features
* **Data Pipeline:** Handles missing values (Imputation) and converts categorical data (One-Hot Encoding).
* **Model:** Uses a Random Forest Classifier (Accuracy: ~80%) to handle non-linear relationships.
* **Interactive UI:** Users can adjust parameters (Age, Fare, Family Size) to see real-time predictions.

## 📂 File Structure
* `train.py`: The training pipeline. Reads raw data, trains the model, and saves the serialized object (`.pkl`).
* `app.py`: The frontend application. Loads the saved model and serves the user interface.
* `requirements.txt`: Dependencies for cloud deployment.

---
*Created by Imadh Aboo Ubaith*
