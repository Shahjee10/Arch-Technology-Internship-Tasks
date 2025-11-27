# Internship Tasks – ARCH Technologies

This repository contains my internship tasks completed at **ARCH Technologies**.  
Each task helped me explore practical Data Science and Machine Learning concepts through hands-on implementation.

---

# 📌 Internship Task 1 – Stock Price Prediction with LSTM

## 📝 Overview
This project was completed as **Internship Task 1** during my internship at **ARCH Technologies**.  
It is a **guided project** where I explored **stock price prediction** using **LSTM (Long Short-Term Memory)** networks in Python.  
The goal of the project was to understand **time series forecasting**, data preparation, model building, and making future predictions.

---

## 🛠 Tools & Libraries Used
- **Python 3**
- **Pandas** – data handling  
- **NumPy** – numerical operations  
- **Matplotlib** – data visualization  
- **Scikit-learn** – scaling (`MinMaxScaler`)  
- **TensorFlow / Keras** – LSTM model  
- **sklearn.metrics** – RMSE evaluation  

---

## 🔄 Project Workflow

### 1️⃣ Data Loading & Exploration
- Loaded `AAPL.csv`  
- Explored data using `.head()` and `.info()`  
- Main columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`

### 2️⃣ Data Preprocessing
- Selected closing prices  
- Visualized trends  
- Applied **MinMaxScaler** to normalize values  

### 3️⃣ Time Series Preparation
- Created sliding window sequences (`time_step = 100`)  
- Split into **65% training** and **35% testing**  
- Reshaped data for LSTM input  

### 4️⃣ Model Building & Training
- Built a **stacked LSTM model**  
- Optimizer: `adam`, Loss: `mse`  
- Trained for **100 epochs**, batch size **64**

### 5️⃣ Prediction & Evaluation
- Generated predictions for train & test sets  
- Inverse transformed values  
- Evaluated using **RMSE**  
- Plotted **Actual vs Predicted** graphs  

### 6️⃣ Future Forecasting
- Predicted **next 30 days**  
- Visualized forecasted trend  

---

## ✅ Results
- Successfully predicted the next 30 days of stock prices  
- Built and trained a complete LSTM time-series model  
- Improved understanding of **deep learning for forecasting**

---

# 📌 Internship Task 2 – Titanic Survival Prediction (ML Classification)

## 📝 Overview
This task focuses on the **Titanic Survival Prediction** problem, a classic **binary classification** dataset.  
The objective was to clean the dataset, encode categorical columns, visualize relationships, and train ML models to predict survival.

---

## 🛠 Tools & Libraries Used
- **Python 3**
- **Pandas** – data cleaning  
- **NumPy** – numerical operations  
- **Matplotlib / Seaborn** – visualizations  
- **Scikit-learn** – ML models & preprocessing  

---

## 🔄 Project Workflow

### 1️⃣ Data Cleaning
- Removed duplicate rows  
- Filled missing values (`Age`, `Embarked`)  
- Dropped unnecessary columns  
- Avoided chained assignment warnings by safe DataFrame updating  

### 2️⃣ Exploratory Data Analysis (EDA)
- Count plots for `Survived`, `Sex`, `Pclass`  
- Heatmap for correlations  
- Distribution plots for Age & Fare  
- Insights: Women & children had higher survival rate

### 3️⃣ Feature Engineering
- Converted categorical columns using:
  ```python
  df.replace({'Sex': {'male': 0, 'female': 1},
              'Embarked': {'S': 0, 'C': 1, 'Q': 2}})
### 3️⃣ Feature Selection & Scaling
- Selected relevant features for training  
- Scaled numerical columns  

### 4️⃣ Model Building
Trained multiple ML models:
- **Logistic Regression**  
- **Random Forest Classifier**  
- **Decision Tree Classifier**  

Evaluated using:
- **Accuracy Score**  
- **Classification Report**  
- **Confusion Matrix**  

### 5️⃣ Results
- Achieved strong accuracy on the Titanic dataset  
- Classification report provided **precision, recall, F1-score**  
- Visualization of model predictions & survival patterns  
