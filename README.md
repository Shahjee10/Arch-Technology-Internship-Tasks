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

---

# 📌 Internship Task 3 – Customer Segmentation using K-Means Clustering

## 📝 Overview
This project was completed as **Internship Task 3 (Month 2)** during my internship at **ARCH Technologies**.  
The objective of this task was to perform **Customer Segmentation** using **Unsupervised Machine Learning** techniques to identify meaningful customer groups based on purchasing behavior.

Using a mall customer dataset, I applied the **K-Means clustering algorithm** to segment customers based on their **Annual Income** and **Spending Score**, helping businesses better understand customer patterns and improve targeted marketing strategies.

---

## 🛠 Tools & Libraries Used
- **Python 3**
- **Pandas** – data handling & preprocessing  
- **NumPy** – numerical operations  
- **Matplotlib / Seaborn** – data visualization  
- **Scikit-learn** – K-Means clustering & preprocessing  

---

## 🔄 Project Workflow

### 1️⃣ Data Loading & Exploration
- Loaded the mall customer dataset  
- Explored dataset structure using `.head()` and `.info()`  
- Key features used: `Annual Income`, `Spending Score`

### 2️⃣ Data Preprocessing
- Selected relevant numerical features  
- Checked for missing values  
- Prepared data for clustering  

### 3️⃣ Optimal Cluster Selection
- Used the **Elbow Method**  
- Identified **5 optimal clusters** for segmentation  

### 4️⃣ Model Building
- Applied **K-Means clustering** with `k = 5`  
- Assigned cluster labels to each customer  

### 5️⃣ Visualization & Interpretation
- Visualized clusters using scatter plots  
- Analyzed customer behavior patterns within each cluster  

---

## ✅ Results & Insights
The model successfully identified **5 distinct customer segments**:

- **High income + high spending** – ideal customers for premium offers  
- **High income + low spending** – strong potential for targeted campaigns  
- **Low income + high spending** – loyal value-seeking customers  
- **Low income + low spending** – budget-conscious customers  
- **Medium income + balanced spending** – stable and consistent group  

This analysis demonstrates how **Unsupervised Learning** can uncover hidden patterns in customer data and support **data-driven business decisions**.

---

# 📌 Internship Task 4 – Movies Rating Prediction (Exploratory Data Analysis)

## 📝 Overview
This project was completed as the **final internship task of Month 2** during my internship at **ARCH Technologies**.  
The objective of this task was to perform **Exploratory Data Analysis (EDA)** on a movie dataset to understand patterns in movie attributes and prepare the data for **future rating prediction models**.

This task focused on converting **raw movie data into meaningful insights** through data cleaning, analysis, and visualization.

---

## 🛠 Tools & Libraries Used
- **Python 3**
- **Pandas** – data loading & preprocessing  
- **NumPy** – numerical operations  
- **Matplotlib** – basic visualizations  
- **Seaborn** – advanced data visualization  

---

## 🔄 Project Workflow

### 1️⃣ Data Loading & Exploration
- Loaded the movie dataset  
- Explored dataset structure using `.head()`, `.tail()`, `.shape()`, and `.info()`  
- Identified key features such as movie runtime and ratings  

### 2️⃣ Data Cleaning & Preparation
- Checked for missing and null values  
- Removed incomplete records to improve data quality  
- Prepared a clean dataset for analysis  

### 3️⃣ Exploratory Data Analysis (EDA)
- Analyzed movie attributes such as **runtime**  
- Identified **top lengthy movies**  
- Explored trends and patterns within the dataset  

### 4️⃣ Data Visualization
- Created informative visualizations using **Seaborn and Matplotlib**  
- Used bar plots to highlight key insights from the data  

---

## ✅ Results & Outcomes
- Successfully cleaned and analyzed a real-world movie dataset  
- Extracted actionable insights using **EDA techniques**  
- Built a strong foundation for **movie rating prediction models**  
- Improved skills in **data analysis, visualization, and data storytelling**

---

This task completed the **Month 2 internship objectives** at **ARCH Technologies** and reinforced the importance of **EDA before applying machine learning models**.


  
