# Internship Task 1 – Stock Price Prediction with LSTM

## Overview

This project was completed as **Internship Task 1** during my internship at **ARCH Technologies**.  
It is a **guided project** where I explored **stock price prediction** using **LSTM (Long Short-Term Memory)** networks in Python.  
The goal of the project was to understand **time series forecasting**, data preparation, model building, and making future predictions.

---

## Tools & Libraries Used

- **Python 3**
- **Pandas** – for data handling
- **NumPy** – for numerical operations
- **Matplotlib** – for data visualization
- **Scikit-learn** – for data scaling (`MinMaxScaler`)
- **TensorFlow / Keras** – for building and training the LSTM model
- **Sklearn.metrics** – for evaluating model performance (RMSE)

---

## Project Workflow

1. **Data Loading & Exploration**  
   - Loaded `AAPL.csv` into a pandas DataFrame.  
   - Explored the dataset using `.head()` and `.tail()`.  
   - Columns included: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`, `Adjusted Close`.

2. **Data Preprocessing**  
   - Extracted closing prices.  
   - Visualized closing price trends.  
   - Applied **MinMaxScaler** to normalize data to [0,1] for LSTM.  

3. **Time Series Preparation**  
   - Created a time series dataset using a sliding window approach (`time_step=100`).  
   - Split data into **training (65%)** and **testing sets (35%)**.  
   - Reshaped data for LSTM input.

4. **Model Building & Training**  
   - Built a **stacked LSTM** model with three LSTM layers and a Dense output layer.  
   - Compiled the model using `adam` optimizer and `mean_squared_error` loss.  
   - Trained the model on training data with validation on test data (100 epochs, batch size 64).  

5. **Prediction & Evaluation**  
   - Generated predictions for training and testing data.  
   - Inverse-transformed predictions to original scale.  
   - Evaluated using **RMSE**.  
   - Plotted **actual vs predicted prices**.

6. **Future Forecasting**  
   - Predicted stock prices for the **next 30 days**.  
   - Visualized future predictions alongside recent trends.  

---

## Results

- Successfully predicted stock prices for the next 30 days.  
- Visualized training, testing, and future predictions.  
- Gained practical understanding of **LSTM networks for time series forecasting**.  

---

## Usage

1. Clone the repository:

```bash
git clone https://github.com/Shahjee10/Arch-Technology-Internship-Tasks.git
