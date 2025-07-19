# Bulldozer_Price_Prediction_Model

This project is focused on predicting the **sale price of bulldozers** using historical sales data. It is based on the Blue Book for Bulldozers Kaggle dataset and applies machine learning for regression.

## 🧠 Problem Statement

Predict the auction sale price of bulldozers based on their usage history, configuration, and other features.

## 📂 Dataset

The dataset used is from the **Bluebook for Bulldozers** competition:
- `TrainAndValid.csv`
- `Test.csv`

Contains information such as:
- Equipment IDs
- Sale date
- Machine specifications
- Usage stats

## ⚙️ Workflow Overview

1. **Data Preprocessing**
   - Parsing datetime (`saledate`)
   - Feature engineering: Year, Month, Day, Day of Year, Day of Week
   - Handling categorical & missing data
   - Label encoding

2. **Model Building**
   - Baseline: `RandomForestRegressor`
   - Evaluation using:
     - MAE
     - RMSLE
     - R² Score

3. **Hyperparameter Tuning**
   - `RandomizedSearchCV`

4. **Prediction**
   - Apply the trained model to the test dataset
   - Output predictions as `test_prediction.csv`

5. **Feature Importance**
   - Visualizing top features affecting bulldozer price

## 📊 Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Log Error (RMSLE)
- R² Score

## 🛠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib

## 📁 Output Files

- `Train_tmp.csv` – cleaned training data
- `test_prediction.csv` – predicted bulldozer prices

## 🚀 How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib

# Run the script
python bulldozer_price_prediction_model.py
