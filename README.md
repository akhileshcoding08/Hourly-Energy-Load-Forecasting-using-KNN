# ⚡ Hourly Energy Load Forecasting using K-Nearest Neighbors (KNN)

A machine learning project that forecasts **hourly electricity load (in MW)** for the PJM Interconnection region using a **K-Nearest Neighbors Regressor**. The project covers the full workflow like data cleaning, feature engineering, exploratory data analysis (EDA), model training, and evaluation.

---

## 📌 Project Overview

Electric utilities need accurate short-term load forecasts to balance supply and demand efficiently. This project uses historical hourly load data from **PJM Interconnection LLC** (a regional transmission organization in the US) to build a KNN regression model that predicts total system load (`PJM_Load`) based on time-based and lag-based features.

---

## 🎯 Objectives

- Clean and preprocess raw hourly energy load data
- Engineer time-based features (hour, day of week, month, season, etc.)
- Explore diurnal, weekly, seasonal, and yearly load patterns through visualizations
- Build lag-based features to capture temporal dependency
- Train a K-Nearest Neighbors Regression model
- Evaluate model performance using RMSE, MAE, and R²

---

## 🗂️ Dataset

**File:** `pjm_hourly_est.csv`

The dataset contains hourly electricity load estimates (in Megawatts) for multiple PJM sub-regions, collected over several years.

| Column | Description |
|---|---|
| `Datetime` | Timestamp of the reading (hourly) |
| `AEP, COMED, DAYTON, DEOK, DOM, DUQ, EKPC, FE, NI, PJME, PJMW` | Load values for individual PJM sub-regions (contain missing values) |
| `PJM_Load` | **Target variable** — total PJM system load (MW) |

> Source: [PJM Hourly Energy Consumption Data (Kaggle)](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Libraries:**
  - `pandas`, `numpy` — data manipulation
  - `matplotlib`, `seaborn` — data visualization
  - `scikit-learn` — machine learning (KNN Regressor, scaling, metrics)
- **Environment:** Jupyter Notebook

---

## 🔍 Project Workflow

1. **Data Loading & Cleaning**
   - Load the CSV file and parse `Datetime`
   - Sort chronologically and drop rows with missing `PJM_Load`

2. **Feature Engineering**
   - Extract `Hour`, `DayOfWeek`, `Month`, `Year`, `DayOfYear` from `Datetime`
   - Create a `Season` column (Winter, Spring, Summer, Autumn)
   - Generate lag features: `Lag_1`, `Lag_2`, `Lag_24` (previous 1hr, 2hr, and 24hr load values)

3. **Exploratory Data Analysis (EDA)**
   - Historical load trend over time
   - Distribution of load values (histogram + KDE)
   - Diurnal pattern (average load by hour of day)
   - Weekly pattern (average load by day of week)
   - Seasonal/monthly load distribution (boxplot)
   - Diurnal profile stratified by season
   - Yearly macro trend
   - Lag plot (load at *t* vs load at *t-1*)

4. **Model Building**
   - Features used: `Hour`, `DayOfWeek`, `Month`, `DayOfYear`, `Lag_1`, `Lag_2`, `Lag_24`
   - Target: `PJM_Load`
   - Train/test split: 80/20
   - Feature scaling using `StandardScaler` (essential for distance-based algorithms like KNN)
   - Model: `KNeighborsRegressor(n_neighbors=7, weights='distance')`

5. **Model Evaluation**
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)
   - R² Score
   - Actual vs Predicted scatter plot

---

## 📊 Results

| Metric | Value |
|---|---|
| **RMSE** | 660.87 MW |
| **MAE** | 482.30 MW |
| **R² Score** | 0.9873 |

The model explains **~98.7%** of the variance in hourly PJM system load, indicating strong predictive performance.

---

## 📁 Repository Structure

```
├── Hourly_energy_load_KNN.ipynb   # Main Jupyter Notebook (EDA + Model)
├── pjm_hourly_est.csv             # Dataset
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/hourly-energy-load-knn.git
cd hourly-energy-load-knn

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
jupyter notebook Hourly_energy_load_KNN.ipynb
```

Run all cells sequentially to reproduce the EDA, model training, and evaluation.

---

## 📦 requirements.txt

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

## 🔮 Future Improvements

- Perform a chronological (time-based) train/test split instead of random split for a more realistic time-series evaluation
- Hyperparameter tuning of `n_neighbors` using cross-validation (e.g., GridSearchCV)
- Compare KNN with other models (Random Forest, XGBoost, LSTM) for benchmarking
- Add weather data (temperature, humidity) as external regressors
- Deploy the model as a REST API or interactive dashboard

---
