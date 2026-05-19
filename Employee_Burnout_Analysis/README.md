<div align="center">

# 🔥 Employee Burnout Prediction

**A supervised machine learning project that predicts employee burnout rates using regression models trained on workplace behavioral and organizational features.**

<br/>

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<br/>

> Predict employee burnout before it happens — helping HR teams take proactive action to protect workforce well-being.

</div>

---

## 📌 Table of Contents

1. [Overview](#-overview)
2. [Project Structure](#-project-structure)
3. [Dataset](#-dataset)
4. [Methodology](#-methodology)
   - [Data Cleaning](#1-data-cleaning)
   - [Feature Engineering](#2-feature-engineering)
   - [Encoding](#3-encoding)
   - [Preprocessing & Scaling](#4-preprocessing--scaling)
   - [Model Building](#5-model-building)
5. [Results](#-results)
6. [Feature Correlation with Burn Rate](#-feature-correlation-with-burn-rate)
7. [Getting Started](#-getting-started)
8. [How to Push to GitHub](#-how-to-push-to-github)
9. [Tech Stack](#-tech-stack)
10. [Known Issues & Improvements](#-known-issues--improvements)
11. [License](#-license)

---

## 🧠 Overview

Employee burnout is a state of physical, emotional, and mental exhaustion caused by prolonged workplace stress. It negatively impacts productivity, job performance, and personal well-being — and it is increasingly common in today's fast-paced work environment.

This project uses **supervised regression** to predict each employee's `Burn Rate` (a continuous value between 0.0 and 1.0) from a set of workplace and demographic features. The goal is to help organizations **identify at-risk individuals early** before burnout becomes critical.

Two models were trained and evaluated:
- Linear Regression
- Decision Tree Regressor

**Best model:** Linear Regression with an R² score of **0.9188** — explaining ~92% of the variance in burnout rate.

---

## 📁 Project Structure

```
employee-burnout-prediction/
│
├── 📂 data/
│   ├── employee_burnout_analysis-AI.xlsx       # Raw dataset from Kaggle
│   └── processed/
│       ├── X_train_processed.csv               # Scaled training features
│       └── y_train_processed.csv               # Training target labels
│
├── 📂 models/
│   └── scaler.pkl                              # Saved StandardScaler object (pickle)
│
├── 📂 notebooks/
│   └── Employee_Burnout_Analysis.ipynb         # Full EDA + model training notebook
│
├── main.py                                     # Entry point script
├── requirements.txt                            # Python dependencies
├── .gitignore                                  # Files excluded from version control
└── README.md                                   # Project documentation (this file)
```

---

## 📊 Dataset

**Source:** [Kaggle — Are Your Employees Burning Out?](https://www.kaggle.com/datasets/blurredmachine/are-your-employees-burning-out)

**Size:** 22,750 rows × 9 columns

| Column | Type | Range / Values | Description |
|---|---|---|---|
| `Employee ID` | String | Unique hex ID | Dropped — no predictive value |
| `Date of Joining` | DateTime | 2008-01-01 to 2008-12-31 | Dropped — near-zero correlation with target |
| `Gender` | Categorical | Male / Female | One-hot encoded |
| `Company Type` | Categorical | Service / Product | One-hot encoded |
| `WFH Setup Available` | Categorical | Yes / No | One-hot encoded |
| `Designation` | Numeric | 0.0 – 5.0 | Role seniority level |
| `Resource Allocation` | Numeric | 1.0 – 10.0 | Assigned working hours |
| `Mental Fatigue Score` | Numeric | 0.0 – 10.0 | Self-reported mental fatigue |
| **`Burn Rate`** *(Target)* | Numeric | 0.0 – 1.0 | Burnout intensity to predict |

**Missing values detected before cleaning:**

| Column | Missing Rows |
|---|---|
| Resource Allocation | 1,381 |
| Mental Fatigue Score | 2,117 |
| Burn Rate (target) | 1,124 |

**Final usable rows after `dropna()`:** 18,590

---

## 🔬 Methodology

### 1. Data Cleaning

- Dropped **`Employee ID`** — a unique identifier with no signal for predicting burnout.
- Dropped all rows where **`Burn Rate`** was `NaN` — supervised regression requires a known target.
- Applied `data.dropna()` to remove remaining rows with any missing feature values.

### 2. Feature Engineering

- **`Date of Joining`** was explored by engineering a `Days` feature: days elapsed since `2008-01-01`.
- Correlation of `Days` with `Burn Rate` was only **0.0003** — statistically negligible.
- The join-date distribution was confirmed to be **uniform across all 12 months of 2008**, offering no seasonal signal.
- Both `Date of Joining` and `Days` were subsequently **dropped** from the feature set.

### 3. Encoding

Categorical columns were converted to numeric using **One-Hot Encoding** (`pd.get_dummies`) with `drop_first=True` to prevent multicollinearity:

| Original Column | Encoded As |
|---|---|
| `Gender` | `Gender_Male` |
| `Company Type` | `Company Type_Service` |
| `WFH Setup Available` | `WFH Setup Available_Yes` |

### 4. Preprocessing & Scaling

- **Train/Test Split:** 70% training, 30% test (`random_state=1`, `shuffle=True`)
- **Feature Scaling:** `StandardScaler` fitted on `X_train` only, then applied to both splits to prevent data leakage
- The fitted scaler was serialized to `models/scaler.pkl` using `pickle` for future inference
- Processed training arrays were saved to `data/processed/` as CSV files for reproducibility

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, shuffle=True, random_state=1
)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)
```

### 5. Model Building

**Model 1 — Linear Regression**

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**Model 2 — Decision Tree Regressor**

```python
from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(criterion='squared_error', max_depth=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

## 📈 Results

| Metric | Linear Regression | Decision Tree Regressor |
|---|---|---|
| **R² Score** | **0.9188** ✅ | 0.9121 |
| **MSE** | 0.003157 | 0.003419 |
| **RMSE** | 0.05619 | 0.05847 |
| **MAE** | 0.04595 | 0.04742 |

**Winner: Linear Regression** — achieves a higher R² and lower error across all four metrics. Its simpler structure also makes it more interpretable for HR decision-making contexts.

An RMSE of ~0.056 means predictions are on average within ±0.056 of the true Burn Rate (on a 0–1 scale), which is strong performance for this task.

---

## 🔗 Feature Correlation with Burn Rate

Pearson correlation coefficients computed on the cleaned, pre-encoded dataset:

| Feature | Correlation | Strength |
|---|---|---|
| **Mental Fatigue Score** | **0.9444** | Very strong 🔴 |
| **Resource Allocation** | **0.8550** | Strong 🟠 |
| **Designation** | 0.7364 | Moderate 🟡 |
| Days since joining | 0.0003 | None ⚪ (dropped) |

`Mental Fatigue Score` is overwhelmingly the strongest predictor of burnout. This confirms real-world intuition — mentally exhausted and overworked employees are at the highest risk.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/employee-burnout-prediction.git
cd employee-burnout-prediction
```

### Step 2 — Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run the Notebook

```bash
jupyter notebook notebooks/Employee_Burnout_Analysis.ipynb
```

Or run the main script:

```bash
python main.py
```

---

## 🐙 How to Push to GitHub

Follow these steps in order.

**Step 1 — Create a GitHub account** (skip if you have one)
→ [github.com/signup](https://github.com/signup)

**Step 2 — Create a new repository on GitHub**
- Click `+` → **New repository**
- Name: `employee-burnout-prediction`
- Visibility: Public or Private
- Do **NOT** check "Initialize with README" (you already have one)
- Click **Create repository**

**Step 3 — Rename `_gitignore` → `.gitignore`**

Your current file uses an underscore. Git only reads files named `.gitignore` (with a dot).

```bash
# Windows
rename _gitignore .gitignore

# macOS / Linux
mv _gitignore .gitignore
```

**Step 4 — Populate `.gitignore`**

Paste the following into `.gitignore` to keep large or unnecessary files off GitHub:

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
.env
venv/
.venv/

# Jupyter checkpoints
.ipynb_checkpoints/

# Large data and model files
data/
*.xlsx
*.pkl
*.csv

# OS files
.DS_Store
Thumbs.db
```

**Step 5 — Initialize Git and commit**

```bash
git init
git add .
git commit -m "Initial commit: Employee Burnout Prediction ML project"
```

**Step 6 — Link remote and push**

```bash
git remote add origin https://github.com/YOUR-USERNAME/employee-burnout-prediction.git
git branch -M main
git push -u origin main
```

**Step 7 — Verify**

Open `https://github.com/YOUR-USERNAME/employee-burnout-prediction` — your project and README should be visible.

> For all future updates:
> ```bash
> git add .
> git commit -m "describe your change here"
> git push
> ```

---

## 🛠 Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.8+ | Core language |
| Pandas | Latest | Data loading, cleaning, manipulation |
| NumPy | Latest | Numerical computation |
| Matplotlib | Latest | Visualizations |
| Seaborn | Latest | Statistical plots and pairplots |
| scikit-learn | Latest | Model training, scaling, evaluation |
| Pickle | Built-in | Saving/loading the scaler |
| Jupyter Notebook | Latest | Interactive development |
| openpyxl | Latest | Reading `.xlsx` files |

---

## ⚠️ Known Issues & Improvements

| # | Issue | Recommended Fix |
|---|---|---|
| 1 | `main.py` only prints "hello world" | Build a prediction pipeline that loads `scaler.pkl`, accepts user input, and returns a burn rate prediction |
| 2 | `requirements.txt` is empty | Run `pip freeze > requirements.txt` after activating your venv |
| 3 | `_gitignore` has wrong name | Rename to `.gitignore` using the steps above |
| 4 | All `NaN` rows dropped (loses ~4,000 rows) | Impute `Resource Allocation` and `Mental Fatigue Score` with column medians instead |
| 5 | Trained model not saved | Save the linear regression model using `pickle` or `joblib` alongside `scaler.pkl` |
| 6 | Only 2 models compared | Try `RandomForestRegressor` or `GradientBoostingRegressor` for potentially higher accuracy |
| 7 | No cross-validation | Add k-fold CV (`cross_val_score`) for more statistically reliable performance estimates |
| 8 | Notebook title typo | "EMPLOYEE BUROUT PREDICTION" → "EMPLOYEE BURNOUT PREDICTION" |

---

## 📄 License

This project is intended for educational purposes.  
Dataset credit: [Kaggle — blurredmachine](https://www.kaggle.com/datasets/blurredmachine/are-your-employees-burning-out)

---

<div align="center">
  <sub>Built with Python & scikit-learn &nbsp;·&nbsp; Employee Burnout Prediction — Mini Project</sub>
</div>
