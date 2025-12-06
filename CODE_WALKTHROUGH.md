# Code Walkthrough: Ad Revenue Modeling Application

This document provides a detailed, professional explanation of the source code (`app.py`). It is designed to help you explain the "what," "why," and "how" of each component to a reviewer.

---

## 1. Imports and Setup

### Code Block
```python
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
...
```

### Explanation
*   **What it does**: Imports necessary libraries for data manipulation (`pandas`, `numpy`), visualization (`matplotlib`, `seaborn`, `plotly`), machine learning (`sklearn`, `statsmodels`), and the web framework (`streamlit`).
*   **Why it's needed**: These are the building blocks of the application. `joblib` is specifically used for saving/loading trained models efficiently.

---

## 2. Configuration & Constants

### Code Block
```python
st.set_page_config(page_title="Ad Revenue Modeling", layout="wide")
MODELS_DIR = "models"
feature_cols_default = [...]
```

### Explanation
*   **What it does**: Sets up the Streamlit page layout to "wide" mode for better visualization space. Defines the directory where models will be saved and lists the default features to be used.
*   **Why it's needed**: Centralizing configuration (like feature names) makes the code maintainable. If we want to add a feature later, we only change it in one place.

---

## 3. Data Loading & Imputation (`get_dataset`)

### Code Block
```python
@st.cache_data
def get_dataset(uploaded_file=None, impute_mode="Drop rows"):
    ...
    if impute_mode == "Fill with median":
        df[col] = df[col].fillna(df[col].median())
    elif impute_mode == "KNN Imputation":
        ...
```

### Explanation
*   **What it does**: Loads the CSV data. It uses `@st.cache_data` to store the result in memory so the app doesn't reload the file on every interaction (speed optimization). It also handles missing values based on the user's choice (Mean, Median, KNN, etc.).
*   **Why it's needed**: Real-world data is often messy. This function ensures the data is clean and ready for the model. The caching mechanism is crucial for a responsive user experience.

---

## 4. Feature Engineering

### Code Block
```python
# Feature Engineering
df["engagement_rate"] = (df["likes"] + df["comments"]) / df["views"]
df["retention_rate"] = df["watch_time_per_view"] / df["video_length_minutes"]
```

### Explanation
*   **What it does**: Creates new meaningful variables from existing ones. For example, `engagement_rate` combines likes and comments relative to views.
*   **Why it's needed**: Raw data isn't always enough. "Domain knowledge" (like knowing that retention matters for ads) is injected into the model through these engineered features, often improving accuracy.

---

## 5. Custom Model Wrapper (`StatsmodelsOLS`)

### Code Block
```python
class StatsmodelsOLS(BaseEstimator, RegressorMixin):
    def fit(self, X, y): ...
    def predict(self, X): ...
```

### Explanation
*   **What it does**: Wraps the `statsmodels` OLS class to look and behave like a `scikit-learn` model.
*   **Why it's needed**: `scikit-learn`'s cross-validation tools (`cross_val_score`) expect a specific interface (`fit` returning `self`, `predict` method). `statsmodels` does not follow this by default. This wrapper bridges the gap, allowing us to evaluate OLS using the same rigorous cross-validation as the other models.

---

## 6. Training Pipeline (`train_pipeline`)

### Code Block
```python
@st.cache_resource
def train_pipeline(df, feature_cols, target_col):
    ...
    preprocessor = ColumnTransformer(...)
    linreg = make_pipeline(preprocessor, LinearRegression())
    ...
    cv_scores = cross_val_score(m, X_train, y_train, cv=5, scoring="r2")
```

### Explanation
*   **What it does**: This is the core engine.
    1.  **Preprocessing**: Standardizes numerical data (scales them to unit variance) and One-Hot Encodes categorical data (converts text categories to numbers).
    2.  **Model Definition**: Defines multiple models (Linear, Ridge, Lasso, ElasticNet).
    3.  **Training & Evaluation**: Fits models, calculates R2/RMSE, and performs 5-fold Cross-Validation.
    4.  **Saving**: Dumps the trained models to disk.
*   **Why it's needed**: It automates the entire ML workflow. Using a `Pipeline` ensures that preprocessing steps are applied correctly during both training and prediction, preventing "data leakage."

---

## 7. Visualization (`plot_section`)

### Code Block
```python
def plot_section(y_true, y_pred, title_prefix, indices=None):
    ...
    fig2 = px.scatter(..., title="Residuals vs Predicted")
    fig2.update_yaxes(range=[-1, 1])
```

### Explanation
*   **What it does**: Generates standard plots for regression analysis: "Actual vs Predicted" and "Residuals vs Predicted".
*   **Why it's needed**: Visuals tell the story better than numbers. The "Residuals" plot helps us check if the model is biased (e.g., consistently overestimating). We fixed the Y-axis to [-1, 1] to clearly see small errors.

---

## 8. Main Application Logic

### Code Block
```python
if page == "1. Train & Evaluate":
    ...
    st.sidebar.multiselect("Select features to include", ...)
    ...
else:
    # Prediction Page
    ...
```

### Explanation
*   **What it does**: Controls the flow of the app. It creates the sidebar navigation and renders either the Training dashboard or the Prediction interface based on user selection.
*   **Why it's needed**: This provides the user interface (UI). It connects the backend logic (functions above) to the frontend widgets (buttons, sliders, file uploaders), making the tool interactive and usable by non-programmers.
