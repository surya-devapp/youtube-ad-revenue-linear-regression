# Ad Revenue Modeling App

This project is a Streamlit application for analyzing and predicting YouTube ad revenue based on various video metrics.

## Installation

1.  **Clone or Download** the project repository.
2.  **Create a Virtual Environment** (recommended):
    ```bash
    python -m venv venv
    ```
3.  **Activate the Virtual Environment**:
    *   **Windows**:
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```
    *   **Mac/Linux**:
        ```bash
        source venv/bin/activate
        ```
4.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Dependencies

The main packages used in this project are:
*   `streamlit`: For the web application interface.
*   `pandas`: For data manipulation and analysis.
*   `numpy`: For numerical operations.
*   `scikit-learn`: For machine learning models (Linear Regression, Ridge, Lasso, ElasticNet).
*   `statsmodels`: For statistical modeling (OLS, GLM).
*   `plotly`: For interactive visualizations.
*   `seaborn` & `matplotlib`: For static plotting.

## Project Structure

*   **`app.py`**: The main application file containing the Streamlit code, model training pipeline, and visualization logic.
*   **`requirements.txt`**: A text file listing all the Python libraries required to run the app.
*   **`youtube_ad_revenue_dataset.csv`**: The dataset file containing the video metrics and revenue data.
*   **`models/`**: A directory where trained models and preprocessors are saved automatically.

## Usage

To run the application, execute the following command in your terminal (ensure your virtual environment is active):

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`.

## Features

### 1. Train & Evaluate
*   **EDA**: View correlation matrices and target distributions.
*   **Feature Engineering**: Automatically calculates metrics like `engagement_rate` and `retention_rate`.
*   **Model Training**: Trains multiple models (Linear Regression, Ridge, Lasso, ElasticNet, OLS, Gamma GLM) with Cross-Validation.
*   **Evaluation**: Compares models using R2 and RMSE scores.
*   **Visualization**: Interactive plots for "Actual vs Predicted" and "Residuals".
*   **Downloads**: Download predictions and trained model files (`.pkl`).

### 2. Predict Revenue
*   **Single Input**: Enter values for a single video to get a revenue prediction.
*   **Batch Prediction**: Upload a CSV file to generate predictions for multiple videos at once.
