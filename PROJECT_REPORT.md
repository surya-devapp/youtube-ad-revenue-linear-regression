# Project Report: Ad Revenue Modeling

## 1. Methodology

### Data Preprocessing
*   **Missing Value Handling**: Implemented multiple strategies to handle missing data:
    *   *Drop rows*: For clean, complete data.
    *   *Fill with Mean/Median*: For simple statistical imputation.
    *   *Fill with 0*: Assuming missing values imply zero activity.
    *   *KNN Imputation*: Using K-Nearest Neighbors to estimate missing values based on similarity.
*   **Feature Engineering**: Constructed new features to capture quality and engagement:
    *   `engagement_rate`: (Likes + Comments) / Views
    *   `retention_rate`: (Watch Time / Views) / Video Length

### Model Selection & Training
*   **Algorithms**: We utilized a suite of linear models to capture relationships:
    *   *Linear Regression*: Baseline model.
    *   *Regularized Models (Ridge, Lasso, ElasticNet)*: To handle multicollinearity and prevent overfitting.
    *   *Statsmodels OLS*: For detailed statistical summaries (p-values, confidence intervals).
    *   *Gamma GLM*: To model revenue (which is strictly positive and often skewed) more naturally.
*   **Cross-Validation**: Implemented 5-fold Cross-Validation to ensure model stability and robustness.

## 2. Code Structure (`app.py`)

### Key Components
*   **`get_dataset()`**: Handles data loading, caching, and the selected imputation strategy.
*   **`train_pipeline()`**:
    *   Splits data into Train/Test sets.
    *   Builds Scikit-Learn pipelines with `StandardScaler`, `OneHotEncoder`, and the regressor.
    *   Calculates R2, RMSE, and Cross-Validation scores.
    *   Saves trained models to the `models/` directory.
*   **`StatsmodelsOLS` Wrapper**: A custom class that adapts the Statsmodels API to be compatible with Scikit-Learn's `cross_val_score`, enabling consistent evaluation metrics.
*   **`plot_section()`**: A reusable function for generating "Actual vs Predicted" and "Residuals" plots using Plotly.

## 3. Findings & Conclusion
*   **Dominant Feature**: The dataset exhibits a near-perfect linear relationship between `watch_time_minutes` and `ad_revenue_usd`.
*   **Model Performance**:
    *   With `watch_time_minutes`: R2 $\approx$ 1.0 (Perfect fit).
    *   Without `watch_time_minutes`: R2 $\approx$ 0.02 (Poor fit).
*   **Conclusion**: The project successfully demonstrates an end-to-end ML pipeline. However, the specific insights derived are heavily influenced by the synthetic nature of the dataset, where revenue is deterministically tied to watch time. Future work on real-world data would likely show more complex, non-linear relationships where `views` and `demographics` play a larger role.
