# Model Interpretation & Insights

## Key Findings

### 1. The "Perfect" Predictor: Watch Time
*   **Observation**: When including `watch_time_minutes` along with `views`, `likes`, and `comments`, the model achieves an **R2 score of ~1.0**.
*   **Insight**: This indicates an extremely strong linear relationship (likely synthetic) where Ad Revenue is almost directly calculated from Watch Time. In this dataset, `watch_time_minutes` acts as a "perfect predictor" or a proxy for the target variable.
*   **Multicollinearity**: There is high multicollinearity between these core metrics. The model relies heavily on `watch_time_minutes` because it contains the most information about the revenue generation formula used to create this data.

### 2. Performance Drop Without Watch Time
*   **Observation**: Removing `watch_time_minutes` from the feature set causes the **R2 score to drop drastically to ~0.024**.
*   **Insight**: This suggests that `views`, `likes`, and `comments` *alone* are insufficient to predict revenue in this specific dataset.
    *   This is counter-intuitive for real-world scenarios (where views usually correlate with revenue), but in this dataset, the "duration" factor is the critical missing link.
    *   Without knowing how long a video was watched, the model cannot estimate revenue effectively, implying the revenue formula is likely `Revenue = k * Watch_Time` rather than `Revenue = k * Views`.

### 3. Feature Engineering Impact
*   **Engineered Features**: We introduced `engagement_rate`, `likes_per_view`, and `retention_rate`.
*   **Effect**: While these features add semantic value, they do not fully compensate for the loss of the raw `watch_time_minutes` variable. The model struggles to reconstruct the total watch time from these rates alone without the base volume metrics being perfectly aligned.

## Recommendations
*   **For Accurate Prediction**: Keep `watch_time_minutes` if the goal is to reproduce the exact revenue values of this dataset.
*   **For Realistic Modeling**: If this were real-world data, we would expect `views` to have a stronger correlation. The current behavior highlights the synthetic nature of the data.
*   **Feature Selection**: Be cautious of "perfect" scores (R2=1.0). In a real deployment, relying on a single dominant feature can be risky if that data source becomes unavailable or changes definition.
