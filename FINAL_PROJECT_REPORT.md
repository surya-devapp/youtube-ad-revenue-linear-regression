# Final Project Report: YouTube Ad Revenue Prediction
## Reverse Engineering the Monetization Algorithm

### 1. Executive Summary
This project aimed to build a predictive model for YouTube Ad Revenue (`ad_revenue_usd`) based on video performance metrics. Unlike standard regression tasks, this project required handling **financial power-law data** (viral hits), strictly positive predictions, and complex feature engineering to mimic the platform's actual algorithm (e.g., "Retention Volume").

The final deliverable is a **Streamlit Web Application** that not only trains and evaluates multiple models but also provides a real-time prediction interface for creators.

---

### 2. Project Lifecycle & Methodology

#### Phase 1: Data Analysis & The "Leakage" Discovery
*   **Initial State:** We obtained a dataset of ~122k video records.
*   **The Problem:** Initial models showed an R² of **1.000 (Perfect Score)**. This is impossible in real life.
*   **The Findings:** We discovered "Data Leakage". The feature `watch_time_minutes` was so perfectly correlated with revenue that the model was just memorizing math, not learning patterns.
*   **The Pivot:** We shifted focus to **Feature Engineering**—creating ratios like `engagement_rate` and `retention_volume` to uncover *true* driver of revenue, rather than just raw volume.

#### Phase 2: Pipeline Architecture
We implemented a robust **Scikit-Learn Pipeline** to ensure reproducibility:
1.  **Preprocessing:**
    *   **Duplicate Removal:** Automatically detected and removed ~2% duplicate records to prevent bias.
    *   **Missing Values:** Handled via row dropping (for training) or smart imputation.
    *   **Categorical Encoding:** OneHotEncoding for `Country` and `Category`.
2.  **Transformation:**
    *   **Log-Transformation (`log1p`)**: Applied to the *Target Variable*. This was crucial (see Section 4).
    *   **Polynomial Features**: Added to capture non-linear growth (exponential viral growth).
3.  **Modeling:**
    *   We tested 4 distinct models: **Linear Regression, Ridge, Lasso, ElasticNet**.
    *   **Winner:** **ElasticNet** provided the best balance of accuracy and stability.

#### Phase 3: Application Development
*   Built an interactive **Streamlit App**.
*   **Features:** Automated training, detailed evaluation metrics (R², RMSE, MAE), interactive Visualizations (Residuals, Actual vs Predicted Distribution), and a Prediction Interface.

---



### 3. The "Outlier" Philosophy: Why We Kept Them
**This is the most critical decision of the project.**

In standard Data Science, "Outliers" are often seen as errors (e.g., a typo in a sensor reading) and are removed to clean the data.

**In YouTube Analytics, Outliers are NOT errors. They are VIRAL HITS.**

*   **The Reality:** YouTube revenue follows a **Power Law (Pareto Distribution)**. The top 1% of videos earn 90% of the money.
*   **The Decision:** If we removed outliers (videos with massive views/revenue), we would be **training the model to fail** on the most important videos. We would be teaching it to predict only "average" videos, effectively capping its utility.
*   **The Solution:** Instead of *removing* outliers, we **tamed** them:
    1.  **Log Transformation (`np.log1p`)**: We squashed the massive values into a manageable linear scale during training, allowing the model to learn from viral hits without being overwhelmed by their magnitude.
    2.  **Robust Evaluation**: We used **MAE (Mean Absolute Error)** alongside RMSE, as MAE is less sensitive to extreme outliers, giving us a saner view of "average" performance.

### 4. Conclusion
We have successfully built a **Production-Grade Revenue Estimator**. It respects the fundamental chaotic nature of social media data (viral hits) while providing a stable, usable tool for content creators.
