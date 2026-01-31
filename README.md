# Ad Revenue Detection & Model Reconstruction

## 🚀 Project Overview
**Objective**: To reverse-engineer the hidden mathematical formula of YouTube's revenue system using advanced Linear Regression techniques.
**Role**: A "Revenue Simulator" and forensic analysis tool for content creators.

This project goes beyond simple prediction. It uses **Polynomial Transformers** and **Regularized Models (Lasso/Ridge)** to mathematically "crack" the platform's logic, moving from a naive 26% accuracy to **98% precision**.

---

## 📊 Key Findings (The "Detection")

### 1. The "Cheat" Variable
*   **Discovery**: Raw interaction metrics (Views, Likes, Comments) are **poor predictors** on their own (R² ≈ 0.26).
*   **The Breakthrough**: `watch_time_minutes` acts as a "Data Leakage" variable with a **0.988 correlation** to revenue.
*   **Conclusion**: Revenue is almost deterministically tied to **Duration** rather than just Volume.

### 2. The "Polynomial Cracking"
*   **Linear Failure**: Standard models failed to capture the exponential nature of viral revenue.
*   **The Solution**: By applying a **Polynomial Transformer (Degree 2)**, the model reconstructed the non-linear "Duration-First" logic, boosting accuracy to **0.979**.

### 3. Feature Engineering Wins
*   **Retention Rate**: Identifying quality over quantity.
*   **Interaction Terms**: `Views * Retention` proved to be the ultimate driver of income.

---

## 🛠️ Technical Architecture

### The "Factory" Pipeline
Information is processed through a strict Scikit-Learn pipeline to prevent Data Leakage:
1.  **Preprocessing**: `ColumnTransformer` handles scaling (StandardScaler) and encoding (OneHotEncoder).
2.  **Feature Engineering**: `PolynomialFeatures` creates interaction terms (e.g., `Views * Likes`).
3.  **Target Transformation**: `TransformedTargetRegressor` with `func=np.log1p` handles the skewed "Power Law" distribution of financial data.

### Model Suite
*   **Linear Regression**: The Baseline.
*   **Ridge/Lasso/ElasticNet**: Regularized models to handle **Multicollinearity** (Stable coefficients).
*   **Gamma GLM**: Specialized for strictly positive, skewed financial data.
*   **Statsmodels Wrapper**: Custom class to extract P-values and confidence intervals.

---

## 📦 Installation & Usage

1.  **Clone the repository**:
    ```bash
    git clone [repo_url]
    ```
2.  **Install requirements**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Streamlit App**:
    ```bash
    streamlit run app.py
    ```

---

## 📂 Project Structure
*   `app.py`: Main application with model pipeline and visualization.
*   `YouTube_Project_Review.pptx`: Comprehensive presentation of findings.
*   `PROJECT_REPORT.md` / `Codebase_Teacher_Guide.md`: Detailed technical documentation.
