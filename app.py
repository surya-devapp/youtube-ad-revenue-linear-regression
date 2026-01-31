import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import KNNImputer
from sklearn.linear_model import (
    LinearRegression,
    RidgeCV,
    LassoCV,
    ElasticNetCV,   
)
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import PolynomialFeatures
import os










class StatsmodelsOLS(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.model = None
        self.results = None

    def fit(self, X, y):
        self.model = sm.OLS(y, X)
        self.results = self.model.fit()
        return self

    def predict(self, X):
        return self.results.predict(X)

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Ad Revenue Modeling", layout="wide")
sns.set_style("whitegrid")

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

feature_cols_default = [
    "views",
    "comments",
    "video_length_minutes",
    "subscribers",
    "category",
    "device",
    "country",
    "retention_volume",
]
target_col_default = "ad_revenue_usd"

# -----------------------------------------------------------------------------
# UTILS
# -----------------------------------------------------------------------------
@st.cache_data
def get_dataset(uploaded_file=None, impute_mode="Drop rows"):

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif os.path.exists("youtube_ad_revenue_dataset.csv"):
        df = pd.read_csv("youtube_ad_revenue_dataset.csv")
    else:
        return None
    
    # Duplicate removal (Requirement: Remove ~2% duplicates)
    # Data Reduction Tracking
    stats = {}
    stats["Original Rows"] = len(df)
    
    # Duplicate removal
    df = df.drop_duplicates()
    stats["After Duplicates Removal"] = len(df)
    
    # Track Nulls (before imputation/drop)
    null_counts = df.isnull().sum()
    stats["Null Counts"] = null_counts[null_counts > 0].to_dict()
    
    if impute_mode == "Drop rows":
        # Drop nulls
        df = df.dropna()
        stats["After Drop Nulls"] = len(df)
    else:
        # Imputation
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                if impute_mode == "Fill with median":
                    df[col] = df[col].fillna(df[col].median())
                elif impute_mode == "Fill with 0":
                    df[col] = df[col].fillna(0)
                elif impute_mode == "KNN Imputation":
                    # We will handle KNN batch-wise or after this loop for all numeric cols
                    # For now, just mark it to be done
                    pass 
                else:
                    df[col] = df[col].fillna(df[col].mean())
            else:
                if len(df[col].mode()) > 0:
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna("Unknown")
        
        if impute_mode == "KNN Imputation":
            # Apply KNN to numeric columns
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) > 0:
                imputer = KNNImputer(n_neighbors=3)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        # In imputation mode, "final" is just current length (usually same as original)
        stats["After Drop Nulls"] = len(df) 
    
    # Feature Engineering
    # Ratios
    df["likes_per_view"] = df["likes"] / df["views"]
    df["comments_per_view"] = df["comments"] / df["views"]
    df["engagement_rate"] = (df["likes"] + df["comments"]) / df["views"]
    
    # Quality proxies
    # watch_time_minutes is total watch time.
    df["watch_time_per_view"] = df["watch_time_minutes"] / df["views"].replace(0, 1)
    df["retention_rate"] = df["watch_time_per_view"] / df["video_length_minutes"].replace(0, 1)
    df["retention_volume"] = df["retention_rate"] * df["views"]
    
    return df, stats

@st.cache_resource
def train_pipeline(df, feature_cols, target_col):
    X = df[feature_cols]

@st.cache_resource
def train_pipeline(df, feature_cols, target_col):
    X = df[feature_cols]
    y = df[target_col].values

    numeric_features = [c for c in feature_cols if df[c].dtype in ['int64', 'float64']]
    categorical_features = [c for c in feature_cols if df[c].dtype == 'object']

    preprocessor = ColumnTransformer(
        transformers=[
            ("poly", PolynomialFeatures(degree=2, include_bias=False), numeric_features),
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        verbose_feature_names_out=False,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    alphas = np.logspace(-3, 3, 30)

    model_logic = LinearRegression() # or RidgeCV()
    wrapped_model = TransformedTargetRegressor(
        regressor=model_logic, 
        func=np.log1p,         # Log transform for training
        inverse_func=np.expm1 # Convert back to dollars for evaluation
    )
    linreg = make_pipeline(preprocessor, wrapped_model)
    ridge_pipe = make_pipeline(preprocessor, RidgeCV(alphas=alphas, cv=5))
    lasso_pipe = make_pipeline(preprocessor, LassoCV(alphas=alphas, cv=5, random_state=42))
    elastic_pipe = make_pipeline(
        preprocessor,
        ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], alphas=alphas, cv=5, random_state=42),
    )

    models_sklearn = {
        "LinearRegression (sklearn)": linreg,
        "Ridge": ridge_pipe,
        "Lasso": lasso_pipe,
        "Elastic Net": elastic_pipe,
    }

    results = []
    for name, m in models_sklearn.items():
        m.fit(X_train, y_train)
        # Cross-validation
        cv_scores = cross_val_score(m, X_train, y_train, cv=5, scoring="r2")
        
        y_pred_train = m.predict(X_train)
        y_pred_test = m.predict(X_test)
        
        results.append({
            "Model": name,
            "Train R2": r2_score(y_train, y_pred_train),
            "Test R2": r2_score(y_test, y_pred_test),
            "CV R2 (Mean)": cv_scores.mean(),
            "CV R2 (Std)": cv_scores.std(),
            "Train RMSE": root_mean_squared_error(y_train, y_pred_train),
            "Test RMSE": root_mean_squared_error(y_test, y_pred_test),
            "y_pred_train": y_pred_train,
            "Train RMSE": root_mean_squared_error(y_train, y_pred_train),
            "Test RMSE": root_mean_squared_error(y_test, y_pred_test),
            "Train MAE": mean_absolute_error(y_train, y_pred_train),
            "Test MAE": mean_absolute_error(y_test, y_pred_test),
            "y_pred_train": y_pred_train,
            "y_pred_test": y_pred_test,
        })

    # Statsmodels
    preprocessor.fit(X_train, y_train)
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    try:
        feature_names_out = preprocessor.get_feature_names_out()
    except:
        feature_names_out = [f"feat_{i}" for i in range(X_train_proc.shape[1])]
    
    X_train_sm = sm.add_constant(pd.DataFrame(X_train_proc, columns=feature_names_out, index=X_train.index))
    X_test_sm = sm.add_constant(pd.DataFrame(X_test_proc, columns=feature_names_out, index=X_test.index))
    
    # Use wrapper for CV
    ols_wrapper = StatsmodelsOLS()
    cv_scores_ols = cross_val_score(ols_wrapper, X_train_sm, y_train, cv=5, scoring="r2")

    ols_model = sm.OLS(y_train, X_train_sm).fit()
    y_pred_train_ols = ols_model.predict(X_train_sm)
    y_pred_test_ols = ols_model.predict(X_test_sm)
    results.append({
        "Model": "OLS (statsmodels)",
        "Train R2": r2_score(y_train, y_pred_train_ols),
        "Test R2": r2_score(y_test, y_pred_test_ols),
        "CV R2 (Mean)": cv_scores_ols.mean(),
        "CV R2 (Std)": cv_scores_ols.std(),
        "Train RMSE": root_mean_squared_error(y_train, y_pred_train_ols),
        "Test RMSE": root_mean_squared_error(y_test, y_pred_test_ols),
        "y_pred_train": y_pred_train_ols,
        "Train RMSE": root_mean_squared_error(y_train, y_pred_train_ols),
        "Test RMSE": root_mean_squared_error(y_test, y_pred_test_ols),
        "Train MAE": mean_absolute_error(y_train, y_pred_train_ols),
        "Test MAE": mean_absolute_error(y_test, y_pred_test_ols),
        "y_pred_train": y_pred_train_ols,
        "y_pred_test": y_pred_test_ols,
    })



    # Auto-save
    joblib.dump(linreg, os.path.join(MODELS_DIR, "linreg.pkl"))
    joblib.dump(ridge_pipe, os.path.join(MODELS_DIR, "ridge.pkl"))
    joblib.dump(lasso_pipe, os.path.join(MODELS_DIR, "lasso.pkl"))
    joblib.dump(elastic_pipe, os.path.join(MODELS_DIR, "elasticnet.pkl"))
    ols_model.save(os.path.join(MODELS_DIR, "ols_sm.pickle"))

    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_cols.pkl"))
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.pkl"))
    joblib.dump(X.iloc[:5], os.path.join(MODELS_DIR, "X_sample.pkl"))

    return results, models_sklearn, ols_model, preprocessor, X_train, X_test, y_train, y_test


import plotly.express as px
import plotly.graph_objects as go

def plot_section(y_true, y_pred, title_prefix, indices=None):
    residuals = y_true - y_pred

    col1, col2 = st.columns(2)

    with col1:
        # Actual vs Predicted
        fig1 = px.scatter(
            x=y_true, 
            y=y_pred, 
            labels={'x': 'Actual', 'y': 'Predicted'},
            title=f"{title_prefix}: Actual vs Predicted"
        )
        # Add perfect prediction line
        mn, mx = float(y_true.min()), float(y_true.max())
        fig1.add_shape(
            type="line", line=dict(dash='dash', color='red'),
            x0=mn, y0=mn, x1=mx, y1=mx
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Residuals vs Predicted
        df_res = pd.DataFrame({
            "Predicted": y_pred,
            "Residuals": residuals
        })
        if indices is not None:
            df_res["Index"] = indices
            hover_data = ["Index"]
        else:
            hover_data = None

        fig2 = px.scatter(
            df_res,
            x="Predicted", 
            y="Residuals", 
            hover_data=hover_data,
            title=f"{title_prefix}: Residuals vs Predicted"
        )
        fig2.add_hline(y=0, line_dash="dash", line_color="red")
        #fig2.update_yaxes(range=[-0.5, 0.5])
        st.plotly_chart(fig2, use_container_width=True)

    # Residual Distribution
    st.markdown("**Residual distribution**")
    fig3 = px.histogram(
        x=residuals, 
        nbins=50, 
        title=f"{title_prefix}: Residuals Distribution",
        labels={'x': 'Residuals'}
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Actual vs Predicted Comparison (Line Chart)
    st.markdown("**Actual vs Predicted Comparison (First 100 samples)**")
    # Limit to 100 for readability
    limit = 100
    y_true_sub = y_true[:limit]
    y_pred_sub = y_pred[:limit]
    
    comp_df = pd.DataFrame({
        "Index": range(len(y_true_sub)),
        "Actual": y_true_sub,
        "Predicted": y_pred_sub
    })
    
    comp_melt = comp_df.melt(id_vars="Index", var_name="Type", value_name="Value")
    
    fig4 = px.line(
        comp_melt, 
        x="Index", 
        y="Value", 
        color="Type", 
        markers=True,
        title=f"{title_prefix}: Actual vs Predicted (Subset)",
        color_discrete_map={"Actual": "blue", "Predicted": "orange"}
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Distribution Comparison (Actual vs Predicted)
    st.markdown(f"**{title_prefix}: Distribution of Actual vs Predicted**")
    # visualising the distribution of the actual and predicted values
    # using a histogram to see how well the model captures the data distribution
    
    fig5 = go.Figure()
    fig5.add_trace(go.Histogram(x=y_true, name="Actual", opacity=0.6, marker_color='blue'))
    fig5.add_trace(go.Histogram(x=y_pred, name="Predicted", opacity=0.6, marker_color='orange'))
    fig5.update_layout(barmode='overlay', title=f"{title_prefix}: Actual vs Predicted Distribution")
    st.plotly_chart(fig5, use_container_width=True)


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["1. Train & Evaluate", "2. Predict Revenue"])

# -----------------------------------------------------------------------------
# PAGE 1: TRAIN & EVALUATE
# -----------------------------------------------------------------------------
if page == "1. Train & Evaluate":
    st.title("Train & Evaluate Revenue Models")

    st.sidebar.header("Data options")
    uploaded_file = st.sidebar.file_uploader("Upload CSV (Optional)", type="csv")
    
    impute_mode = "Drop rows"
    
    df_result = get_dataset(uploaded_file, impute_mode)
    if df_result is None: # Handle case where file not found or returned None
        st.info("Upload a CSV to begin.")
        st.stop()
    
    # Handle tuple unpack (df, stats)
    if isinstance(df_result, tuple):
        df, stats = df_result
        
        # Display Data Processing Stats if available
        if stats:
             st.write("### Data Processing Stats")
             duplicates_removed = stats.get("Original Rows", 0) - stats.get("After Duplicates Removal", 0)
             total_removed = stats.get("Original Rows", 0) - stats.get("After Drop Nulls", 0)
             reduction_pct = (total_removed / stats.get("Original Rows", 1)) * 100
             
             col1, col2, col3, col4 = st.columns(4)
             col1.metric("Original", f"{stats.get('Original Rows', 0):,}")
             col2.metric("After Dupes", f"{stats.get('After Duplicates Removal', 0):,}", delta=f"-{duplicates_removed}")
             col3.metric("Final (Clean)", f"{stats.get('After Drop Nulls', 0):,}", delta=f"-{stats.get('After Duplicates Removal', 0) - stats.get('After Drop Nulls', 0)}")
             col4.metric("Total Reduction", f"{reduction_pct:.1f}%", delta=f"-{total_removed} rows", delta_color="inverse")
             
             if duplicates_removed > 0:
                  st.toast(f"Removed {duplicates_removed} duplicate rows.", icon="🧹")
             
             # Show Null Counts if any
             null_counts = stats.get("Null Counts", {})
             if null_counts:
                 st.caption("Rows with Null Values (by Column):")
                 st.dataframe(pd.DataFrame(list(null_counts.items()), columns=["Column", "Null Count"]).set_index("Column").T)
    else:
        df = df_result # Fallback for legacy behavior

    if df is None:
        st.info("Upload a CSV to begin.")
        st.stop()

    st.subheader("Raw data preview")
    st.caption(f"Total rows: {len(df)}")
    st.dataframe(df.head())

    # Feature selection
    st.sidebar.subheader("Feature Selection")
    target_col = target_col_default
    feature_cols = st.sidebar.multiselect(
        "Select features to include",
        options=[c for c in df.columns if c != target_col and c not in ["video_id", "date"]],
        default=[c for c in feature_cols_default if c in df.columns]
    )

    if not feature_cols:
        st.error("Please select at least one feature.")
        st.stop()

    # -------------------------------------------------------------------------
    # 1. EDA & Data Cleaning
    # -------------------------------------------------------------------------
    st.subheader("1. Exploratory Data Analysis")
    # (Imputation is done in get_dataset)

    # Correlation Matrix (Numerical)
    # Correlation Matrix (Numerical)
    st.markdown("**Correlation Matrix (Numerical)**")
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    # Check for high correlation with target
    if target_col in corr_matrix.columns:
        high_corr_cols = corr_matrix[target_col][abs(corr_matrix[target_col]) > 0.8].drop(target_col, errors='ignore')
        if not high_corr_cols.empty:
            st.warning(f"⚠️ Suspiciously high correlation detected with target '{target_col}':")
            for col, val in high_corr_cols.items():
                st.write(f"- **{col}**: {val:.4f}")
            st.info("High correlation (>0.8) often indicates data leakage (the feature might be a proxy for the target). Consider removing these features in the sidebar.")

    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)

    # Target Distribution
    st.markdown("**Target Distribution**")
    fig_dist, ax_dist = plt.subplots(figsize=(10, 4))
    sns.histplot(df[target_col], kde=True, ax=ax_dist)
    st.pyplot(fig_dist)

    # Outlier Detection (Boxplots)
    st.markdown("**Outlier Detection (Boxplots)**")
    numeric_cols = numeric_df.columns
    # Allow user to select column to visualize outliers
    box_col = st.selectbox("Select column for Outlier check", numeric_cols)
    fig_box, ax_box = plt.subplots(figsize=(10, 2))
    sns.boxplot(x=df[box_col], ax=ax_box, color="orange")
    st.pyplot(fig_box)


    # 2. Preprocessing & Split & Train
    # -------------------------------------------------------------------------
    st.subheader("2. Model Training")
    
    with st.spinner("Training models..."):
        results, models_sklearn, ols_model, preprocessor, X_train, X_test, y_train, y_test = train_pipeline(df, feature_cols, target_col)
    
    st.success("Models trained and saved automatically.")

    # Evaluation table
    st.subheader("Evaluation metrics")
    eval_df = pd.DataFrame(
        [
            {
                "Model": r["Model"],
                "Train R2": r["Train R2"],
                "CV R2 (Mean)": r["CV R2 (Mean)"],
                "CV R2 (Std)": r["CV R2 (Std)"],
                "Test R2": r["Test R2"],
                "Train RMSE": r["Train RMSE"],
                "Test RMSE": r["Test RMSE"],
                "Train MAE": r["Train MAE"],
                "Test MAE": r["Test MAE"],
            }
            for r in results
        ]
    )
    st.dataframe(
        eval_df.style.format(
            {
                "Train R2": "{:.3f}",
                "CV R2 (Mean)": "{:.3f}",
                "CV R2 (Std)": "{:.3f}",
                "Test R2": "{:.3f}",
                "Train RMSE": "{:.3f}",
                "Train RMSE": "{:.3f}",
                "Test RMSE": "{:.3f}",
                "Train MAE": "{:.3f}",
                "Test MAE": "{:.3f}",
            }
        )
    )

    # Visualization
    st.subheader("Residuals & Predictions")

    model_names = [r["Model"] for r in results]
    selected_model = st.selectbox("Select model to visualize", model_names)

    res = next(r for r in results if r["Model"] == selected_model)

    # Prepare data for download
    # y_train and y_test are numpy arrays, so we use X_train.index and X_test.index
    train_res_df = pd.DataFrame({
        "Actual": y_train,
        "Predicted": res["y_pred_train"]
    }, index=X_train.index)

    test_res_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": res["y_pred_test"]
    }, index=X_test.index)

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.download_button(
            label=f"Download Train Predictions",
            data=train_res_df.to_csv(index=True).encode("utf-8"),
            file_name=f"Train_Actual_vs_Predicted_{selected_model}.csv",
            mime="text/csv",
        )
    with col_d2:
        st.download_button(
            label=f"Download Test Predictions",
            data=test_res_df.to_csv(index=True).encode("utf-8"),
            file_name=f"Test_Actual_vs_Predicted_{selected_model}.csv",
            mime="text/csv",
        )
    
    # Model download
    model_file_map = {
        "LinearRegression (sklearn)": "linreg.pkl",
        "Ridge": "ridge.pkl",
        "Lasso": "lasso.pkl",
        "Elastic Net": "elasticnet.pkl",
        "OLS (statsmodels)": "ols_sm.pickle",

    }
    
    with col_d3:
        if selected_model in model_file_map:
            model_path = os.path.join(MODELS_DIR, model_file_map[selected_model])
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    st.download_button(
                        label=f"Download Model ({selected_model})",
                        data=f,
                        file_name=model_file_map[selected_model],
                        mime="application/octet-stream",
                    )
            else:
                st.warning("Model file not found.")

    tab1, tab2 = st.tabs(["Train set", "Test set"])

    with tab1:
        plot_section(y_train, res["y_pred_train"], "Train", indices=X_train.index)

    with tab2:
        plot_section(y_test, res["y_pred_test"], "Test", indices=X_test.index)

    # Feature Importance
    st.subheader("Feature Importance")
    try:
        if "Lasso" in models_sklearn:
            model = models_sklearn["Lasso"]
            coefs = model.named_steps['lassocv'].coef_
            feats = model.named_steps['columntransformer'].get_feature_names_out()
            
            fi_df = pd.DataFrame({"Feature": feats, "Coefficient": coefs})
            fi_df = fi_df.sort_values(by="Coefficient", key=abs, ascending=False).head(20)
            
            fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
            sns.barplot(data=fi_df, x="Coefficient", y="Feature", ax=ax_fi, palette="viridis", hue="Feature", legend=False)
            ax_fi.set_title("Top 20 Feature Coefficients (Lasso)")
            st.pyplot(fig_fi)
    except Exception as e:
        st.info(f"Could not plot feature importance: {e}")
    
    best_name = eval_df.sort_values("Test RMSE").iloc[0]["Model"]
    st.session_state["best_model_name"] = best_name
    st.session_state["feature_cols"] = feature_cols



# -----------------------------------------------------------------------------
# PAGE 2: PREDICT
# -----------------------------------------------------------------------------
else:
    st.title("Predict Ad Revenue")

    # Load models
    # Load data
    try:
        # We don't need stats here really, just the df to get columns/types if needed
        # Or we load from cache (which now returns tuple)
        df_result = get_dataset(None, "Drop rows") 
        if isinstance(df_result, tuple):
            df_loaded, _ = df_result
        else:
            df_loaded = df_result
            
        feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
        # We don't strictly need train_data.pkl for prediction if we have the preprocessor
        # But we might use it for stats
        linreg = joblib.load(os.path.join(MODELS_DIR, "linreg.pkl"))
        ridge_pipe = joblib.load(os.path.join(MODELS_DIR, "ridge.pkl"))
        lasso_pipe = joblib.load(os.path.join(MODELS_DIR, "lasso.pkl"))
        elastic_pipe = joblib.load(os.path.join(MODELS_DIR, "elasticnet.pkl"))
        ols_model = sm.load(os.path.join(MODELS_DIR, "ols_sm.pickle"))

        preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))
        X_sample = joblib.load(os.path.join(MODELS_DIR, "X_sample.pkl"))
    except Exception as e:
        st.error(f"Models not found. Please go to 'Train & Evaluate' and train + save models first. Error: {e}")
        st.stop()

    models = {
        "LinearRegression (sklearn)": linreg,
        "Ridge": ridge_pipe,
        "Lasso": lasso_pipe,
        "Elastic Net": elastic_pipe,
        "OLS (statsmodels)": ols_model,

    }

    mode = st.sidebar.radio("Input type", ["Single input", "CSV upload"])

    default_model = st.session_state.get("best_model_name", "Elastic Net")
    model_name = st.selectbox("Choose model for prediction", list(models.keys()),
                              index=list(models.keys()).index(default_model)
                              if default_model in models else 0)

    st.markdown("### Input features")

    # Helper function to compute engineered features (Defined here to be used in both modes)
    def add_engineered_features(d):
        # Safe division helper
        def safe_div(a, b):
            return a / b if b != 0 else 0
        
        d["likes_per_view"] = d.apply(lambda r: safe_div(r.get("likes",0), r.get("views",1)), axis=1)
        d["comments_per_view"] = d.apply(lambda r: safe_div(r.get("comments",0), r.get("views",1)), axis=1)
        d["engagement_rate"] = d.apply(lambda r: safe_div(r.get("likes",0)+r.get("comments",0), r.get("views",1)), axis=1)
        
        d["watch_time_per_view"] = d.apply(lambda r: safe_div(r.get("watch_time_minutes",0), r.get("views",1)), axis=1)
        d["retention_rate"] = d.apply(lambda r: safe_div(r.get("watch_time_per_view",0), r.get("video_length_minutes",1)), axis=1)
        d["retention_volume"] = d["retention_rate"] * d.get("views", 0)
        return d

    if mode == "Single input":
        cols = st.columns(3)
        inputs = {}
        
        # Base columns needed for feature engineering
        base_cols = ["views", "likes", "comments", "watch_time_minutes", "video_length_minutes", "subscribers", "category", "device", "country"]
        # Filter feature_cols to only show base columns + any other non-engineered ones
        # We assume known engineered columns: likes_per_view, comments_per_view, engagement_rate, watch_time_per_view, retention_rate, retention_volume
        engineered_cols = ["likes_per_view", "comments_per_view", "engagement_rate", "watch_time_per_view", "retention_rate", "retention_volume"]
        
        # Display inputs for necessary columns
        # Note: If feature_cols contains other columns not in base, we should show them too.
        display_cols = [c for c in feature_cols if c not in engineered_cols]
        # Also ensure base_cols are present if they are needed for calculation but not in feature_cols (unlikely but possible)
        # For simplicity, we just iterate through feature_cols and if it is engineered, we skip asking user.
        # BUT we must ask for base cols if they are missing from feature_cols? 
        # Actually our models trained on feature_cols. So we only need to provide feature_cols.
        # If feature_cols HAS retention_volume, we need views, watch_time... to calculate it.
        
        # Let's ask for ALL base columns to be safe, then compute, then filter to feature_cols
        for i, col_name in enumerate(base_cols):
             with cols[i % 3]:
                if col_name in ["category", "device", "country"]:
                     inputs[col_name] = st.text_input(col_name, value="Unknown") # Simplified
                else:
                     inputs[col_name] = st.number_input(col_name, min_value=0.0, value=100.0, step=1.0)
        
        if st.button("Predict revenue"):
            # Create DataFrame
            x_new = pd.DataFrame([inputs])
            
            # Ensure types match
            for c in x_new.columns:
                if c in X_sample.select_dtypes(include=[np.number]).columns:
                    x_new[c] = pd.to_numeric(x_new[c])
            
            # Compute engineered features
            x_new = add_engineered_features(x_new)
            
            # Keep only required features
            x_new_final = x_new[feature_cols]

            if model_name == "OLS (statsmodels)":
                # Transform
                x_new_proc = preprocessor.transform(x_new_final)
                try:
                    feature_names_out = preprocessor.get_feature_names_out()
                except:
                    feature_names_out = [f"feat_{i}" for i in range(x_new_proc.shape[1])]
                
                x_sm = sm.add_constant(pd.DataFrame(x_new_proc, columns=feature_names_out), has_constant="add")
                pred = models[model_name].predict(x_sm)[0]
            else:
                pred = models[model_name].predict(x_new_final)[0]

            st.metric("Predicted ad_revenue_usd", f"{pred:,.2f}")

    else:
        file = st.file_uploader("Upload CSV with feature columns", type="csv")
        if file is not None:
            df_new = pd.read_csv(file)
            st.write("Preview:", df_new.head())
            
            # Apply feature engineering to uploaded CSV
            try:
                df_new = add_engineered_features(df_new)
                st.success("Automatically calculated derived features (retention_volume, etc.)")
            except Exception as e:
                st.warning(f"Could not calculate derived features: {e}. Expecting raw columns like views, likes, watch_time_minutes.")
            
            # Handle Missing Values (Models cannot handle NaNs)
            if df_new.isnull().sum().sum() > 0:
                init_len = len(df_new)
                df_new = df_new.dropna()
                st.warning(f"⚠️ Dropped {init_len - len(df_new)} rows containing missing values (Model cannot handle NaNs).")
                
                if len(df_new) == 0:
                    st.error("All rows were dropped due to missing values. Please check your CSV.")
                    st.stop()
            else:
                st.success("No missing values found.")

            missing = [c for c in feature_cols if c not in df_new.columns]
            if missing:
                st.error(f"CSV is missing columns: {missing}")
            else:
                X_new = df_new[feature_cols] # DataFrame

                if st.button("Predict revenue for all rows"):
                    if model_name == "OLS (statsmodels)":
                        X_new_proc = preprocessor.transform(X_new)
                        try:
                            feature_names_out = preprocessor.get_feature_names_out()
                        except:
                            feature_names_out = [f"feat_{i}" for i in range(X_new_proc.shape[1])]
                        
                        X_sm_new = sm.add_constant(pd.DataFrame(X_new_proc, columns=feature_names_out), has_constant="add")
                        preds = models[model_name].predict(X_sm_new)
                    else:
                        preds = models[model_name].predict(X_new)

                    df_new["predicted_ad_revenue_usd"] = preds
                    st.dataframe(df_new.head())
                    
                    # Score calculation might fail if we don't have y_all loaded or if it's not comparable
                    # Skipping score for now or using dummy
                    st.success("Predictions generated.")

                    st.download_button(
                        "Download predictions as CSV",
                        data=df_new.to_csv(index=False).encode("utf-8"),
                        file_name=f"Predictions_{model_name}.csv",
                        mime="text/csv",
                    )
