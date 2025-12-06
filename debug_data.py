import pandas as pd
import numpy as np

# Load dataset
try:
    df = pd.read_csv("youtube_ad_revenue_dataset.csv")
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

target_col = "ad_revenue_usd"
feature_cols = [
    "views",
    "likes",
    "comments",
    "watch_time_minutes",
    "video_length_minutes",
    "subscribers",
    "category",
    "device",
    "country",
]

# Check if target is in features
if target_col in feature_cols:
    print(f"WARNING: Target column '{target_col}' is in feature columns!")

# Check correlations
print("\nCorrelations with target:")
numeric_df = df.select_dtypes(include=[np.number])
corrs = numeric_df.corr()[target_col].sort_values(ascending=False)
print(corrs)

# Check for perfect predictors
for col in feature_cols:
    if col in df.columns and df[col].dtype in [np.number]:
        corr = df[col].corr(df[target_col])
        if abs(corr) > 0.99:
            print(f"\nWARNING: Feature '{col}' has extremely high correlation ({corr:.4f}) with target!")

# Check for duplicates
print("\nChecking for duplicate columns...")
print(df.head())
