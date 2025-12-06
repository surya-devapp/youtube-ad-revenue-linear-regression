import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv("youtube_ad_revenue_dataset.csv")

df = df.dropna(subset=["watch_time_minutes", "ad_revenue_usd"])

X = df["watch_time_minutes"]
y = df["ad_revenue_usd"]

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

print(model.summary())
