import pandas as pd

df = pd.read_csv("youtube_ad_revenue_dataset.csv")
print(df[["views", "watch_time_minutes", "video_length_minutes", "likes", "comments"]].head())
print("\nStats:")
print(df[["views", "watch_time_minutes", "video_length_minutes"]].describe())
