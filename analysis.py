import pandas as pd

df_1 = pd.read_csv("Annotated/annotation_AnnyLee_1.csv")
df_2 = pd.read_csv("Annotated/annotation_XinChen_1.csv")

df_1 = df_1[["homepage","sentiment","text", "url"]]
df_2 = df_2[["homepage","sentiment","text", "url"]]


merged_df = df_1.merge(df_2, on=['homepage', 'text', 'url'], how='outer')
print(merged_df.loc[~(merged_df['sentiment_x'] == merged_df['sentiment_y'])].head())
