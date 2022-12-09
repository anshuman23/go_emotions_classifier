import pandas as pd
import numpy as np

df = pd.read_csv('output/FB_combined_results_incl_emotion.csv')

print(df)

#df['post_emotion'] = df['post_emotion'].fillna('unknown')
#df['comment_emotion'] = df['comment_emotion'].fillna('unknown')

df['post_political'] = df['post_political'].fillna(-100)
df['comment_political'] = df['comment_political'].fillna(-100)
df['post_ideology'] = df['post_ideology'].fillna(-100)
df['comment_ideology'] = df['comment_ideology'].fillna(-100)

print(df)

df.to_csv('FB_FINAL_EVERYTHING.csv', index=False)
