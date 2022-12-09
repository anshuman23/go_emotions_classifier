import pandas as pd
from tqdm import tqdm
import numpy as np

dfc = pd.read_pickle('data/FB_comments_results_incl_text.pkl')
dfp = pd.read_pickle('data/FB_posts_results_incl_text.pkl')

dfp = dfp[pd.isnull(dfp['post_id']) == False]
dfc = dfc[pd.isnull(dfc['post_id']) == False]

#print(dfp, dfc)

dfp = dfp[['post_id', 'political?', 'ideology?', 'posts_text']]
dfp.rename(columns={'political?': 'post_political', 'ideology?': 'post_ideology'}, inplace=True)
dfc.rename(columns={'political?': 'comment_political', 'ideology?': 'comment_ideology'}, inplace=True)

dfz = pd.merge(dfc,dfp, on='post_id')
dfz = dfz[pd.isnull(dfz['post_political']) == False]
dfz = dfz[pd.isnull(dfz['comment_political']) == False]
dfz = dfz[pd.isnull(dfz['post_id']) == False]
dfz = dfz[pd.isnull(dfz['comment_id']) == False]
dfz = dfz[dfz['category'] != 'Kids']
dfz.loc[dfz['category'] == 'Shows', 'category'] = 'Entertainment'

print(dfz, dfz.columns)

dfz.to_pickle('data/FB_combined_results_incl_text.pkl')
