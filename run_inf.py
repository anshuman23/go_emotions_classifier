import numpy as np
import pandas as pd
import csv
import os
from collections import defaultdict
import pickle
import sys
from tqdm import tqdm
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline
import torch

os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
#print(device)


tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")
goemotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa', device=0) #device=0


df = pd.read_pickle('data/FB_combined_results_incl_text.pkl')

if os.path.isfile('temp.csv'):
    temp_df = pd.read_csv('temp.csv')
    start_val = int(temp_df.columns[0].split('_')[0])
    post_emotion = temp_df[temp_df.columns[0]].to_list()
    comment_emotion = temp_df[temp_df.columns[1]].to_list()
else:
    post_emotion = []
    comment_emotion = []
    start_val = 0

num_batches = 1000 #10000
chunk_size = int(len(df)/num_batches)

for start in tqdm(range(0, df.shape[0], chunk_size)):
    if start < start_val:
        continue

    df_sub = df.iloc[start:start + chunk_size]

    comment_texts, post_texts = df_sub['comments_text'].to_list(), df_sub['posts_text'].to_list()

    
    try:
        raw_post_preds = goemotion(post_texts, batch_size=chunk_size, truncation=True, padding=True)
        for post_pred in raw_post_preds:
            post_emotion.append(post_pred['label'])
    except:
        post_emotion += ['unknown']*len(df_sub)

    try:
        raw_comment_preds = goemotion(comment_texts, batch_size=chunk_size, truncation=True, padding=True)
        for comment_pred in raw_comment_preds:
            comment_emotion.append(comment_pred['label'])
    except:
        comment_emotion += ['unknown']*len(df_sub)

    pd.DataFrame({str(start)+'_post_preds': post_emotion, str(start)+'_comment_preds': comment_emotion}).to_csv('temp.csv', index=False)


print(len(post_emotion), len(comment_emotion))

df['post_emotion'] = post_emotion
df['comment_emotion'] = comment_emotion

print(df)

#Full data storage
df.to_pickle('output/FB_combined_results_incl_text_emotion.pkl')

#Removing the column containing text for csv output used for analysis
df.drop('comment_text', axis=1, inplace=True)
df.drop('post_text', axis=1, inplace=True)
df.to_csv('output/FB_combined_results_incl_emotion.csv', index=False)

