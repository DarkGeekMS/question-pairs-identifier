import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt

table = pd.read_csv('../dataset/train.csv')



print('Number of question pairs {}'.format(len(table)))
print('Duplicate pairs percentage: {}%'.format(round(table['is_duplicate'].mean()*100, 2)))
print('Non-Duplicate pairs percentage: {}%'.format(round((1 - table['is_duplicate']).mean()*100, 2)))
qids = pd.Series(table['qid1'].tolist() + table['qid2'].tolist())
print('Number of unique questions: {}'.format(len(np.unique(qids))))
print('Number of questions that appear multiple times: {}'.format(np.sum(qids.value_counts() > 1)))

plt.figure(figsize=(12, 5))
plt.hist(qids.value_counts(), bins=50)
plt.yscale('log', nonposy='clip')
plt.title('Log-Histogram of question appearance counts')
plt.xlabel('Number of occurences of question')
plt.ylabel('Number of questions')
plt.savefig('./plots/analysis1.png')

from wordcloud import WordCloud
train_qs = pd.Series(table['question1'].tolist() + table['question2'].tolist()).astype(str)
cloud = WordCloud(width=1440, height=1080).generate(" ".join(train_qs.astype(str)))
plt.figure(figsize=(20, 15))
plt.imshow(cloud)
plt.axis('off')
plt.savefig('./plots/cloud.png')
