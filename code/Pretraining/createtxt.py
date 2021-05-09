import pandas as pd
import glob
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
import numpy as np
from random import shuffle

texts = []
b = False
wcs = np.array([0])

for x in glob.glob('data/*_submission.csv'):
    if not ('future' in x or 'feature' in x):
        df = pd.read_csv(x, index_col=None, header=0)
        i = 0
        for _, row in df.iterrows():
            text = f"{row['title']}. {row['text']}"

            # text = text.split('\n')
            # text = '.'.join(text)
            # text = text.split('.')
            text = text.replace('\n', ' ')

            # if '* CBIS (Longer term hold)' in text:
            #     print(repr(text))
            #     b = True
            text = text.replace('\r', ' ')
            text = sent_tokenize(text)
            # if b:
            #     # print(text)
            #     1/0
            new_text = list(filter(lambda t: len(t.split()) <= 128 and len(t.split()) > 3, text))
            # new_text = [' '.join(word_tokenize(t)) for t in sentences]

            # lengths = []
            # for t in text:
            #     lengths.append(len(t.split()))

            # wcs = np.concatenate((wcs, np.array(lengths)))
            if len(new_text) > 0:
                texts.append('\n'.join(new_text))
            # i += 1
            # if i > 6:
            #     break

# for x in glob.glob('data/*_comments.csv'):
#     if not ('future' in x or 'feature' in x):
#         df = pd.read_csv(x, index_col=None, header=0)
#         i = 0
#         for _, row in df.iterrows():
#             text = row['text']
#             text = text.replace('\n', ' ')
#             text = text.replace('\r', ' ')
#             text = sent_tokenize(text)
#             text = filter(len, text)
#             texts.append('\n'.join(text))
#             # i += 1
#             # if i > 6:
#             #     break


print(wcs.mean())
print(np.percentile(wcs, 90))
print(np.percentile(wcs, 99))
print(wcs.max())

shuffle(texts)

s1 = int(.8 *  len(texts))
s2 = int(.9* len(texts))

with open('./pretraining-data/train.txt', 'w', encoding="utf-8") as f:
    f.write('\n\n'.join(texts[:s1]))

with open('./pretraining-data/dev.txt', 'w', encoding="utf-8") as f:
    f.write('\n\n'.join(texts[s1:s2]))

with open('./pretraining-data/test.txt', 'w', encoding="utf-8") as f:
    f.write('\n\n'.join(texts[s2:]))