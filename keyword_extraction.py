import gensim.downloader as api
import numpy as np
import pandas as pd

from eventdetector_ts import config_dict
from eventdetector_ts.metamodel.meta_model import MetaModel

from data import keywords_anarchism
from utils.utils import go_tokenize, color_predicted_keywords

file_name = "./data/anarchism.txt"  # file_name = "./data/autism.txt"

with open(file_name, 'r') as file:
    content = file.read()

tokens = go_tokenize(content=content, use_stop_words=False, use_is_alpha=False)

df = pd.DataFrame(index=pd.date_range(start='2023-03-23 00:00:00', periods=len(tokens),
                                      freq='1S'))
# Load Word2Vec pre-trained model (you can use other word embedding models too)
word2vec_model = api.load("word2vec-google-news-300")
# Represent each word as a float value vector using Word2Vec embeddings
df['WordVector'] = [word2vec_model[word] if word in word2vec_model else np.zeros(300) for word in tokens]
df = pd.DataFrame(df['WordVector'].to_list(), index=df.index, columns=[f'WordVector_{i + 1}' for i in range(300)])

keywords = keywords_anarchism  # keywords = keywords_autism

# Get the index positions of important words in the list of tokens
keywords_positions = [index for index, token in enumerate(tokens) if token in keywords]

events = []
for p in keywords_positions:
    events.append(df.index[p])

width = 3

meta_model = MetaModel(dataset=df, events=events, width=width, step=1, width_events=1,
                       remove_overlapping_events=True,
                       output_dir="test")

meta_model.fit()

color_predicted_keywords(tokens=tokens, width=width, meta_model=meta_model, config_dict=config_dict)  # noqa: E501
