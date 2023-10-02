#   Introduction
In this work, we propose an innovative approach by adapting the technique used for Event Detection in Multivariate Time Series Data (https://doi.org/10.31219/osf.io/uabjg) to address tasks in Natural Language Processing (NLP), specifically keyword extraction and the identification of adjectives for part-of-speech tagging (POS) in textual data.

# Requirements
You must have the `eventdetector-ts` package installed. You can find it [here](https://pypi.org/project/eventdetector-ts/). Additionally, please refer to the `requirements.txt` file for a list of other necessary libraries.

# Text As Time Series
To utilize the `eventdetector-ts` package, originally tailored for multivariate real-time series data and temporal events, we must first convert these texts into real-time series. Subsequently, we’ll establish a mapping between keywords and adjectives to represent them as real-time events. To convert a text into a real-time series, we initially tokenize the text into individual words. Following tokenization,
we leverage word embedding techniques like Word2Vec. Word2Vec transforms words into dense numerical representations within a high-dimensional space, typically comprising 300 features.
The following code snippet illustrates this process:
```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import pandas as pd


text = ...

# Tokenize the text into words
words = word_tokenize(text)

# Load Word2Vec pre-trained model
model_name = "word2vec-google-news-300"
word2vec_model = Word2Vec.load(model_name)

# Represent each word as a float value vector using Word2Vec embeddings
word_vectors = [word2vec_model[word] if word in word2vec_model else np.zeros(300) for word in words]

# Create a DataFrame with the word vectors and set the index to have a frequency of 1 second
start_date = '2023-03-23'
time_series = pd.DataFrame(word_vectors, index=pd.date_range(start=start_date, periods=len(words), freq='1S'), columns=[f'WordVector_{i + 1}' for i in range(300)])
```