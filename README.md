#   Introduction
In this work, we propose an innovative approach by adapting the technique used for Event Detection in Multivariate Time Series Data [1] to address tasks in Natural Language Processing (NLP), specifically keyword extraction and the identification of adjectives for part-of-speech tagging (POS) in textual data.

# Requirements
You must have the `eventdetector-ts` package installed. You can find it [here](https://pypi.org/project/eventdetector-ts/). Additionally, please refer to the `requirements.txt` file for a list of other necessary libraries.

# Datasets
We use two extensive English texts sourced from the Wikipedia dump [2]. One of these texts centers around Autism [3], while the other delves into Anarchism [3]. To effectively utilize `eventdetector-ts`  package, originally tailored for real-time series data and temporal events, we must first convert these texts into real-time series.
And, represent ground true keywords and tags as real-time events.

# Text As Time Series using Word2Vec
To convert a text into a real-time series, we initially tokenize the text into individual words. Following tokenization, we leverage word embedding techniques like Word2Vec [4]. Word2Vec transforms words into dense numerical representations within a high-dimensional space, typically comprising 300 features.
The following code snippet illustrates this process:
```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np


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

Given those keywords and tags may occur at various positions within the text (which has been transformed into a time series represented as a dataframe, `time_seres`), we establish a mapping
between these positions and corresponding timestamps based on the index of `time_series`. This mapping enables us to convert occurrences of keywords and adjectives within the text into temporal values. Subsequently, by specifying a value for the event’s width (`width_events`), we represent them as temporal events.

# Keyword Extraction & Part of Speech Tagging
For this evaluation, we have selected a set of 20 reference keywords associated with Autism and Anarchism texts. Additionally, the list of ground true tags (here we choose only adjectives) has been obtained using the Natural Language Toolkit library [5] on these texts.
```python
# Get the index positions of important words in the list of tokens
keywords_positions = [index for index, token in enumerate(tokens) if token in keywords]

events = []
for p in keywords_positions:
    events.append(time_series.index[p])
```

As a result, for each of these texts, we create two cases: one for keyword extraction (Autism (Keys) and Anarchism (Keys)) and another for finding adjectives (Autism (POS) and Anarchism (POS)), as outlined in “TABLE. I”.

To execute keyword extraction just run `keyword_extraction.py` and to execute part of speech tagging just run `part_of_speech_tagging.py`.

The result evaluations of our method across the specified use cases, assessing its performance using three key metrics: F1-Score, Precision, and Recall. The results of these evaluations are presented in “TABLE. I”. Each block represents a use case, and each line within the block is characterized by a distinct configuration of the meta model, involving variations in sliding window width and stacked models.

![TABLE. I](https://raw.githubusercontent.com/menouarazib/InformationRetrievalInNLP/master/images/Results.png)

# References
[1] M. Azib, B. Renard, P. Garnier, V. Génot, and N. André, “Universal Event Detection in Time Series,” 2023, [Online]. Available: https://doi.org/10.31219/osf.io/uabjg.

[2] Wikimedia Foundation, “Wikimedia Downloads,” [Online]. Available: https://dumps.wikimedia.org.

[3] Hugging Face. ‘Wikipedia Dataset.’ [Online]. Available: https://huggingface.co/datasets/wikipedia/viewer/20220301.en/train.

[4] T. Mikolov, K. Chen, G. Corrado, and J. Dean, “Efficient Estimation of Word Representations in Vector Space,” in Proceedings of Workshop at ICLR, 2013, arXiv:1301.3781v1.

[5] S. Bird, E. Klein, and E. Loper, “Natural Language Processing with Python: Analyzing Text with the Natural Language Toolkit”. O’Reilly Media, Inc., 2009.