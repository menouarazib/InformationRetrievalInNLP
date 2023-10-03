The Continuous Bag of Words (CBOW) model of Word2Vec is trained using the context of a word to predict the word itself. Here's a step-by-step explanation of how it works:

1. **Preprocessing**: The input text is preprocessed to convert it into a suitable format. This usually involves tokenization (splitting the text into individual words), and may also involve other steps like lowercasing, removal of punctuation, etc.

2. **Building the Vocabulary**: A vocabulary of unique words is built from the preprocessed text. Each word in the vocabulary is assigned a unique integer index.

3. **Creating Training Data**: For each word in the text, we create a context window of a certain size around the word (for example, 2 words on either side). The target word and its context words form a training example. For instance, in the sentence "The cat sat on the mat", if "sat" is our target word and we're using a context window of size 2, our input would be ["The", "cat", "on", "the"] and our target would be "sat".

4. **One-hot Encoding**: Both the input context words and the target word are one-hot encoded based on the vocabulary. In one-hot encoding, each word is represented as a vector of length equal to the vocabulary size, with a 1 at the index corresponding to the word, and 0s elsewhere.

5. **Defining the Model Architecture**: The CBOW model architecture consists of three layers: an input layer, a hidden layer, and an output layer. The input layer takes in the one-hot encoded context words and passes them through a hidden layer (which is a fully connected layer with weights initialized randomly) to produce an 'embedding' for each context word. These embeddings are averaged to produce a single embedding.

6. **Training**: The averaged embedding is passed through the output layer (another fully connected layer), which transforms it back to the dimension of the vocabulary size. This output is then softmaxed to produce probabilities for each word in the vocabulary. The model is trained to maximize the probability for the actual target word.

7. **Updating Weights**: Weights are updated using backpropagation and an optimization algorithm such as Stochastic Gradient Descent (SGD).

8. **Extracting Word Vectors**: After training, the weights of the hidden layer form our word vectors or 'embeddings'. Each row corresponds to a word in our vocabulary.

This process allows us to learn dense vectors for each word that capture semantic meanings based on their context.