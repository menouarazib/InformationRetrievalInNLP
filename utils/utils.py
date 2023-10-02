from typing import Dict, List

import nltk
import numpy as np

from matplotlib import pyplot as plt
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.ndimage import gaussian_filter1d


def go_tokenize(content: str, use_stop_words=False, use_is_alpha=False):
    """
    This function tokenizes the input content into words. It also provides options to filter out stop words and non-alphabetic words.

    Parameters:
    content (str): The text content to be tokenized.
    use_stop_words (bool, optional): If True, filters out stop words from the tokenized words. Default is False.
    use_is_alpha (bool, optional): If True, filters out non-alphabetic words from the tokenized words. Default is False.

    Returns:
    list: A list of tokenized words from the input content after applying the specified filters.

    """

    # Download the required NLTK resources
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')

    # Tokenize the text into words
    words = word_tokenize(content)

    # Filter out stopwords and punctuation
    if use_stop_words:
        stop_words = set(stopwords.words('english'))
    else:
        stop_words = []

    filtered_words = [word.lower() for word in words if
                      (use_is_alpha and word.isalpha() or not use_is_alpha) and word.lower() not in stop_words]

    return filtered_words


def extract_specific_pos_tags(words: list, pos_tag_name="JJ"):
    """
    This function performs part-of-speech (POS) tagging on a list of words and extracts words with a specific POS tag.

    Parameters:
    words (list): The list of words to be tagged.
    pos_tag_name (str, optional): The POS tag to filter the words. Default is "JJ" (adjective).

    Returns:
    list: A list of words from the input that have the specified POS tag.

    """

    # Perform part-of-speech tagging to identify words with the specified POS tag
    pos_tags = pos_tag(words)

    # Extract words with the specified POS tag
    tagged_words = [word for word, tag in pos_tags if tag == pos_tag_name]

    return tagged_words


def color_predicted_keywords(tokens: List, width: int, meta_model, config_dict: Dict) -> None:
    # Create sliding windows from tokens
    sliding_windows = [tokens[i:i + width] for i in range(len(tokens) - width + 1)]

    # Get the test data
    test_x_bg_i = len(sliding_windows) - len(meta_model.data_splitter.test_x)
    tokens_test = sliding_windows[test_x_bg_i:]

    # Predict the output
    y = meta_model.optimization_data.predicted_op
    sigma, m, h = config_dict["best_combination"]
    y = gaussian_filter1d(y, sigma=sigma, radius=m)
    y /= np.max(y)  # Normalize y

    color_map = plt.get_cmap('jet')

    # Function to convert a normalized value to an RGB color
    def get_color(value):
        color_ = color_map(value)[:3]  # Extract RGB values from the colormap
        return tuple(int(c * 255) for c in color_)

    # Print the text with highlighted words
    for current_index, phrase in enumerate(tokens_test):
        color_value = y[current_index]
        for word in [phrase[1]]:
            if color_value >= h:
                color = get_color(color_value)
                print(f"\033[38;2;{color[0]};{color[1]};{color[2]}m{word}\033[0m", end=' ')
            else:
                print(word, end=' ')
        if current_index != 0 and current_index % 15 == 0:
            print()
