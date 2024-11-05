import numpy as np
import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab') 

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    Tokenizes a sentence by splitting it into individual components (words, punctuation, numbers).
    Each component in the sentence becomes an element in the output list.
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    Reduces a word to its root form using stemming.
    Example:
    words = ["organize", "organizes", "organizing"]
    stemmed_words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    """
    Creates a binary array representing the presence of known words in a tokenized sentence.
    
    Each position in the array corresponds to a word from the known vocabulary (words). 
    If a word in 'words' exists in the tokenized sentence, the position is marked with 1; otherwise, it's 0.
    
    Example:
    tokenized_sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    output = [0, 1, 0, 1, 0, 0, 0]
    """
    # Process each word in the sentence to its root form
    sentence_roots = [stem(word) for word in tokenized_sentence]
    # Initialize a binary array for the known vocabulary
    bag = np.zeros(len(words), dtype=np.float32)
    # Mark 1 if word from vocabulary exists in the sentence
    for idx, w in enumerate(words):
        if w in sentence_roots: 
            bag[idx] = 1

    return bag
