import numpy as np
from vector_store import VectorStore
from collections import defaultdict
import os

def read_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            sentences = file.readlines()
        return sentences
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

# Initialize VectorStore
try:
    vector_store = VectorStore()
except Exception as e:
    print(f"Error initializing VectorStore: {e}")
    vector_store = None

# Read text from file
file_path = "C:\\Users\\Admin\\Desktop\\vector_test.txt"
sentences = read_file(file_path)

# Tokenization and Vocabulary Creation
vocabulary = set()
for sentence in sentences:
    tokens = sentence.lower().split()
    vocabulary.update(tokens)

# Assign unique indices to words in the vocabulary
word_to_index = {word: i for i, word in enumerate(vocabulary)}

# Vectorization
sentence_vectors = defaultdict(np.zeros)
for sentence in sentences:
    tokens = sentence.lower().split()
    for token in tokens:
        if token in word_to_index:
            sentence_vectors[sentence][word_to_index[token]] += 1

# Storing in VectorStore
if vector_store:
    for sentence, vector in sentence_vectors.items():
        vector_store.add_vector(sentence, vector)

# Searching for Similarity
query_sentence = "who premiered in 2
