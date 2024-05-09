from vector_store import VectorStore
import numpy as np

# Create a VectorStore instance
vector_store = VectorStore()

# Read text from file
file_path = "C:\\Users\\Admin\\Desktop\\vector_test.txt"
with open(file_path, "r", encoding="utf-8") as file:
    sentences = file.readlines()

# Tokenization and Vocabulary Creation
vocabulary = set()
for sentence in sentences:
    tokens = sentence.lower().split() # in this format they convert the sentance into the token like sentance convert into the array.
    vocabulary.update(tokens) # set all the tokesn data into a single disctonary object.

# Assign unique indices to words in the vocabulary
word_to_index = {word: i for i, word in enumerate(vocabulary)} # Assign all the unique words with a index value start from the 0

# Vectorization
sentence_vectors = {}
for sentence in sentences:
    tokens = sentence.lower().split() # in this format they convert the sentance into the token like sentance convert into the array.
    vector = np.zeros(len(vocabulary))
    for token in tokens:
        vector[word_to_index[token]] += 1
    sentence_vectors[sentence] = vector

# Storing in VectorStore
for sentence, vector in sentence_vectors.items():
    vector_store.add_vector(sentence, vector)

# Searching for Similarity
query_sentence = "who premiered in 2016."
query_vector = np.zeros(len(vocabulary))
query_tokens = query_sentence.lower().split()

for token in query_tokens:
    if token in word_to_index:
        query_vector[word_to_index[token]] += 1
similar_sentences = vector_store.find_similar_vectors(query_vector, num_results=3)

# Print similar sentences
print("Query Sentence:", query_sentence)
print("Similar Sentences:")
for sentence, similarity in similar_sentences:
    print(f"{sentence}: Similarity = {similarity:.4f}")