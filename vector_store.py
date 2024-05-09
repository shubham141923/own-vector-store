import numpy as np

class VectorStore:
    def __init__(self):
        self.vector_data = {} # A dictionary to store the vector
        self.vector_index = {} # A dictionary to store the indexing structure

    def add_vector(self, vector_id, vector):
        """
        Add a vector to the vector store

        Args:
            vector_id (str or int): The unique id for the vector
            vector: The vector to be stored
        """
        self.vector_data[vector_id] = vector
        self.update_index(vector_id, vector)
    
    def get_vector(self, vector_id):
        """
        Get a vector from the vector store

        Args:
            vector_id (str or int): The unique id for the vector

        Returns:
            numpy.darray: the vector data if found , or none if not found
        """
        return self.vector_data.get(vector_id)
    
    def update_index(self, vector_id, vector):
        """
        Update the indexing structure for the vector store

        Args:
            vector_id (str or int): The unique id for the vector
            vector (numpy.darray): the vector data to be stored
        """

        for existing_id, existing_vector in self.vector_data.items():
            similarity = np.dot(vector, existing_vector) / (np.linalg.norm(vector) * np.linalg.norm(existing_vector) )
            if existing_id not in self.vector_index:
                self.vector_index[existing_id] = {}
            self.vector_index[existing_id][vector_id] = similarity

    def find_similar_vectors(self, query_vector, num_results=5):
        """
        Find similar vectors to the query vector using brute-force search.

        Args:
            query_vector (numpy.ndarray): The query vector for similarity search.
            num_results (int): The number of similar vectors to return.

        Returns:
            list: A list of (vector_id, similarity_score) tuples for the most similar vectors.


        Reffrance:-
            np.dot(query_vector, vector)
            # Example 1: Two 1D arrays
                a = np.array([1, 2, 3])
                b = np.array([4, 5, 6])
                result = np.dot(a, b)
                print(result)  # Output: 32 (1*4 + 2*5 + 3*6)            
        """
        results = []
        for vector_id, vector in self.vector_data.items():
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            results.append((vector_id, similarity))

        # Sort by similarity in descending order
        results.sort(key=lambda x: x[1], reverse=True)

        # Return the top N results
        return results[:num_results]
    
