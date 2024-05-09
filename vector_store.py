import numpy as np

class VectorStore:
    """
    A class to store vectors and find similar vectors using brute-force search.
    """

    def __init__(self):
        """
        Initialize the vector store.
        """
        self.vector_data = {}  # A dictionary to store the vector
        self.vector_index = {}  # A dictionary to store the indexing structure

    def add_vector(self, vector_id: str | int, vector: np.ndarray) -> None:
        """
        Add a vector to the vector store.

        Args:
            vector_id (str or int): The unique id for the vector
            vector: The vector to be stored
        """
        self.vector_data[vector_id] = vector
        self.update_index(vector_id, vector)

    def get_vector(self, vector_id: str | int) -> np.ndarray | None:
        """
        Get a vector from the vector store.

        Args:
            vector_id (str or int): The unique id for the vector

        Returns:
            numpy.ndarray: the vector data if found, or None if not found
        """
        return self.vector_data.get(vector_id)

    def update_index(self, vector_id: str | int, vector: np.ndarray) -> None:
        """
        Update the indexing structure for the vector store.

        Args:
            vector_id (str or int): The unique id for the vector
            vector (numpy.ndarray): the vector data to be stored
        """
        for existing_id, existing_vector in self.vector_data.items():
            if existing_id == vector_id:
                continue
            similarity = np.dot(vector, existing_vector) / (
                np.linalg.norm(vector) * np.linalg.norm(existing_vector)
            )
            if existing_id not in self.vector_index:
                self.vector_index[existing_id] = {}
            self.vector_index[existing_id][vector_id] = similarity

    def find_similar_vectors(
        self, query_vector: np.ndarray, num_results: int = 5
    ) -> list:
        """
        Find similar vectors to the query vector using brute-force search.

        Args:
            query_vector (numpy.ndarray): The query vector for similarity search.
            num_results (int): The number of similar vectors to return.

        Returns:
            list: A list of (vector_id, similarity_score) tuples for the most similar vectors.
        """
        results = []
        for vector_id, vector in self.vector_data.items():
            similarity = np.dot(query_vector, vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vector)
            )
            if np.linalg.norm(query_vector) == 0:
                similarity = 0
            else:
                similarity /= np.linalg.norm(query_vector)
            results.append((vector_id, similarity))

        # Sort by similarity in descending order
        results.sort(key=lambda x: x[1], reverse=True)

        # Return the top N results
        return results[:num_results]
