import os
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def read_files_from_directory(directory):
    """Reads all .txt files from the given directory and returns their content."""
    file_contents = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                file_contents[filename] = file.read()
    return file_contents

def calculate_similarity(texts):
    """Calculates cosine similarity between text documents."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return cosine_similarity(tfidf_matrix)

def main():
    # Get all text files from the directory
    directory = os.path.dirname(os.path.abspath(__file__))
    file_contents = read_files_from_directory(directory)

    # If no text files found, exit
    if len(file_contents) < 2:
        print("Please add at least two .txt files to compare.")
        return

    # Compute similarity
    file_names = list(file_contents.keys())
    similarity_matrix = calculate_similarity(list(file_contents.values()))

    # Print similarity results
    print("\nPlagiarism Check Results:")
    for (i, j) in itertools.combinations(range(len(file_names)), 2):
        print(f"({file_names[i]} vs {file_names[j]}): {similarity_matrix[i][j]:.3f}")

if __name__ == "__main__":
    main()
