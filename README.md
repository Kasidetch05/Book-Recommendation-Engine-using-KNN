# Book-Recommendation-Engine-using-KNN
---

Introduction

This project implements a Book Recommendation System using Collaborative Filtering based on the K-Nearest Neighbors (KNN) algorithm.
The model analyzes how users rate books to find patterns and suggest new titles based on your past reading history and the preferences of similar users.

---

The Core Idea: Item-Based Filtering

Imagine you love a specific book. This system works by finding other books that are consistently liked and highly rated by the same group of readers who liked your favorite book. It's like asking millions of readers: "If you liked X, what else did you enjoy?"

The Model: K-Nearest Neighbors (KNN)
- What it does: KNN finds the 5 closest neighboring books in the dataset that are most similar to the book you input.
- How it measures similarity: We use Cosine Distance. A distance close to 0 means the books are highly similar (they are rated similarly by the same users), while a distance close to 1 means they are very different.

---

Data Preparation (Making Sense of Ratings)

Working with millions of ratings requires clever organization and cleanup:

1. Strict Filtering: To ensure quality recommendations, we filtered the data aggressively:
   -  Users: Only users who have rated 200 or more books are kept.
   -  Books: Only books that have received 100 or more ratings are kept. This step removes noisy data and focuses on active users and popular books.

2. The User-Book Matrix: We transform the filtered data into a massive digital spreadsheet (a pivot table). Each row is a book title and each column is a user ID. The values inside the matrix are the user's ratings for that book.
   -   Efficiency: This matrix is converted into a Sparse Matrix to save memory, since most users have only rated a small fraction of all available books (lots of empty spaces/zeros!).

---

Key Function for Recommendations
The entire prediction process is wrapped in one easy-to-use function, defined in CELL 6:

<img width="521" height="267" alt="image" src="https://github.com/user-attachments/assets/00aacb65-a299-40e2-b4e8-d98686b7336b" />

---

Code Overview
The script is structured to manage the entire workflow, from data preparation to final testing:

- CELL 1-3: Loads data, imports libraries, and performs the critical data filtering and preprocessing steps.

- CELL 4: Creates the User-Book Matrix and the efficient Sparse Matrix.

- CELL 5: Builds and trains the KNN model using Cosine Distance.

- CELL 6: Defines the core get_recommends function.

- CELL 7-9: Tests the function with examples and includes an optional feature to visualize similarity distances using Matplotlib.

- CELL 10: Runs a comprehensive test to verify the output format is correct and robust.
---


