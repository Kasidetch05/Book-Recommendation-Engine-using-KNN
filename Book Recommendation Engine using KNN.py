# ============================================
# CELL 1: Import Required Libraries
# ============================================
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# ============================================
# CELL 2: Load and Explore the Dataset
# ============================================
# The dataset should already be imported in the notebook
# Typically loaded as 'books' and 'ratings' dataframes

# Display basic information about the data
print("Books shape:", books.shape)
print("Ratings shape:", ratings.shape)
print("\nFirst few books:")
print(books.head())
print("\nFirst few ratings:")
print(ratings.head())

# ============================================
# CELL 3: Data Preprocessing
# ============================================
# Filter users with less than 200 ratings
user_counts = ratings['user_id'].value_counts()
valid_users = user_counts[user_counts >= 200].index
ratings_filtered = ratings[ratings['user_id'].isin(valid_users)]

# Filter books with less than 100 ratings
book_counts = ratings_filtered['isbn'].value_counts()
valid_books = book_counts[book_counts >= 100].index
ratings_filtered = ratings_filtered[ratings_filtered['isbn'].isin(valid_books)]

print(f"\nFiltered ratings shape: {ratings_filtered.shape}")
print(f"Number of users: {ratings_filtered['user_id'].nunique()}")
print(f"Number of books: {ratings_filtered['isbn'].nunique()}")

# Merge with books data to get titles
ratings_with_titles = ratings_filtered.merge(books, on='isbn')

# ============================================
# CELL 4: Create User-Item Matrix
# ============================================
# Create a pivot table (user-item matrix)
user_book_matrix = ratings_with_titles.pivot_table(
    index='title',
    columns='user_id',
    values='rating'
).fillna(0)

print(f"\nUser-Book Matrix shape: {user_book_matrix.shape}")

# Convert to sparse matrix for efficiency
user_book_sparse = csr_matrix(user_book_matrix.values)

# ============================================
# CELL 5: Build KNN Model
# ============================================
# Create and fit the NearestNeighbors model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6)
model_knn.fit(user_book_sparse)

print("\nKNN model trained successfully!")

# ============================================
# CELL 6: Create Recommendation Function
# ============================================
def get_recommends(book_title):
    """
    Get book recommendations based on KNN algorithm.
    
    Args:
        book_title (str): The title of the book to get recommendations for
        
    Returns:
        list: A list containing the book title and 5 similar books with distances
    """
    # Check if book exists in the dataset
    if book_title not in user_book_matrix.index:
        print(f"Book '{book_title}' not found in dataset")
        return [book_title, []]
    
    # Get the index of the book
    book_idx = user_book_matrix.index.get_loc(book_title)
    
    # Find nearest neighbors (6 because first one is the book itself)
    distances, indices = model_knn.kneighbors(
        user_book_matrix.iloc[book_idx].values.reshape(1, -1),
        n_neighbors=6
    )
    
    # Create list of recommendations (excluding the first one which is the book itself)
    recommendations = []
    for i in range(1, len(distances[0])):
        book_name = user_book_matrix.index[indices[0][i]]
        distance = distances[0][i]
        recommendations.append([book_name, distance])
    
    # Return in the required format: [book_title, [[book1, distance1], [book2, distance2], ...]]
    return [book_title, recommendations]

# ============================================
# CELL 7: Test the Recommendation Function
# ============================================
# Test with the example book
test_book = "The Queen of the Damned (Vampire Chronicles (Paperback))"
recommendations = get_recommends(test_book)

print("\nRecommendations for:", recommendations[0])
print("\nSimilar books:")
for book, distance in recommendations[1]:
    print(f"  - {book}: {distance:.4f}")

# ============================================
# CELL 8: Additional Testing (Optional)
# ============================================
# Test with other popular books
test_books = [
    "The Queen of the Damned (Vampire Chronicles (Paperback))",
    "Interview with the Vampire",
    "The Vampire Lestat (Vampire Chronicles, Book II)"
]

print("\n" + "="*80)
print("Testing multiple books:")
print("="*80)

for book in test_books:
    try:
        result = get_recommends(book)
        print(f"\nBook: {result[0]}")
        print("Recommendations:")
        for rec_book, dist in result[1]:
            print(f"  {rec_book}: {dist:.6f}")
    except:
        print(f"\nBook '{book}' not found in dataset")

# ============================================
# CELL 9: Visualize Book Similarity (Optional)
# ============================================
def visualize_recommendations(book_title):
    """Visualize the similarity distances of recommended books"""
    recommendations = get_recommends(book_title)
    
    if not recommendations[1]:
        print("No recommendations found")
        return
    
    books = [rec[0][:30] + "..." if len(rec[0]) > 30 else rec[0] for rec in recommendations[1]]
    distances = [rec[1] for rec in recommendations[1]]
    
    plt.figure(figsize=(10, 6))
    plt.barh(books, distances, color='skyblue')
    plt.xlabel('Distance (Lower = More Similar)')
    plt.title(f'Book Recommendations for:\n{book_title[:50]}...')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# Visualize recommendations for the test book
if 'test_book' in locals():
    visualize_recommendations(test_book)

# ============================================
# CELL 10: Final Testing Cell
# ============================================
# This cell is for running the final test
# The test should verify that get_recommends returns the correct format

def test_book_recommendation():
    """Test the recommendation function"""
    test_pass = True
    
    # Test 1: Check return format
    result = get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")
    
    if not isinstance(result, list):
        print("❌ Result should be a list")
        test_pass = False
    
    if len(result) != 2:
        print("❌ Result should have 2 elements")
        test_pass = False
    
    if not isinstance(result[0], str):
        print("❌ First element should be a string (book title)")
        test_pass = False
    
    if not isinstance(result[1], list):
        print("❌ Second element should be a list of recommendations")
        test_pass = False
    
    if len(result[1]) != 5:
        print("❌ Should return exactly 5 recommendations")
        test_pass = False
    
    # Test 2: Check recommendation format
    for rec in result[1]:
        if not isinstance(rec, list) or len(rec) != 2:
            print("❌ Each recommendation should be [book_title, distance]")
            test_pass = False
            break
        if not isinstance(rec[0], str):
            print("❌ Book title should be a string")
            test_pass = False
            break
        if not isinstance(rec[1], (int, float)):
            print("❌ Distance should be a number")
            test_pass = False
            break
    
    if test_pass:
        print("✅ All tests passed!")
        print("\nExpected output format:")
        print(result)
    
    return test_pass

# Run the test
test_book_recommendation()