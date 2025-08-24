# Group Recommendation System using Machine Learning (KNN Enhanced)
# Author: NIT Warangal B.Tech Student

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ----------------------
# Sample User Preferences
# ----------------------
user_preferences = pd.DataFrame({
    'Product1': [5, 3, 0, 1],
    'Product2': [4, 0, 0, 1],
    'Product3': [1, 1, 0, 5],
    'Product4': [0, 0, 5, 4],
    'Product5': [0, 1, 5, 4]
}, index=['User1', 'User2', 'User3', 'User4'])

# ----------------------
# User-based Recommendation using KNN
# ----------------------
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(user_preferences.values)

def knn_group_recommendation(group_users, user_preferences, top_n=3):
    # Aggregate preferences
    group_data = user_preferences.loc[group_users].mean(axis=0)
    # Recommend top N items
    recommended_products = group_data.sort_values(ascending=False).head(top_n)
    return recommended_products

# Example KNN-based recommendation
group_users = ['User1', 'User2']
recommendations_knn = knn_group_recommendation(group_users, user_preferences, top_n=3)
print("KNN Group Recommendations for Users {}:".format(group_users))
print(recommendations_knn)

# ----------------------
# Sample Flipkart Products with Descriptions for NLP
# ----------------------
products = pd.DataFrame({
    'Product': ['Product1', 'Product2', 'Product3', 'Product4', 'Product5'],
    'Description': [
        'Smartphone with high-resolution camera and fast processor',
        'Wireless earbuds with noise cancellation and long battery',
        'Laptop with SSD storage and high RAM for gaming',
        'Digital watch with fitness tracking and water resistance',
        'Bluetooth speaker with deep bass and compact design'
    ]
})

# Convert product descriptions to TF-IDF vectors
vectorizer = TfidfVectorizer()
product_vectors = vectorizer.fit_transform(products['Description'])

# Compute similarity using cosine similarity
similarity_matrix = cosine_similarity(product_vectors)
similarity_df = pd.DataFrame(similarity_matrix, index=products['Product'], columns=products['Product'])
print("\nProduct Similarity Matrix based on NLP descriptions:")
print(similarity_df)

# ----------------------
# NLP-based Recommendation Function
# ----------------------
def recommend_similar_products(product_name, similarity_df, top_n=3):
    similar_products = similarity_df[product_name].sort_values(ascending=False).drop(product_name).head(top_n)
    return similar_products

# Example NLP-based recommendation
recommended_products_nlp = recommend_similar_products('Product1', similarity_df, top_n=3)
print("\nTop Flipkart product recommendations similar to Product1:")
print(recommended_products_nlp)

# ----------------------
# Combined Recommendation (KNN + NLP)
# ----------------------
def combined_group_product_recommendation(group_users, user_preferences, similarity_df, top_n=5):
    user_based = knn_group_recommendation(group_users, user_preferences, top_n=top_n*2)
    nlp_based_products = []
    for product in user_based.index:
        nlp_products = recommend_similar_products(product, similarity_df, top_n=1)
        nlp_based_products.extend(nlp_products.index.tolist())
    final_recommendations = list(user_based.index) + nlp_based_products
    final_recommendations = final_recommendations[:top_n]
    return final_recommendations

# Example combined recommendation
final_group_product_recommendations = combined_group_product_recommendation(group_users, user_preferences, similarity_df, top_n=5)
print("\nFinal Combined Flipkart Product Recommendations for Users {}:".format(group_users))
print(final_group_product_recommendations)
