import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
import tensorflow as tf
import logging
import keras
import os

# Global variables to store data and models
ratings_df = None
user_based_similarity = None
item_based_similarity = None
matrix_fact_model = None
user_factors = None
item_factors = None
user_mapping = None
item_mapping = None
reverse_user_mapping = None
reverse_item_mapping = None
deep_learning_model = None
def load_data():
    """Load and preprocess the ratings data."""
    global ratings_df
    
    # Get dataset path
    dataset_path = os.path.join('attached_assets', 'preprocessed_dataset1.csv')
    
    # Load the dataset
    try:
        ratings_df = pd.read_csv(dataset_path)
        logging.info(f"Dataset loaded with shape: {ratings_df.shape}")
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        # Use a small sample for testing if file not found
        data = {
            'UserId': ['U1', 'U1', 'U2', 'U2', 'U3'],
            'ProductId': ['P1', 'P2', 'P2', 'P3', 'P1'],
            'Rating': [5, 4, 5, 3, 4],
            'Timestamp': [1000, 1001, 1002, 1003, 1004]
        }
        ratings_df = pd.DataFrame(data)
        logging.warning("Using fallback sample data since dataset couldn't be loaded")
    
    # Convert user and product IDs to strings to ensure consistency
    ratings_df['UserId'] = ratings_df['UserId'].astype(str)
    ratings_df['ProductId'] = ratings_df['ProductId'].astype(str)
    
    return ratings_df

def create_user_item_matrix():
    """Create a user-item matrix from the ratings data."""
    if ratings_df is None:
        load_data()
    
    # Create the user-item matrix
    user_item_matrix = ratings_df.pivot_table(
        index='UserId', 
        columns='ProductId', 
        values='Rating',
        fill_value=0
    )
    
    return user_item_matrix

def train_user_based_model():
    """Train the user-based collaborative filtering model."""
    global user_based_similarity
    
    # Create user-item matrix
    user_item_matrix = create_user_item_matrix()
    
    # Calculate user-user similarity using cosine similarity
    user_based_similarity = cosine_similarity(user_item_matrix)
    
    # Create a DataFrame with the similarity matrix
    user_based_similarity = pd.DataFrame(
        user_based_similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index
    )
    
    logging.info("User-based collaborative filtering model trained")

def train_item_based_model():
    """Train the item-based collaborative filtering model."""
    global item_based_similarity
    
    # Create user-item matrix
    user_item_matrix = create_user_item_matrix()
    
    # Calculate item-item similarity using cosine similarity
    item_based_similarity = cosine_similarity(user_item_matrix.T)
    
    # Create a DataFrame with the similarity matrix
    item_based_similarity = pd.DataFrame(
        item_based_similarity,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )
    
    logging.info("Item-based collaborative filtering model trained")

def create_mapping_dictionaries():
    """Create mappings between original IDs and integer indices for deep learning model."""
    global user_mapping, item_mapping, reverse_user_mapping, reverse_item_mapping
    
    if ratings_df is None:
        load_data()
    
    # Create user mapping (user_id -> index)
    unique_users = ratings_df['UserId'].unique()
    user_mapping = {user: i for i, user in enumerate(unique_users)}
    reverse_user_mapping = {i: user for user, i in user_mapping.items()}
    
    # Create item mapping (product_id -> index)
    unique_products = ratings_df['ProductId'].unique()
    item_mapping = {product: i for i, product in enumerate(unique_products)}
    reverse_item_mapping = {i: product for product, i in item_mapping.items()}
    
    logging.info(f"Created mappings for {len(user_mapping)} users and {len(item_mapping)} items")

def train_deep_learning_model():
    """Train a neural network model for collaborative filtering."""
    global deep_learning_model, user_mapping, item_mapping
    
    if ratings_df is None:
        load_data()
    
    # Create mappings if not already created
    if user_mapping is None or item_mapping is None:
        create_mapping_dictionaries()
    
    # Map user and item IDs to indices
    user_indices = ratings_df['UserId'].map(user_mapping)
    item_indices = ratings_df['ProductId'].map(item_mapping)
    
    # Normalize ratings to [0, 1] range
    ratings = ratings_df['Rating'] / 5.0
    
    # Define model parameters
    n_users = len(user_mapping)
    n_items = len(item_mapping)
    n_factors = 50
    
    # Define model architecture
    user_input = keras.Input(shape=(1,), name='user_input')
    item_input = keras.Input(shape=(1,), name='item_input')
    
    user_embedding = keras.layers.Embedding(n_users, n_factors, name='user_embedding')(user_input)
    item_embedding = keras.layers.Embedding(n_items, n_factors, name='item_embedding')(item_input)
    
    user_flatten = keras.layers.Flatten()(user_embedding)
    item_flatten = keras.layers.Flatten()(item_embedding)
    
    concat = keras.layers.Concatenate()([user_flatten, item_flatten])
    dense1 = keras.layers.Dense(128, activation='relu')(concat)
    dense2 = keras.layers.Dense(64, activation='relu')(dense1)
    output = keras.layers.Dense(1, activation='sigmoid')(dense2)
    
    deep_learning_model = keras.Model(inputs=[user_input, item_input], outputs=output)
    deep_learning_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error'
    )
    
    # Train the model with a small number of epochs for demonstration
    deep_learning_model.fit(
        [user_indices, item_indices],
        ratings,
        batch_size=64,
        epochs=5,
        verbose=1,
        validation_split=0.1
    )
    
    logging.info("Deep learning collaborative filtering model trained")

def get_user_based_recommendations(user_id, top_n=5):
    """Get recommendations for a user using user-based collaborative filtering."""
    global user_based_similarity
    
    if user_based_similarity is None:
        train_user_based_model()
    
    
    if user_id not in user_based_similarity.index:
        raise ValueError(f"User ID '{user_id}' not found in the dataset.")
    
    user_item_matrix = create_user_item_matrix()
    
    # Get similarity scores for the target user
    user_similarities = user_based_similarity.loc[user_id]
    
    # Get items that the target user has not rated
    user_rated_items = user_item_matrix.loc[user_id]
    unrated_items = user_rated_items[user_rated_items == 0].index
    
    # Calculate predicted ratings for unrated items
    item_scores = {}
    for item in unrated_items:
        # Find users who have rated this item
        item_raters = user_item_matrix[item]
        item_raters = item_raters[item_raters > 0].index
        
        if len(item_raters) == 0:
            continue
        
        # Calculate the weighted average of ratings from similar users
        similarities = user_similarities[item_raters]
        ratings = user_item_matrix.loc[item_raters, item]
        
        # Skip if all similarities are zero to avoid division by zero
        if sum(similarities) == 0:
            continue
        
        # Calculate weighted average
        predicted_rating = sum(similarities * ratings) / sum(similarities)
        item_scores[item] = predicted_rating
    
    # Sort items by predicted rating
    recommended_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Format recommendations
    recommendations = [
        {
            'product_id': item,
            'predicted_rating': round(score, 2)
        }
        for item, score in recommended_items
    ]
    
    return recommendations

def get_item_based_recommendations(product_id, top_n=5):
    """Get recommendations for a product using item-based collaborative filtering."""
    global item_based_similarity
    
    if item_based_similarity is None:
        train_item_based_model()
    
    # Check if product exists
    if product_id not in item_based_similarity.index:
        raise ValueError(f"Product ID '{product_id}' not found in the dataset.")
    
    # Get similarity scores for the target product
    item_similarities = item_based_similarity.loc[product_id]
    
    # Get most similar items (excluding the input item itself)
    similar_items = item_similarities.drop(product_id).sort_values(ascending=False).head(top_n)
    
    # Format recommendations
    recommendations = [
        {
            'product_id': item,
            'similarity_score': round(score, 2)
        }
        for item, score in similar_items.items()
    ]
    
    return recommendations

def get_deep_learning_recommendations(user_id, top_n=5):
    """Get recommendations for a user using the deep learning model."""
    global deep_learning_model, user_mapping, item_mapping, reverse_item_mapping
    
    if deep_learning_model is None:
        train_deep_learning_model()
    
    # Check if user exists
    if user_id not in user_mapping:
        raise ValueError(f"User ID '{user_id}' not found in the dataset.")
    
    # Get user index
    user_idx = user_mapping[user_id]
    
    # Create a list of all items
    all_items = list(item_mapping.keys())
    
    # Get user-item matrix to find unrated items
    user_item_matrix = create_user_item_matrix()
    
    # Get items that the user has not rated (if user is in the matrix)
    if user_id in user_item_matrix.index:
        user_rated_items = user_item_matrix.loc[user_id]
        rated_items = user_rated_items[user_rated_items > 0].index
        unrated_items = [item for item in all_items if item not in rated_items]
    else:
        # If user has no ratings in the matrix, recommend based on all items
        unrated_items = all_items
    
    # Make predictions for unrated items
    predictions = []
    
    # Use batching for efficiency
    batch_size = 100
    for i in range(0, len(unrated_items), batch_size):
        batch_items = unrated_items[i:i+batch_size]
        item_indices = [item_mapping[item] for item in batch_items]
        
        # Create input arrays
        user_array = np.array([user_idx] * len(batch_items))
        item_array = np.array(item_indices)
        
        # Make predictions
        batch_predictions = deep_learning_model.predict([user_array, item_array], verbose=0)
        
        # Add to predictions list
        for j, item in enumerate(batch_items):
            predictions.append((item, float(batch_predictions[j][0])))
    
    # Sort by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Get top-n recommendations
    top_recommendations = predictions[:top_n]
    
    # Format recommendations
    recommendations = [
        {
            'product_id': item,
            'predicted_rating': round(score * 5, 2)  # Scale back to 1-5 rating
        }
        for item, score in top_recommendations
    ]
    
    return recommendations

def initialize_models():
    """Initialize all recommendation models."""
    # Load data
    load_data()
    
    # Train user-based model
    train_user_based_model()
    
    # Train item-based model
    train_item_based_model()
    
    # Create mappings for deep learning model
    create_mapping_dictionaries()
    
    # Train deep learning model
    train_deep_learning_model()
    
    logging.info("All recommendation models initialized")
