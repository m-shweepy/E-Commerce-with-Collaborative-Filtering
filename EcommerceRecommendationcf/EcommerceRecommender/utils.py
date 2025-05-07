import pandas as pd
import numpy as np
import os
import logging
from models import UserItemRating
from app import db
from recommenders import load_data, initialize_models

# Global variables
ratings_df = None

def initialize_data():
    """Initialize data and models on application start."""
    global ratings_df
    
    # Load data into memory
    ratings_df = load_data()
    
    # Import data to database if it's empty
    if UserItemRating.query.count() == 0:
        import_data_to_db(ratings_df)
    
    # Initialize models (can be done in a background thread for large datasets)
    try:
        initialize_models()
    except Exception as e:
        logging.error(f"Error initializing models: {e}")

def import_data_to_db(df):
    """Import data from DataFrame to database."""
    try:
        logging.info("Importing data to database...")
        
        # Sample a smaller subset for faster processing
        sample_size = min(10000, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)
        
        # Add rows to database in batches
        batch_size = 500
        for i in range(0, len(sample_df), batch_size):
            batch = sample_df.iloc[i:i+batch_size]
            
            for _, row in batch.iterrows():
                rating = UserItemRating(
                    user_id=str(row['UserId']),
                    product_id=str(row['ProductId']),
                    rating=float(row['Rating']),
                    timestamp=int(row['Timestamp']) if not pd.isna(row['Timestamp']) else None
                )
                db.session.add(rating)
            
            db.session.commit()
            
        logging.info(f"Imported {sample_size} ratings to the database")
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error importing data to database: {e}")

def get_dataset_stats():
    """Get statistics about the dataset."""
    global ratings_df
    
    if ratings_df is None:
        ratings_df = load_data()
    
    # Calculate statistics
    user_count = len(ratings_df['UserId'].unique())
    product_count = len(ratings_df['ProductId'].unique())
    rating_count = len(ratings_df)
    avg_rating = ratings_df['Rating'].mean()
    rating_distribution = ratings_df['Rating'].value_counts().sort_index().to_dict()
    
    # Get number of ratings per user
    ratings_per_user = ratings_df.groupby('UserId').size()
    avg_ratings_per_user = ratings_per_user.mean()
    min_ratings_per_user = ratings_per_user.min()
    max_ratings_per_user = ratings_per_user.max()
    
    # Get number of ratings per product
    ratings_per_product = ratings_df.groupby('ProductId').size()
    avg_ratings_per_product = ratings_per_product.mean()
    min_ratings_per_product = ratings_per_product.min()
    max_ratings_per_product = ratings_per_product.max()
    
    # Create a dictionary with the statistics
    stats = {
        'user_count': int(user_count),
        'product_count': int(product_count),
        'rating_count': int(rating_count),
        'avg_rating': round(float(avg_rating), 2),
        'rating_distribution': {int(k): int(v) for k, v in rating_distribution.items()},
        'avg_ratings_per_user': round(float(avg_ratings_per_user), 2),
        'min_ratings_per_user': int(min_ratings_per_user),
        'max_ratings_per_user': int(max_ratings_per_user),
        'avg_ratings_per_product': round(float(avg_ratings_per_product), 2),
        'min_ratings_per_product': int(min_ratings_per_product),
        'max_ratings_per_product': int(max_ratings_per_product),
    }
    
    return stats

def get_available_user_ids(limit=10):
    """Get a sample of available user IDs to help the user."""
    global ratings_df
    
    if ratings_df is None:
        ratings_df = load_data()
    
    user_ratings_count = ratings_df.groupby('UserId').size()
    # Get users with at least 5 ratings for better recommendations
    active_users = user_ratings_count[user_ratings_count >= 5].index.tolist()
    
    # Return a sample of active users
    if len(active_users) > limit:
        return np.random.choice(active_users, limit, replace=False).tolist()
    return active_users

def get_available_product_ids(limit=10):
    """Get a sample of available product IDs to help the user."""
    global ratings_df
    
    if ratings_df is None:
        ratings_df = load_data()
    
    product_ratings_count = ratings_df.groupby('ProductId').size()
    # Get products with at least 5 ratings for better recommendations
    popular_products = product_ratings_count[product_ratings_count >= 5].index.tolist()
    
    # Return a sample of popular products
    if len(popular_products) > limit:
        return np.random.choice(popular_products, limit, replace=False).tolist()
    return popular_products
