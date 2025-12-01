import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def create_movielens_dataset(dataset_name='ml-100k'):
    """
    Creates the MovieLens user features dataset.
    
    User features are the weighted average of the features of the movies they have watched,
    weighted by the score they have given to them.
    
    Args:
        dataset_name (str): The name of the dataset to load ('ml-100k' or 'ml-1m').
    """
    if dataset_name == 'ml-100k':
        # File paths
        data_dir = "MovieLens/ml-100k/"
        u_data_path = data_dir + "u.data"
        u_item_path = data_dir + "u.item"
        u_user_path = data_dir + "u.user"

        # Load data
        ratings = pd.read_csv(u_data_path, sep='\\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python')
        
        movie_genres = [
            "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", 
            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", 
            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
        ]
        movies = pd.read_csv(u_item_path, sep='|', header=None, names=['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL'] + movie_genres, encoding='latin-1')
        
        users = pd.read_csv(u_user_path, sep='|', header=None, names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])

        # Movie features are the genres
        movie_features = movies[movie_genres].values

    elif dataset_name == 'ml-1m':
        data_dir = "MovieLens/ml-1m/"
        ratings_path = data_dir + "ratings.dat"
        movies_path = data_dir + "movies.dat"
        users_path = data_dir + "users.dat"

        ratings = pd.read_csv(ratings_path, sep='::', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'], engine='python')
        
        movies_df = pd.read_csv(movies_path, sep='::', header=None, names=['movie_id', 'title', 'genres'], engine='python', encoding='latin-1')
        
        users = pd.read_csv(users_path, sep='::', header=None, names=['user_id', 'gender', 'age', 'occupation', 'zip_code'], engine='python')

        # Process movie genres
        genre_dummies = movies_df['genres'].str.get_dummies('|')
        movies = pd.concat([movies_df, genre_dummies], axis=1)
        movie_genres = genre_dummies.columns.tolist()
        movie_features = movies[movie_genres].values

    else:
        raise ValueError("Dataset not supported. Choose 'ml-100k' or 'ml-1m'.")

    movie_id_to_features = {movie_id: features for movie_id, features in zip(movies['movie_id'], movie_features)}

    # Create user features
    user_ids = users['user_id'].unique()
    user_features_list = []

    for user_id in sorted(user_ids):
        user_ratings = ratings[ratings['user_id'] == user_id]
        
        if user_ratings.empty:
            # Handle users with no ratings if any, though the dataset description says each user has at least 20.
            # The number of features is the number of genres.
            num_movie_features = len(movie_genres)
            user_features_list.append(np.zeros(num_movie_features))
            continue

        # Get movie features for rated movies
        rated_movie_ids = user_ratings['item_id']
        
        # Some movie IDs in ratings might not be in our movie features map
        valid_movie_features = np.array([movie_id_to_features[movie_id] for movie_id in rated_movie_ids if movie_id in movie_id_to_features])
        
        # Filter ratings to only include movies we have features for
        valid_ratings = user_ratings[user_ratings['item_id'].isin(movie_id_to_features.keys())]
        movie_ratings = valid_ratings['rating'].values

        if len(valid_movie_features) == 0:
            num_movie_features = len(movie_genres)
            user_features_list.append(np.zeros(num_movie_features))
            continue

        # Calculate weighted average
        weighted_features = np.dot(movie_ratings, valid_movie_features)
        sum_of_ratings = np.sum(movie_ratings)
        
        if sum_of_ratings > 0:
            user_feature_vector = weighted_features / sum_of_ratings
        else:
            # Handle case where sum of ratings is zero, if that's possible.
            user_feature_vector = np.zeros(valid_movie_features.shape[1])
            
        user_features_list.append(user_feature_vector)

    X = np.array(user_features_list)
    
    return X

def preprocess_data(X, method='pca', n_components=2, **kwargs):
    """
    Preprocesses the data using PCA or t-SNE for dimensionality reduction and normalization.

    Args:
        X (np.array): The input dataset.
        method (str): The method to use ('pca' or 'tsne').
        n_components (int): The number of components to reduce to.
        **kwargs: Additional arguments for the dimensionality reduction method.

    Returns:
        np.array: The preprocessed and normalized data.
    """
    if method == 'pca':
        reducer = PCA(n_components=n_components, **kwargs)
        X_reduced = reducer.fit_transform(X)
        cumulative_variance = np.sum(reducer.explained_variance_ratio_)
        print(f"Cumulative variance explained by {n_components} components: {cumulative_variance:.2%}")
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, **kwargs)
        X_reduced = reducer.fit_transform(X)
        print(f"KL divergence for t-SNE with {n_components} components: {reducer.kl_divergence_:.4f}")
    else:
        raise ValueError("Method not supported. Choose 'pca' or 'tsne'.")
    
    # Normalize the data to be between 0 and 1
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X_reduced)
    
    return X_normalized

def visualize_data(X, title=''):
    """
    Visualizes the 2D preprocessed data.

    Args:
        X (np.array): The 2D data to visualize.
        title (str): The title for the plot.
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], alpha=0.7)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # # --- For ml-100k ---
    # print("--- Processing ml-100k dataset ---")
    # X_100k = create_movielens_dataset(dataset_name='ml-100k')
    
    # print("Shape of the user features dataset (X_100k):", X_100k.shape)
    
    # # Preprocess and visualize using PCA for ml-100k
    # print("\nPreprocessing with PCA for ml-100k...")
    # X_100k_pca = preprocess_data(X_100k, method='pca', n_components=2)
    # print("Shape after PCA:", X_100k_pca.shape)
    # visualize_data(X_100k_pca, title='User Features - ml-100k (PCA)')

    # # Preprocess and visualize using t-SNE for ml-100k
    # print("\nPreprocessing with t-SNE for ml-100k...")
    # X_100k_tsne = preprocess_data(X_100k, method='tsne', n_components=2, perplexity=30, n_iter=300)
    # print("Shape after t-SNE:", X_100k_tsne.shape)
    # visualize_data(X_100k_tsne, title='User Features - ml-100k (t-SNE)')

    # --- For ml-1m ---
    print("\n--- Processing ml-1m dataset ---")
    X_1m = create_movielens_dataset(dataset_name='ml-1m')
    
    print("Shape of the user features dataset (X_1m):", X_1m.shape)

    # Preprocess and visualize using PCA for ml-1m
    print("\nPreprocessing with PCA for ml-1m...")
    X_1m_pca = preprocess_data(X_1m, method='pca', n_components=10)
    print("Shape after PCA:", X_1m_pca.shape)
    visualize_data(X_1m_pca, title='User Features - ml-1m (PCA)')
    np.save('movielens_1m_user_features_pca.npy', X_1m_pca)
    print("Saved the ml-1m PCA processed user features to 'movielens_1m_user_features_pca.npy'")

    # # Preprocess and visualize using t-SNE for ml-1m
    # print("\nPreprocessing with t-SNE for ml-1m...")
    # sample_size = 2000  # Using a smaller subset for t-SNE on ml-1m due to computational cost
    # if len(X_1m) > sample_size:
    #     indices = np.random.choice(len(X_1m), sample_size, replace=False)
    #     X_1m_sample = X_1m[indices]
    # else:
    #     X_1m_sample = X_1m
        
    # X_1m_tsne = preprocess_data(X_1m_sample, method='tsne', n_components=2, perplexity=50, n_iter=400)
    # print("Shape after t-SNE:", X_1m_tsne.shape)
    # visualize_data(X_1m_tsne, title='User Features - ml-1m (t-SNE, sampled)')
    
    # # Save the final processed dataset
    # np.save('movielens_1m_user_features_tsne.npy', X_1m_tsne)
    # print("Saved the ml-1m t-SNE processed user features to 'movielens_1m_user_features_tsne.npy'")
