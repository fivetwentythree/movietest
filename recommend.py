import argparse
import pandas as pd
from surprise import Dataset, Reader, SVDpp


def load_data(ratings_file: str):
    ratings_df = pd.read_csv(ratings_file)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
    return data, ratings_df


def train_model(data):
    trainset = data.build_full_trainset()
    algo = SVDpp()
    algo.fit(trainset)
    return algo


def get_unrated_movies(ratings_df, movies_df, user_id):
    rated_movies = set(ratings_df[ratings_df['userId'] == user_id]['movieId'])
    unrated = movies_df[~movies_df['movieId'].isin(rated_movies)]
    return unrated['movieId'].tolist()


def recommend(algo, user_id, movies_df, unrated_movie_ids, n=5):
    predictions = [
        (movie_id, algo.predict(str(user_id), str(movie_id)).est)
        for movie_id in unrated_movie_ids
    ]
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n = predictions[:n]
    for movie_id, rating in top_n:
        title = movies_df.loc[movies_df['movieId'] == movie_id, 'title'].values[0]
        print(f"{title} (predicted rating: {rating:.2f})")


def main():
    parser = argparse.ArgumentParser(description="Movie recommendation using SVD++")
    parser.add_argument('--user', type=int, required=True, help='User ID for recommendations')
    parser.add_argument('--ratings', default='ratings.csv', help='Path to ratings CSV')
    parser.add_argument('--movies', default='movies.csv', help='Path to movies CSV')
    parser.add_argument('-n', type=int, default=5, help='Number of recommendations')
    args = parser.parse_args()

    movies_df = pd.read_csv(args.movies)
    data, ratings_df = load_data(args.ratings)
    algo = train_model(data)

    unrated_movie_ids = get_unrated_movies(ratings_df, movies_df, args.user)
    if not unrated_movie_ids:
        print('No unrated movies available for recommendation.')
        return

    recommend(algo, args.user, movies_df, unrated_movie_ids, n=args.n)


if __name__ == '__main__':
    main()
