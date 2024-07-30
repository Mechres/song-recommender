import os
import time
import random
import json
from functools import wraps
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv1D, LSTM, GlobalMaxPooling1D, concatenate

app = Flask(__name__)
CORS(app)

# Spotify API credentials
client_id = 'your_client_id'
client_secret = 'your_client_secret'

# Initialize Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager, retries=10, status_retries=10,
                     backoff_factor=0.1)


def exponential_backoff(retries):
    return min(60, (2 ** retries) + (random.randint(0, 1000) / 1000))


def retry_with_exponential_backoff(
        func,
        retries=10,
        backoff_in_seconds=1,
        max_backoff_in_seconds=300,
        jitter=True
):
    @wraps(func)
    def wrapper(*args, **kwargs):
        x = 0
        while True:
            try:
                return func(*args, **kwargs)
            except spotipy.exceptions.SpotifyException as e:
                if x == retries:
                    print(f"Max retries reached for function {func.__name__}")
                    raise
                if e.http_status == 429:  # Too Many Requests
                    sleep = (backoff_in_seconds * 2 ** x
                             + (random.randint(0, 1000) / 1000.0 if jitter else 0))
                    sleep = min(sleep, max_backoff_in_seconds)
                    print(f"Rate limited. Retrying {func.__name__} in {sleep:.2f} seconds.")
                    time.sleep(sleep)
                    x += 1
                else:
                    raise

    return wrapper


def get_tracks_features(track_ids, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            features = sp.audio_features(track_ids)
            return features
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 429:
                wait_time = exponential_backoff(retries)
                print(f"Rate limited. Waiting for {wait_time} seconds.")
                time.sleep(wait_time)
                retries += 1
            else:
                print(f"Error fetching track features: {str(e)}")
                return None
        except Exception as e:
            print(f"Unexpected error fetching track features: {str(e)}")
            return None

    print(f"Max retries reached for tracks {track_ids}")
    return None


@retry_with_exponential_backoff
def get_track_data(track_id):
    track_features = sp.audio_features([track_id])[0]
    track_info = sp.track(track_id)
    artist_info = sp.artist(track_info['artists'][0]['id'])

    return {
        'danceability': track_features['danceability'],
        'energy': track_features['energy'],
        'key': track_features['key'],
        'loudness': track_features['loudness'],
        'mode': track_features['mode'],
        'speechiness': track_features['speechiness'],
        'acousticness': track_features['acousticness'],
        'instrumentalness': track_features['instrumentalness'],
        'liveness': track_features['liveness'],
        'valence': track_features['valence'],
        'tempo': track_features['tempo'],
        'name': track_info['name'],
        'id': track_info['id'],
        'artist': track_info['artists'][0]['name'],
        'popularity': track_info['popularity'],
        'genres': ' '.join(artist_info['genres']) if artist_info['genres'] else 'unknown'
    }


def update_dataset(new_track_data):
    global df, X_audio, X_audio_scaled, X_genres

    # Add new track to dataframe
    df = pd.concat([df, pd.DataFrame([new_track_data])], ignore_index=True)

    # Update audio features
    X_audio = df[audio_feature_cols]
    X_audio_scaled = scaler.fit_transform(X_audio)

    # Update genre features
    X_genres = tfidf.fit_transform(df['genres'])

    # Save updated dataset
    df.to_csv('songs_dataset.csv', index=False)


def get_tracks_info(track_ids, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            tracks_info = sp.tracks(track_ids)
            return tracks_info['tracks']
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 429:
                wait_time = exponential_backoff(retries)
                print(f"Rate limited. Waiting for {wait_time} seconds.")
                time.sleep(wait_time)
                retries += 1
            else:
                print(f"Error fetching track info: {str(e)}")
                return None
        except Exception as e:
            print(f"Unexpected error fetching track info: {str(e)}")
            return None

    print(f"Max retries reached for tracks {track_ids}")
    return None


def get_artists_info(artist_ids, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            artists_info = sp.artists(artist_ids)
            return artists_info['artists']
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 429:
                wait_time = exponential_backoff(retries)
                print(f"Rate limited. Waiting for {wait_time} seconds.")
                time.sleep(wait_time)
                retries += 1
            else:
                print(f"Error fetching artist info: {str(e)}")
                return None
        except Exception as e:
            print(f"Unexpected error fetching artist info: {str(e)}")
            return None

    print(f"Max retries reached for artists {artist_ids}")
    return None


def sanitize_filename(filename):
    return ''.join(c for c in filename if c.isalnum() or c in ['-', '_']).rstrip()


def save_progress(playlist_id, processed_tracks):
    sanitized_id = sanitize_filename(playlist_id)
    with open(f'progress_{sanitized_id}.json', 'w') as f:
        json.dump(processed_tracks, f)


def load_progress(playlist_id):
    sanitized_id = sanitize_filename(playlist_id)
    try:
        with open(f'progress_{sanitized_id}.json', 'r') as f:
            return set(json.load(f))
    except FileNotFoundError:
        return set()


def load_or_create_dataset():
    if os.path.exists('songs_dataset.csv'):
        return pd.read_csv('songs_dataset.csv')
    else:
        # Spotify Api probably won't allow this
        playlist_ids = [
            'spotify:playlist:37i9dQZF1DX4o1oenSJRJd',  # All Out 00s
            'spotify:playlist:37i9dQZF1DX4io1yPyoLtv',  # Turkish 80's
            'spotify:playlist:7gyeEhrwgxezLxlqhuJa1b',  # Japanese mix
            'spotify:playlist:54H9JhlPEskeP134ljcW6d',  # Japanese Tiktok
            'spotify:playlist:5q0MbTyQ0o954AVRhlAwMB',  # Russian Tiktok
        ]

        all_data = []
        for playlist_id in playlist_ids:
            processed_tracks = load_progress(playlist_id)
            results = sp.playlist_items(playlist_id)
            tracks = results['items']

            batch_size = 50  # Spotify API allows up to 50 tracks per request
            for i in range(0, len(tracks), batch_size):
                batch = tracks[i:i + batch_size]
                track_ids = [track['track']['id'] for track in batch if track['track']['id'] not in processed_tracks]

                if not track_ids:
                    continue

                features = get_tracks_features(track_ids)
                tracks_info = get_tracks_info(track_ids)
                artist_ids = list(set(track['artists'][0]['id'] for track in tracks_info))
                artists_info = get_artists_info(artist_ids)

                if features and tracks_info and artists_info:
                    for feature, track_info in zip(features, tracks_info):
                        artist_info = next(
                            (artist for artist in artists_info if artist['id'] == track_info['artists'][0]['id']), None)
                        if feature and artist_info:
                            data = {
                                'danceability': feature['danceability'],
                                'energy': feature['energy'],
                                'key': feature['key'],
                                'loudness': feature['loudness'],
                                'mode': feature['mode'],
                                'speechiness': feature['speechiness'],
                                'acousticness': feature['acousticness'],
                                'instrumentalness': feature['instrumentalness'],
                                'liveness': feature['liveness'],
                                'valence': feature['valence'],
                                'tempo': feature['tempo'],
                                'name': track_info['name'],
                                'id': track_info['id'],
                                'artist': track_info['artists'][0]['name'],
                                'popularity': track_info['popularity'],
                                'genres': ' '.join(artist_info['genres']) if artist_info['genres'] else 'unknown'
                            }
                            all_data.append(data)
                            processed_tracks.add(track_info['id'])

                save_progress(playlist_id, list(processed_tracks))
                time.sleep(1)  # Add a small delay between batches

        df = pd.DataFrame(all_data)
        df.to_csv('songs_dataset.csv', index=False)
        return df


# Load the dataset
df = load_or_create_dataset()

# Handle NaN values in genres
df['genres'] = df['genres'].fillna('unknown')

# Prepare features for similarity calculation
audio_feature_cols = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                      'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
X_audio = df[audio_feature_cols]
scaler = MinMaxScaler()
X_audio_scaled = scaler.fit_transform(X_audio)

# Prepare genre features
tfidf = TfidfVectorizer()
X_genres = tfidf.fit_transform(df['genres'])


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


# Deep Learning model
def create_hybrid_model(input_shape):
    inputs = Input(shape=input_shape)

    # CNN branch
    conv = Conv1D(64, kernel_size=3, activation='relu')(inputs)
    conv = Conv1D(64, kernel_size=3, activation='relu')(conv)
    conv = GlobalMaxPooling1D()(conv)

    # RNN branch
    lstm = LSTM(64, return_sequences=True)(inputs)
    lstm = LSTM(64)(lstm)

    # Concatenate CNN and RNN outputs
    concatenated = concatenate([conv, lstm])

    # Dense layers
    dense = Dense(64, activation='relu')(concatenated)
    outputs = Dense(len(audio_feature_cols))(dense)  # Output dimension matches input features

    model = Model(inputs=inputs, outputs=outputs)
    return model


# Train or load the model
model_path = 'music_recommendation_model.h5'
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    X_train = X_audio_scaled.reshape(X_audio_scaled.shape[0], X_audio_scaled.shape[1], 1)
    model = create_hybrid_model((X_train.shape[1], 1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, X_audio_scaled, epochs=100, batch_size=32, validation_split=0.2)
    model.save(model_path)


# Function to get deep learning features
def get_dl_features(audio_features):
    audio_features_scaled = scaler.transform(audio_features)
    audio_features_reshaped = audio_features_scaled.reshape(audio_features_scaled.shape[0],
                                                            audio_features_scaled.shape[1], 1)
    return model.predict(audio_features_reshaped)


@app.route('/recommend', methods=['POST'])
def recommend():
    global df, X_audio_scaled, X_genres

    data = request.json
    input_track_id = data['track_id']
    use_deep_learning = data.get('use_deep_learning', False)

    try:
        # Check if the track is in our dataset
        if input_track_id not in df['id'].values:
            print(f"Track {input_track_id} not found in dataset. Fetching from Spotify API.")
            new_track_data = get_track_data(input_track_id)
            update_dataset(new_track_data)
            print(f"Added new track {new_track_data['name']} to dataset.")

        # Get input track data
        input_track = df[df['id'] == input_track_id].iloc[0]

        if use_deep_learning:
            # Use deep learning model for similarity
            input_audio_features = input_track[audio_feature_cols].values.reshape(1, -1)
            input_dl_features = get_dl_features(input_audio_features)
            all_dl_features = get_dl_features(X_audio.values)
            audio_similarities = cosine_similarity(input_dl_features, all_dl_features)
        else:
            # Use original cosine similarity
            input_audio_features_scaled = scaler.transform(input_track[audio_feature_cols].values.reshape(1, -1))
            audio_similarities = cosine_similarity(input_audio_features_scaled, X_audio_scaled)

        input_genres = tfidf.transform([input_track['genres']])
        genre_similarities = cosine_similarity(input_genres, X_genres)

        # Combine similarities (you can adjust weights here)
        combined_similarities = 0.7 * audio_similarities + 0.3 * genre_similarities

        # Get top 10 similar tracks
        similar_indices = combined_similarities[0].argsort()[-11:-1][::-1]  # Exclude the input track itself
        recommendations = df.iloc[similar_indices][['name', 'id', 'artist', 'genres']].to_dict('records')

        return jsonify(recommendations)

    except Exception as e:
        print(f"Error in recommend function: {str(e)}")
        return jsonify({"error": "An error occurred while processing your request"}), 500


if __name__ == '__main__':
    app.run(debug=True)
