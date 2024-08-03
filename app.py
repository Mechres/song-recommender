import os
import time
import random
import json
from datetime import timedelta
from functools import wraps
from flask import Flask, request, jsonify, send_from_directory, session, redirect, url_for
from flask_cors import CORS
import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Conv1D, LSTM, GlobalMaxPooling1D, concatenate
import requests
import logging
import uuid

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('urllib3').setLevel(logging.WARNING)

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:5000"])  # Replace with your frontend URL

app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(24)

app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)
app.config['SESSION_COOKIE_SECURE'] = True  # for HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # To improve security against CSRF attacks.

# Spotify API credentials
client_id = 'client_id'
client_secret = 'client_secret'
redirect_uri = 'http://127.0.0.1:5000/callback'  # Update this with your redirect URI

# Initialize Spotify client
sp_oauth = SpotifyOAuth(client_id=client_id,
                        client_secret=client_secret,
                        redirect_uri=redirect_uri,
                        scope='user-library-read user-read-recently-played playlist-read-private playlist-read-collaborative playlist-modify-public playlist-modify-private',
                        cache_path=None)  # Disabled caching

# Initialize Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager, retries=10, status_retries=10,
                     backoff_factor=0.1)


@app.before_request
def make_session_permanent():
    session.permanent = True
    app.permanent_session_lifetime = timedelta(days=1)


def exponential_backoff(retries):
    return min(60, (2 ** retries) + (random.randint(0, 1000) / 1000))


def retry_with_exponential_backoff(
        func,
        retries=5,
        backoff_in_seconds=1,
        max_backoff_in_seconds=60
):
    @wraps(func)
    def wrapper(*args, **kwargs):
        x = 0
        while True:
            try:
                return func(*args, **kwargs)
            except spotipy.SpotifyException as e:
                if x == retries:
                    raise
                if e.http_status == 429:
                    sleep = min(backoff_in_seconds * 2 ** x + random.uniform(0, 1), max_backoff_in_seconds)
                    logging.warning(f"Rate limited. Retrying in {sleep:.2f} seconds")
                    time.sleep(sleep)
                    x += 1
                else:
                    raise

    return wrapper


@retry_with_exponential_backoff
def get_user_playlists(sp):
    return sp.current_user_playlists()


@app.route('/recommendations')
def recommendations_page():
    if not session.get('token_info'):
        return redirect(url_for('login'))
    return send_from_directory('.', 'recommendations.html')


@app.route('/check-auth')
def check_auth():
    if not session.get('token_info'):
        return jsonify({"authenticated": False})
    return jsonify({"authenticated": True})


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
    if os.path.exists('new_dataset.csv'):
        df = pd.read_csv('new_dataset.csv')
        df['genres'] = df['genres'].fillna('unknown')
        return df
    else:
        # Collect data for a diverse set of songs (Too much for rate limit)
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

        # Explicitly convert genres to string to prevent future issues
        df['genres'] = df['genres'].astype(str)

        df.to_csv('new_dataset.csv', index=False)
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

tfidf = TfidfVectorizer()
X_genres = tfidf.fit_transform(df['genres'])


# Deep Learning model
def create_hybrid_model(input_shape):
    inputs = Input(shape=input_shape)
    conv = Conv1D(64, kernel_size=3, activation='relu')(inputs)
    conv = Conv1D(64, kernel_size=3, activation='relu')(conv)
    conv = GlobalMaxPooling1D()(conv)
    lstm = LSTM(64, return_sequences=True)(inputs)
    lstm = LSTM(64)(lstm)
    concatenated = concatenate([conv, lstm])
    dense = Dense(64, activation='relu')(concatenated)
    outputs = Dense(len(audio_feature_cols))(dense)
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


def get_input_tracks(sp, input_type, input_id):
    if input_type == 'track':
        return [input_id]
    elif input_type == 'playlist':
        playlist_tracks = sp.playlist_tracks(input_id)
        return [item['track']['id'] for item in playlist_tracks['items']]
    elif input_type == 'recent':
        recent_tracks = sp.current_user_recently_played(limit=20)
        return [item['track']['id'] for item in recent_tracks['items']]
    else:
        raise ValueError("Invalid input type")


def get_recommendations(similarities):
    similar_indices = similarities[0].argsort()[::-1][1:]
    return df.iloc[similar_indices][['name', 'id', 'artist', 'genres']].to_dict('records')


def calculate_similarities(input_track, use_deep_learning):
    if use_deep_learning:
        input_audio_features = input_track[audio_feature_cols].values.reshape(1, -1)
        input_dl_features = get_dl_features(input_audio_features)
        all_dl_features = get_dl_features(X_audio.values)
        audio_similarities = cosine_similarity(input_dl_features, all_dl_features)
    else:
        input_audio_features_scaled = scaler.transform(input_track[audio_feature_cols].values.reshape(1, -1))
        audio_similarities = cosine_similarity(input_audio_features_scaled, X_audio_scaled)

    input_genres = tfidf.transform([input_track['genres']])
    genre_similarities = cosine_similarity(input_genres, X_genres)

    return 0.7 * audio_similarities + 0.3 * genre_similarities  # 70 % Audio Features, %30 Genre


@app.route('/recommend', methods=['POST'])
def recommend():
    sp = get_spotify_client()
    if not sp:
        logging.error("Failed to get Spotify client in recommend route")
        return jsonify({"error": "Not authenticated"}), 401

    data = request.json
    input_type = data.get('input_type', 'track')
    input_id = data.get('input_id')
    use_deep_learning = data.get('use_deep_learning', False)
    num_recommendations = data.get('num_recommendations', 10)

    try:
        input_tracks = get_input_tracks(sp, input_type, input_id)
        all_recommendations = []

        for track_id in input_tracks:
            if track_id not in df['id'].values:
                new_track_data = get_track_data(track_id)
                new_track_data['genres'] = str(new_track_data['genres'])
                update_dataset(new_track_data)

            input_track = df[df['id'] == track_id].iloc[0]
            input_track['genres'] = input_track['genres'].lower()

            # Cache similarities calculation
            similarities = get_cached_similarities(input_track, use_deep_learning)

            # Get recommendations without caching
            recommendations = get_recommendations(similarities)
            all_recommendations.extend(recommendations)

        # Deduplication logic (Fixes showing one song more than one!)
        seen_tracks = set()
        unique_recommendations = []
        for rec in all_recommendations:
            track_name = rec['name'].lower() if isinstance(rec['name'], str) else rec['name']
            artist_name = rec['artist'].lower() if isinstance(rec['artist'], str) else rec['artist']
            track_key = (track_name, artist_name)
            if track_key not in seen_tracks:
                seen_tracks.add(track_key)
                unique_recommendations.append(rec)

        top_recommendations = unique_recommendations[:int(num_recommendations)]
        return jsonify(top_recommendations)

    except Exception as e:
        print(f"Error in recommend function: {str(e)}")
        return jsonify({"error": "An error occurred while processing your request"}), 500


def get_cached_similarities(input_track, use_deep_learning):
    return calculate_similarities(input_track, use_deep_learning)


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/login')
def login():
    logging.debug("Login route accessed")
    # Generate a unique state for this login attempt
    state = str(uuid.uuid4())
    session['state'] = state
    auth_url = sp_oauth.get_authorize_url(state=state)
    logging.debug(f"Generated auth URL: {auth_url}")
    return redirect(auth_url)


@app.route('/callback')
def callback():
    logging.debug("Callback route accessed")
    code = request.args.get('code')
    state = request.args.get('state')

    if state != session.get('state'):
        logging.error("State mismatch in callback")
        return jsonify({"error": "State mismatch. Possible CSRF attack."}), 400

    if code:
        try:
            token_info = sp_oauth.get_access_token(code, check_cache=False)
            session['token_info'] = token_info

            sp = spotipy.Spotify(auth=token_info['access_token'])
            user_info = sp.me()
            session['user_id'] = user_info['id']

            return redirect('/')
        except Exception as e:
            logging.error(f"Error in callback: {str(e)}")
            return jsonify({"error": str(e)}), 500
    else:
        logging.error("No code received in callback")
        return jsonify({"error": "Authorization code missing"}), 400


@app.route('/logout')
def logout():
    token_info = session.get('token_info')
    user_id = session.get('user_id')

    if token_info:
        # Revoke token using Spotify's Web API (Doesn't work! I'll fix this later.)
        url = 'https://accounts.spotify.com/api/token/revoke'
        data = {
            'token': token_info['access_token'],
            'client_id': client_id,
            'client_secret': client_secret
        }
        try:
            response = requests.post(url, data=data)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error revoking token: {str(e)}")

    # Clear session
    session.clear()

    # Clear cache
    if user_id:
        try:
            os.remove(f".cache-{user_id}")
        except FileNotFoundError:
            pass

    return redirect(url_for('index'))


def get_spotify_client():
    token_info = session.get('token_info', None)
    logging.debug(f"Token info from session: {token_info}")
    if not token_info:
        logging.error("No token info in session")
        return None

    now = int(time.time())
    is_expired = token_info['expires_at'] - now < 60

    if is_expired:
        try:
            token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
            session['token_info'] = token_info
        except Exception as e:
            logging.error(f"Error refreshing token: {str(e)}")
            return None

    return spotipy.Spotify(auth=token_info['access_token'])


@app.route('/user-playlists')
def get_user_playlists():
    sp = get_spotify_client()
    if not sp:
        logging.error("Failed to get Spotify client in user-playlists route")
        return jsonify({"error": "Not authenticated"}), 401

    try:
        playlists = sp.current_user_playlists()
        logging.debug(f"Spotify API response for playlists: {playlists}")
        return jsonify(playlists)
    except Exception as e:
        logging.error(f"Error in get_user_playlists: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/recently-played')
def get_recently_played():
    sp = get_spotify_client()
    if not sp:
        return jsonify({"error": "Not authenticated"}), 401

    recent_tracks = sp.current_user_recently_played()
    return jsonify(recent_tracks)


@app.route('/create-playlist', methods=['POST'])
def create_playlist():
    sp = get_spotify_client()
    if not sp:
        return jsonify({"error": "Not authenticated"}), 401

    data = request.json
    playlist_name = data.get('name', 'My Recommended Playlist')
    track_uris = data.get('tracks', [])

    try:
        user_id = sp.me()['id']
        playlist = sp.user_playlist_create(user_id, playlist_name)
        sp.playlist_add_items(playlist['id'], track_uris)
        return jsonify({"message": "Playlist created successfully", "playlist_id": playlist['id']}), 200
    except Exception as e:
        print(f"Error creating playlist: {str(e)}")
        return jsonify({"error": "An error occurred while creating the playlist"}), 500


# Add this to your routes
@app.before_request
def log_request_info():
    logging.debug('Headers: %s', request.headers)
    logging.debug('Body: %s', request.get_data())


@app.after_request
def log_response_info(response):
    logging.debug('Response Status: %s', response.status)
    logging.debug('Response Headers: %s', response.headers)
    return response


if __name__ == '__main__':
    app.run(debug=True)
