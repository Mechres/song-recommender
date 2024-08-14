import os
import time
import random
import json
from datetime import timedelta
from functools import wraps
from flask import Flask, request, jsonify, send_from_directory, session, redirect, url_for, render_template, flash
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
from collections import Counter
import sqlite3
from db import create_database
import bcrypt
from flask_login import UserMixin, LoginManager
from flask_login import login_required, login_user, logout_user, current_user
from flask_paginate import Pagination, get_page_parameter
from werkzeug.security import generate_password_hash, check_password_hash
import secrets


logging.basicConfig(level=logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

app = Flask(__name__, template_folder='templates')
CORS(app, supports_credentials=True, origins=["http://127.0.0.1/"])
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(24)

app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)
app.config['SESSION_COOKIE_SECURE'] = True  # for HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # to improve security against CSRF attacks.

# Spotify API credentials
client_id = 'Your_Client_ID' # Update this with your ID
client_secret = 'Your_Client_Secret' # Update this with your Secret
redirect_uri = 'http://127.0.0.1:5000/callback'  # Update this with your redirect URI

# Initialize Spotify client
sp_oauth = SpotifyOAuth(client_id=client_id,
                        client_secret=client_secret,
                        redirect_uri=redirect_uri,
                        scope='user-library-read user-read-recently-played playlist-read-private '
                              'playlist-read-collaborative playlist-modify-public playlist-modify-private '
                              'user-top-read user-read-private user-read-email',
                        cache_path=None)  # Disabled caching

# Initialize Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager, retries=10, status_retries=10,
                     backoff_factor=0.1)

create_database()

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'admin_login'


class AdminUser(UserMixin):
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash)


def create_tables():
    conn = sqlite3.connect('songs_database.db')
    cursor = conn.cursor()

    # Create admin_users table
    cursor.execute('''CREATE TABLE IF NOT EXISTS admin_users
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                       username TEXT UNIQUE NOT NULL,
                       password_hash TEXT NOT NULL)''')

    # Create users table
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id TEXT PRIMARY KEY, 
                        username TEXT NOT NULL,
                        email TEXT)''')

    # Create recommendations table
    cursor.execute('''CREATE TABLE IF NOT EXISTS recommendations
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                       user_id TEXT NOT NULL,
                       song_id TEXT NOT NULL,
                       recommended_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                       FOREIGN KEY (user_id) REFERENCES users (id),
                       FOREIGN KEY (song_id) REFERENCES songs (id))''')

    conn.commit()
    conn.close()


@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect('songs_database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM admin_users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    if user:
        return AdminUser(user[0], user[1], user[2])
    return None


# Call this function once to create an initial admin user
# create_admin_user('admin', 'your_secure_password')
def create_admin_user(username, password):
    conn = sqlite3.connect('songs_database.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS admin_users
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                       username TEXT UNIQUE NOT NULL,
                       password_hash BLOB NOT NULL)''')
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    cursor.execute("INSERT INTO admin_users (username, password_hash) VALUES (?, ?)",
                   (username, hashed_password))
    conn.commit()
    conn.close()

def insert_or_update_user(user_id, username, email):
    conn = sqlite3.connect('songs_database.db')
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO users (id, username, email)
            VALUES (?, ?, ?)
        ''', (user_id, username, email))
        conn.commit()
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()

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

@app.route('/admin/', methods=['GET', 'POST'])
def admin():
    return redirect('/admin/login')

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('songs_database.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM admin_users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user:
            stored_password = user[2]  # Assuming the password hash is stored in the third column
            if isinstance(stored_password, str):
                stored_password = stored_password.encode('utf-8')

            if bcrypt.checkpw(password.encode('utf-8'), stored_password):
                admin_user = AdminUser(user[0], user[1], stored_password)
                login_user(admin_user)
                return redirect(url_for('admin_dashboard'))

        flash('Invalid username or password', 'error')

    return render_template('admin_login.html')


# Add a route for song search
@app.route('/admin/search_songs', methods=['GET'])
@login_required
def search_songs():
    query = request.args.get('query', '')
    conn = sqlite3.connect('songs_database.db')
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM songs 
        WHERE name LIKE ? OR artist LIKE ? OR genres LIKE ?
        ORDER BY id DESC
        LIMIT 50
    """, ('%' + query + '%', '%' + query + '%', '%' + query + '%'))
    songs = cursor.fetchall()
    conn.close()
    return render_template('admin_search_results.html', songs=songs, query=query)


# Add a route for song details
@app.route('/admin/song_details/<song_id>')
@login_required
def song_details(song_id):
    conn = sqlite3.connect('songs_database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM songs WHERE id = ?", (song_id,))
    song = cursor.fetchone()
    conn.close()
    if song:
        return render_template('admin_song_details.html', song=song)
    else:
        flash('Song not found', 'error')
        return redirect(url_for('admin_songs'))


@app.route('/admin/logout')
@login_required
def admin_logout():
    logout_user()
    return redirect(url_for('admin_login'))


@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    conn = sqlite3.connect('songs_database.db')
    cursor = conn.cursor()

    # Get total number of songs
    cursor.execute("SELECT COUNT(*) FROM songs")
    total_songs = cursor.fetchone()[0]

    # Get total number of users
    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]

    # Get total number of recommendations
    cursor.execute("SELECT COUNT(*) FROM recommendations")
    total_recommendations = cursor.fetchone()[0]

    # Get top 5 users with most recommendations
    cursor.execute("""
        SELECT users.username, COUNT(recommendations.id) as rec_count
        FROM users
        LEFT JOIN recommendations ON users.id = recommendations.user_id
        GROUP BY users.id
        ORDER BY rec_count DESC
        LIMIT 5
    """)
    top_users = cursor.fetchall()

    conn.close()

    return render_template('admin_dashboard.html',
                           total_songs=total_songs,
                           total_users=total_users,
                           total_recommendations=total_recommendations,
                           top_users=top_users)
@app.route('/admin/songs', methods=['GET', 'POST'])
@login_required
def admin_songs():
    conn = sqlite3.connect('songs_database.db')
    cursor = conn.cursor()

    if request.method == 'POST':
        song_id = request.form['id']
        name = request.form['name']
        artist = request.form['artist']
        genres = request.form['genres']

        cursor.execute("""
            UPDATE songs 
            SET name = ?, artist = ?, genres = ? 
            WHERE id = ?
        """, (name, artist, genres, song_id))
        conn.commit()
        flash('Song updated successfully!', 'success')

    page = request.args.get(get_page_parameter(), type=int, default=1)
    per_page = 20
    offset = (page - 1) * per_page

    cursor.execute("SELECT COUNT(*) FROM songs")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT * FROM songs ORDER BY id DESC LIMIT ? OFFSET ?", (per_page, offset))
    songs = cursor.fetchall()

    pagination = Pagination(page=page, total=total, per_page=per_page, css_framework='bootstrap4')

    conn.close()
    return render_template('admin_songs.html', songs=songs, pagination=pagination)


@app.route('/admin/delete_song/<song_id>', methods=['POST'])
@login_required
def delete_song(song_id):
    conn = sqlite3.connect('songs_database.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM songs WHERE id = ?", (song_id,))
    conn.commit()
    conn.close()
    flash('Song deleted successfully!', 'success')
    return redirect(url_for('admin_songs'))


@app.route('/admin/statistics')
@login_required
def admin_statistics():
    conn = sqlite3.connect('songs_database.db')
    cursor = conn.cursor()

    # Get total number of users
    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]

    # Get total number of songs
    cursor.execute("SELECT COUNT(*) FROM songs")
    total_songs = cursor.fetchone()[0]

    # Get top 10 most recommended songs
    cursor.execute("""
        SELECT songs.name, songs.artist, COUNT(*) as recommend_count
        FROM recommendations
        JOIN songs ON recommendations.song_id = songs.id
        GROUP BY songs.id
        ORDER BY recommend_count DESC
        LIMIT 10
    """)
    top_recommendations = cursor.fetchall()

    conn.close()

    return render_template('admin_statistics.html', total_users=total_users,
                           total_songs=total_songs, top_recommendations=top_recommendations)


@app.route('/admin/users')
@login_required
def admin_users():
    conn = sqlite3.connect('songs_database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users ORDER BY username ASC")  # ordering by username
    users = cursor.fetchall()
    conn.close()
    return render_template('admin_users.html', users=users)


@app.route('/admin/create_admin', methods=['GET', 'POST'])
@login_required
def create_admin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)

        conn = sqlite3.connect('songs_database.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO admin_users (username, password_hash) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        conn.close()

        flash('New admin user created successfully!', 'success')
        return redirect(url_for('admin_dashboard'))

    return render_template('create_admin.html')


"""
@app.route('/admin/songs')
@login_required
def admin_songs():
    conn = sqlite3.connect('songs_database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM songs LIMIT 100")  # Limit to 100 for performance
    songs = cursor.fetchall()
    conn.close()
    return render_template('admin_songs.html', songs=songs)"""


def insert_recommendation(user_id, song_id):
    conn = sqlite3.connect('songs_database.db')
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO recommendations (user_id, song_id)
            VALUES (?, ?)
        ''', (user_id, song_id))
        conn.commit()
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()


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
def get_track_data(sp, track_id):
    try:
        track_features = sp.audio_features([track_id])[0]
        track_info = sp.track(track_id)
        artist_info = sp.artist(track_info['artists'][0]['id'])

        return {
            'id': track_id,
            'name': track_info['name'],
            'artist': track_info['artists'][0]['name'],
            'genres': ' '.join(artist_info['genres']) if artist_info['genres'] else 'unknown',
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
            'popularity': track_info['popularity']
        }
    except Exception as e:
        print(f"Error in get_track_data: {str(e)}")
        return None


def update_dataset(new_track_data):
    insert_song(new_track_data)

    # Re-load the entire dataset
    global df, X_audio, X_audio_scaled, X_genres
    df = load_or_create_dataset()

    # Update audio features
    X_audio = df[audio_feature_cols]
    X_audio_scaled = scaler.fit_transform(X_audio)

    # Update genre features
    X_genres = tfidf.fit_transform(df['genres'])


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
    conn = sqlite3.connect('songs_database.db')
    cursor = conn.cursor()

    # Check if the songs table exists and has data
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='songs'")
    table_exists = cursor.fetchone()

    if table_exists:
        cursor.execute("SELECT COUNT(*) FROM songs")
        row_count = cursor.fetchone()[0]
    else:
        row_count = 0

    if row_count > 0:
        # If the table exists and has data, load it into a DataFrame
        df = pd.read_sql_query("SELECT * FROM songs", conn)
        print(f"Loaded {len(df)} songs from the existing database.")
    else:
        print("No existing data found. Creating new dataset...")
        # Your Spotify API credentials
        client_id = 'your_client_id'
        client_secret = 'your_client_secret'

        # Initialize Spotify client
        client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

        # List of playlist IDs to fetch songs from
        playlist_ids = [
            'spotify:playlist:37i9dQZF1DX4o1oenSJRJd',  # All Out 00s
            'spotify:playlist:37i9dQZF1DX4io1yPyoLtv',  # Turkish 80's
            'spotify:playlist:7gyeEhrwgxezLxlqhuJa1b',  # Japanese mix
            'spotify:playlist:54H9JhlPEskeP134ljcW6d',  # Japanese Tiktok
            'spotify:playlist:5q0MbTyQ0o954AVRhlAwMB',  # Russian Tiktok
        ]

        all_tracks = []

        for playlist_id in playlist_ids:
            offset = 0
            while True:
                response = sp.playlist_items(playlist_id,
                                             offset=offset,
                                             fields='items.track.id,items.track.name,items.track.artists,total',
                                             additional_types=['track'])

                if len(response['items']) == 0:
                    break

                for item in response['items']:
                    if item['track']:
                        track = item['track']
                        all_tracks.append({
                            'id': track['id'],
                            'name': track['name'],
                            'artist': track['artists'][0]['name'] if track['artists'] else 'Unknown Artist'
                        })

                offset = offset + len(response['items'])
                time.sleep(1)  # To avoid hitting API rate limits

        # Fetch audio features for all tracks
        for i in range(0, len(all_tracks), 100):  # Spotify allows up to 100 tracks per request
            track_ids = [track['id'] for track in all_tracks[i:i + 100]]
            audio_features = sp.audio_features(track_ids)

            for j, features in enumerate(audio_features):
                if features:
                    all_tracks[i + j].update(features)

            time.sleep(1)  # To avoid hitting API rate limits

        # Fetch artist genres
        for track in all_tracks:
            artist_id = sp.track(track['id'])['artists'][0]['id']
            artist_info = sp.artist(artist_id)
            track['genres'] = ' '.join(artist_info['genres']) if artist_info['genres'] else 'unknown'
            time.sleep(0.1)  # To avoid hitting API rate limits

        # Create DataFrame
        df = pd.DataFrame(all_tracks)

        # Save to SQLite database
        df.to_sql('songs', conn, if_exists='replace', index=False)
        print(f"Created new dataset with {len(df)} songs and saved to database.")

    conn.close()
    return df


create_tables()
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


# Function to insert a song into the database
def insert_song(song_data, conn):
    c = conn.cursor()

    columns = '''(id, name, artist, popularity, genres, danceability, energy, key, 
                loudness, mode, speechiness, acousticness, instrumentalness, 
                liveness, valence, tempo)'''

    # Construct the value tuple in the correct order (matching the columns order)
    values = (song_data['id'], song_data['name'], song_data['artist'], song_data['popularity'],
              song_data['genres'], song_data['danceability'], song_data['energy'], song_data['key'],
              song_data['loudness'], song_data['mode'], song_data['speechiness'],
              song_data['acousticness'], song_data['instrumentalness'],
              song_data['liveness'], song_data['valence'], song_data['tempo'])

    sql = f'''INSERT OR REPLACE INTO songs {columns} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'''
    c.execute(sql, values)

    conn.commit()


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
    try:
        if input_type == 'track':
            return [input_id]
        elif input_type == 'playlist':
            playlist_tracks = sp.playlist_tracks(input_id)
            return [item['track']['id'] for item in playlist_tracks['items'] if item['track']]
        elif input_type == 'recent':
            recent_tracks = sp.current_user_recently_played(limit=20)
            return [item['track']['id'] for item in recent_tracks['items']]
        else:
            raise ValueError("Invalid input type")
    except Exception as e:
        print(f"Error in get_input_tracks: {str(e)}")
        return []


def get_recommendations(similarities, df, num_recommendations):
    try:
        similar_indices = similarities.argsort()[::-1][1:num_recommendations + 1]
        return df.iloc[similar_indices][['name', 'id', 'artist', 'genres']].to_dict('records')
    except Exception as e:
        print(f"Error in get_recommendations: {str(e)}")
        return []


def calculate_similarities(input_track, X_audio, X_genres, use_deep_learning, weights, audio_feature_cols, scaler,
                           tfidf, model):
    try:
        input_audio_features = np.array([input_track[col] for col in audio_feature_cols]).reshape(1, -1)

        weight_array = np.array([weights.get(col, 1) for col in audio_feature_cols]).reshape(1, -1)

        weighted_input_features = input_audio_features * weight_array
        weighted_features = X_audio.values * weight_array

        if use_deep_learning:
            input_dl_features = model.predict(weighted_input_features)
            all_dl_features = model.predict(weighted_features)
            audio_similarities = cosine_similarity(input_dl_features, all_dl_features)[0]
        else:
            input_audio_features_scaled = scaler.transform(weighted_input_features)
            X_audio_scaled = scaler.transform(weighted_features)
            audio_similarities = cosine_similarity(input_audio_features_scaled, X_audio_scaled)[0]

        input_genres = tfidf.transform([input_track['genres']])
        genre_similarities = cosine_similarity(input_genres, X_genres)[0]

        return 0.9 * audio_similarities + 0.1 * genre_similarities
    except Exception as e:
        print(f"Error in calculate_similarities: {str(e)}")
        return np.zeros(len(X_audio))


@app.route('/recommend', methods=['POST'])
def recommend():
    sp = get_spotify_client()
    if not sp:
        return jsonify({"error": "Not authenticated"}), 401

    conn = sqlite3.connect('songs_database.db')
    data = request.json
    input_type = data.get('input_type', 'track')
    input_id = data.get('input_id')
    weights = data.get('weights', {})
    exclude_artists = data.get('exclude_artists', [])
    exclude_genres = data.get('exclude_genres', [])
    num_recommendations = int(data.get('num_recommendations', 10))
    use_deep_learning = data.get('use_deep_learning', False)
    mood = data.get('mood', '')


    mood_adjustments = {
        'happy': {'valence': 0.7, 'energy': 0.6},
        'sad': {'valence': 0.3, 'energy': 0.4},
        'energetic': {'energy': 0.8, 'tempo': 0.7},
        'relaxed': {'energy': 0.3, 'acousticness': 0.6}
    }
    if mood in mood_adjustments:
        for feature, value in mood_adjustments[mood].items():
            weights[feature] = weights.get(feature, 0.5) * value

    try:
        input_tracks = get_input_tracks(sp, input_type, input_id)
        all_recommendations = []

        # Load all songs from the database
        df = pd.read_sql_query("SELECT * FROM songs", conn)

        # Prepare features for similarity calculation
        audio_feature_cols = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                              'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        X_audio = df[audio_feature_cols]
        scaler = MinMaxScaler()
        X_audio_scaled = scaler.fit_transform(X_audio)

        tfidf = TfidfVectorizer()
        X_genres = tfidf.fit_transform(df['genres'])

        for track_id in input_tracks:
            # Check if track exists in database
            track_df = df[df['id'] == track_id]
            if track_df.empty:
                new_track_data = get_track_data(sp, track_id)
                if new_track_data:
                    new_track_data['genres'] = str(new_track_data['genres'])
                    insert_song(new_track_data, conn)  # Pass both new_track_data and conn
                    df = pd.read_sql_query("SELECT * FROM songs", conn)
                    track_df = df[df['id'] == track_id]
                else:
                    continue  # Skip this track if we couldn't get its data

            track_df.iloc[0, track_df.columns.get_loc('genres')] = track_df.iloc[0, track_df.columns.get_loc('genres')].lower()
            input_track = track_df.iloc[0]

            similarities = calculate_similarities(input_track, X_audio, X_genres, use_deep_learning, weights,
                                                  audio_feature_cols, scaler, tfidf, model)
            recommendations = get_recommendations(similarities, df,
                                                  num_recommendations * 2)  # Get more recommendations to allow for filtering

            # Filter out excluded artists and genres
            filtered_recommendations = [
                track for track in recommendations
                if track['artist'].lower() not in [a.lower() for a in exclude_artists]
                   and not any(genre.lower() in exclude_genres for genre in track['genres'].split())
            ]

            all_recommendations.extend(filtered_recommendations)

        # Deduplication logic
        seen_tracks = set()
        unique_recommendations = []
        for rec in all_recommendations:
            track_name = rec['name'].lower()
            artist_name = rec['artist'].lower()
            track_key = (track_name, artist_name)
            if track_key not in seen_tracks:
                seen_tracks.add(track_key)
                unique_recommendations.append(rec)

        top_recommendations = unique_recommendations[:num_recommendations]

        # Fetch additional track information from Spotify API
        track_ids = [track['id'] for track in top_recommendations]
        tracks_info = sp.tracks(track_ids)['tracks']
        audio_features = sp.audio_features(track_ids)

        # Combine track info with audio features and save recommendations
        user_id = session.get('user_id')  # Make sure you have the user_id in the session
        for track, info, features in zip(top_recommendations, tracks_info, audio_features):
            track.update({
                'preview_url': info['preview_url'],
                'album': {
                    'name': info['album']['name'],
                    'images': info['album']['images']
                },
                'danceability': features['danceability'],
                'energy': features['energy'],
                'valence': features['valence'],
                'acousticness': features['acousticness'],
                'instrumentalness': features['instrumentalness'],
                'liveness': features['liveness']
            })
            # Save this recommendation to the database
            insert_recommendation(user_id, track['id'])

        return jsonify(top_recommendations)

    except Exception as e:
        print(f"Error in recommend function: {str(e)}")
        return jsonify({"error": "An error occurred while processing your request"}), 500
    finally:
        conn.close()


# New route to get access token for Spotify Web Playback SDK
@app.route('/get-spotify-token', methods=['GET'])
def get_spotify_token():
    sp = get_spotify_client()
    if not sp:
        return jsonify({"error": "Not authenticated"}), 401

    token_info = session.get('token_info', None)
    if token_info:
        return jsonify({"token": token_info['access_token']})
    else:
        return jsonify({"error": "No token available"}), 401


@app.route('/get-token')
def get_token():
    token_info = session.get('token_info', None)
    if not token_info:
        return jsonify({"error": "Not authenticated"}), 401

    now = int(time.time())
    is_expired = token_info['expires_at'] - now < 60

    if is_expired:
        try:
            token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
            session['token_info'] = token_info
        except Exception as e:
            return jsonify({"error": "Failed to refresh token"}), 500

    return jsonify({"token": token_info['access_token']})


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

            user_id = str(user_info['id'])
            username = str(user_info['display_name'])
            email = str(user_info['email']) if user_info['email'] is not None else ""

            # Check if the user already exists
            conn = sqlite3.connect('songs_database.db')
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            existing_user = cursor.fetchone()

            if not existing_user:
                print(f"Inserting user: {user_id}, {username}, {email}")
                cursor.execute("INSERT INTO users (id, username, email) VALUES (?, ?, ?)",
                               (user_id, username, email))
                conn.commit()
            else:
                conn.close()

            session['user_id'] = user_id

            return redirect('/')

        except sqlite3.Error as e:
            logging.error(f"Error in callback (SQL): {str(e)}")
            return jsonify({"error": str(e)}), 500
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
        # Revoke token using Spotify's Web API(Doesn't exist!)
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
    try:
        token_info = session.get('token_info', None)
        if not token_info:
            raise Exception("No token info in session")

        now = int(time.time())
        is_expired = token_info['expires_at'] - now < 60

        if is_expired:
            sp_oauth = spotipy.oauth2.SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope='user-library-read user-read-recently-played playlist-read-private playlist-read-collaborative user-top-read'
            )
            token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
            session['token_info'] = token_info

        return spotipy.Spotify(auth=token_info['access_token'])
    except Exception as e:
        print(f"Error in get_spotify_client: {str(e)}")
        return None


@app.route('/user-playlists')
def get_user_playlists():
    sp = get_spotify_client()
    if not sp:
        logging.error("Failed to get Spotify client in user-playlists route")
        return jsonify({"error": "Not authenticated"}), 401

    try:
        playlists = sp.current_user_playlists()
        logging.debug(f"Spotify API response for playlists: {playlists}")  # Add this line
        return jsonify(playlists)
    except Exception as e:
        logging.error(f"Error in get_user_playlists: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/recently-played')
def get_recently_played():
    sp = get_spotify_client()
    if not sp:
        return jsonify({"error": "Not authenticated"}), 401

    recent_tracks = sp.current_user_recently_played(limit=20)
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


@app.before_request
def log_request_info():
    logging.debug('Headers: %s', request.headers)
    logging.debug('Body: %s', request.get_data())


@app.after_request
def log_response_info(response):
    logging.debug('Response Status: %s', response.status)
    logging.debug('Response Headers: %s', response.headers)
    return response


@app.route('/profile')
def profile():
    sp = get_spotify_client()
    if not sp:
        return redirect(url_for('login'))

    try:
        user_info = sp.me()
        top_tracks = sp.current_user_top_tracks(limit=10, time_range='medium_term')['items']
        top_artists = sp.current_user_top_artists(limit=10, time_range='medium_term')['items']
        playlists = sp.current_user_playlists(limit=10)['items']
        recently_played = sp.current_user_recently_played(limit=20)['items']  # Fetch recently played tracks

        # Get favorite genres
        all_genres = [genre for artist in top_artists for genre in artist['genres']]
        genre_counts = Counter(all_genres)
        favorite_genres = genre_counts.most_common(5)

        return render_template('profile.html',
                               user_info=user_info,
                               top_tracks=top_tracks,
                               top_artists=top_artists,
                               favorite_genres=favorite_genres,
                               playlists=playlists,
                               recently_played=recently_played)  # Pass recently played tracks to the template
    except Exception as e:
        print(f"Error fetching user profile data: {str(e)}")
        return jsonify({"error": "An error occurred while fetching your profile data"}), 500


if __name__ == '__main__':
    app.run(debug=True)
