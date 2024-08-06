# Song Recommender



This Flask-based application utilizes Spotify's API and machine learning techniques to provide personalized song recommendations. It analyzes track features (like tempo, danceability, energy) and genres to find songs you'll likely enjoy.

You can get recommendations by Single Track, Playlist or Recently Played. And create a playlist with recommended songs!

Dataset and pre-trained model included.

Updated Version

## Youtube Video:

[![Youtube](https://img.youtube.com/vi/i_qr6q522V8/0.jpg)](https://www.youtube.com/watch?v=i_qr6q522V8)

## Features

-   **Hybrid Recommendation Engine:** Combines content-based filtering (audio features, genres) with collaborative filtering and a deep learning model for enhanced accuracy.
-   **Spotify Integration:** Fetches track and artist data directly from Spotify's API, ensuring up-to-date information.
-   **Genre Analysis:** Incorporates artist genre information to provide recommendations across musical styles.
-   **User-Friendly Interface:** A simple frontend allows users to easily input a Spotify track ID and receive relevant suggestions.
-   **Scalable Dataset:** Dynamically updates the dataset with new tracks fetched from Spotify, enhancing the breadth of recommendations.
-   **Robust Error Handling:** Implements retry mechanisms for Spotify API calls, ensuring a smoother user experience.

## How It Works

1.  **User Input:** The user enters the Spotify ID of a track they like.
2.  **Data Retrieval:** The app fetches audio features and genre data for the input track. If the track isn't in the local dataset, it's retrieved from Spotify's API.
3.  **Feature Transformation:** Audio features are scaled and transformed for model compatibility. Genre data is vectorized using TF-IDF.
4.  **Similarity Calculation:**
    -   **Audio Similarity:** Cosine similarity is calculated between the input track's audio features and those in the dataset. You have the option to use a deep learning model for this step.
    -   **Genre Similarity:** Cosine similarity is calculated between the input track's genre vector and those in the dataset.
    -   **Combined Similarity:** The audio and genre similarities are weighted and combined into a final similarity score.
5.  **Recommendation:** The top 10 most similar tracks are returned to the user.

## Installation and Setup

1.  **Clone the Repository:**
    

    
    ```    Bash
    git clone https://github.com/Mechres/song-recommender.git
    
    ```
    

    
2.  **Install Dependencies:**
    

    
    ```    Bash
    pip install -r requirements.txt 
    
    ```
    
    
3.  **Spotify API Credentials:**
    -   Obtain your Spotify Client ID and Client Secret.
    -   Edit the app.py and add add your credentials.
        
        ``` python
        SPOTIPY_CLIENT_ID="your_client_id"
        SPOTIPY_CLIENT_SECRET="your_client_secret"
        
        ```
        
4.  **Run the App:**
    

    
    ```    Bash
    python app.py
    
    ```
    
    
5.  **Open in Browser:**
    -   Visit `http://127.0.0.1:5000/` in your web browser.

## Usage

1.  Enter a valid Spotify Track ID into the input field.
2.  Click "Get Recommendations."
3.  View the recommended tracks, along with artist and genre information. You can click on each track to open it directly in Spotify.

## API
 Base URL:

``` http
http://127.0.0.1:5000/

```
 Endpoint:

```
/recommend

```

 Method:

```
POST

```

 ### Request Body:



```JSON
{
  "track_id": "spotify track id", 
  "use_deep_learning": true (optional)
}

```
-   **track_id (required):** The Spotify ID of the track for which you want recommendations.
-   **use_deep_learning (optional, default: false):** Set to `true` to use the deep learning model for generating recommendations.


### Response (Success):
``` JSON
[
  {
    "name": "Song Title",
    "id": "Spotify Track ID",
    "artist": "Artist Name",
    "genres": "Genre1 Genre2 ..."
  },
  {
    "name": "Another Song Title",
    "id": "Another Spotify Track ID",
    "artist": "Another Artist Name",
    "genres": "GenreX GenreY ..."
  },
  // ... more recommendations
]
```
An array of JSON objects, each representing a recommended track:

-   **name:** The title of the song.
-   **id:** The Spotify ID of the song.
-   **artist:** The name of the artist.
-   **genres:** The genres associated with the artist.

### Response (Error):
``` JSON
{
  "error": "An error occurred while processing your request"
}
```

In case of an error, the API will return a JSON object with an error message. The HTTP status code will be 500 (Internal Server Error).



### Notes

-   The API may take a few seconds to respond, especially if the input track is not in the local dataset and needs to be fetched from the Spotify API.
-   The `use_deep_learning` parameter is optional and can be omitted if you don't want to use the deep learning model.

## Disclaimer

This project is for educational and personal use. Please adhere to Spotify's API usage guidelines.

## License

This project is open source and available under the [MIT License](LICENSE).
