let player;
let recommendedTracks = [];

window.onSpotifyWebPlaybackSDKReady = () => {
    player = new Spotify.Player({
        name: 'Song Recommender Web Player',
        getOAuthToken: cb => {
            fetch('/get-token')
                .then(response => response.json())
                .then(data => cb(data.token));
        }
    });

    player.connect();
};

function showLoader() {
    document.getElementById('loader').style.display = 'block';
}

function hideLoader() {
    document.getElementById('loader').style.display = 'none';
}

function updateSliderValue(sliderId) {
    const slider = document.getElementById(sliderId);
    const output = document.getElementById(sliderId + '-value');
    output.innerHTML = slider.value;
    slider.oninput = function() {
        output.innerHTML = this.value;
    }
}

// Initialize slider values
updateSliderValue('danceability');
updateSliderValue('energy');
updateSliderValue('valence');

function getRecommendations() {
    showLoader();

    const inputType = document.getElementById('input-type').value;
    const inputId = document.getElementById('input-id').value;
    const weights = {
        danceability: document.getElementById('danceability').value,
        energy: document.getElementById('energy').value,
        valence: document.getElementById('valence').value
    };
    const excludeArtists = document.getElementById('exclude-artists').value.split(',').map(a => a.trim());
    const excludeGenres = document.getElementById('exclude-genres').value.split(',').map(g => g.trim());

    $.ajax({
        url: '/recommend',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            input_type: inputType,
            input_id: inputId,
            weights: weights,
            exclude_artists: excludeArtists,
            exclude_genres: excludeGenres
        }),
        success: function(data) {
            hideLoader();
            recommendedTracks = data;
            displayRecommendations(data);
        },
        error: function(jqXHR, textStatus, errorThrown) {
            hideLoader();
            alert('Error getting recommendations: ' + errorThrown);
        }
    });
}

function displayRecommendations(recommendations) {
    const container = document.getElementById('recommendations');
    container.innerHTML = '<h2>Recommendations</h2>';

    recommendations.forEach(track => {
        const trackElement = document.createElement('div');
        trackElement.className = 'track-item';
        trackElement.id = `track-${track.id}`;
        trackElement.innerHTML = `
            <p>${track.name} by ${track.artists[0].name}</p>
            <button onclick="previewTrack('${track.id}')">Preview</button>
        `;
        container.appendChild(trackElement);
    });

    document.getElementById('createPlaylistSection').style.display = 'block';
}

function previewTrack(trackId) {
    if (player) {
        player.play(`spotify:track:${trackId}`);
    } else {
        alert('Spotify player is not ready. Please try again in a moment.');
    }
}

function createPlaylist() {
    const playlistName = document.getElementById('playlistName').value;
    if (!playlistName) {
        alert('Please enter a playlist name.');
        return;
    }

    const trackUris = recommendedTracks.map(track => track.uri);

    $.ajax({
        url: '/create-playlist',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            name: playlistName,
            tracks: trackUris
        }),
        success: function(data) {
            alert(`Playlist "${playlistName}" created successfully!`);
        },
        error: function(jqXHR, textStatus, errorThrown) {
            alert('Error creating playlist: ' + errorThrown);
        }
    });
}