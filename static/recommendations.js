
document.addEventListener('DOMContentLoaded', function() {
    const singleTrackBtn = document.getElementById('singleTrackBtn');
    const playlistBtn = document.getElementById('playlistBtn');
    const recentlyPlayedBtn = document.getElementById('recentlyPlayedBtn');
    const inputField = document.getElementById('inputField');
    const playlistDropdown = document.getElementById('playlistDropdown');
    const playlistSelect = document.getElementById('playlistSelect');
    const getRecommendationsBtn = document.getElementById('getRecommendationsBtn');
    const recommendationsContainer = document.getElementById('recommendations');
    const createPlaylistSection = document.getElementById('createPlaylistSection');
    const createPlaylistBtn = document.getElementById('createPlaylistBtn');
    const audioFeaturesChart = document.getElementById('audioFeaturesChart');
    let radarChart;

    let currentInputType = 'track';
    let recommendedTracks = [];

    singleTrackBtn.addEventListener('click', () => setInputType('track'));
    playlistBtn.addEventListener('click', () => setInputType('playlist'));
    recentlyPlayedBtn.addEventListener('click', () => setInputType('recent'));

    function setInputType(type) {
        currentInputType = type;
        inputField.classList.toggle('hidden', type !== 'track');
        playlistDropdown.classList.toggle('hidden', type !== 'playlist');
        if (type === 'playlist') {
            fetchPlaylists();
        }
    }

    function fetchPlaylists() {
        fetch('/user-playlists')
            .then(response => response.json())
            .then(data => {
                playlistSelect.innerHTML = '<option value="">Select a playlist</option>';
                data.items.forEach(playlist => {
                    const option = document.createElement('option');
                    option.value = playlist.id;
                    option.textContent = playlist.name;
                    playlistSelect.appendChild(option);
                });
            })
            .catch(error => console.error('Error fetching playlists:', error));
    }

    getRecommendationsBtn.addEventListener('click', getRecommendations);

    function getRecommendations() {
        const loader = document.getElementById('loader');
        loader.classList.remove('hidden');
        recommendationsContainer.innerHTML = '';
        audioFeaturesChart.classList.add('hidden');

        const inputId = currentInputType === 'track' ? document.getElementById('trackInput').value :
                        currentInputType === 'playlist' ? playlistSelect.value : '';

        const requestBody = {
            input_type: currentInputType,
            input_id: inputId,
            weights: {
                danceability: document.getElementById('danceability').value,
                energy: document.getElementById('energy').value,
                valence: document.getElementById('valence').value
            },
            exclude_artists: document.getElementById('excludeArtists').value.split(',').map(artist => artist.trim()),
            exclude_genres: document.getElementById('excludeGenres').value.split(',').map(genre => genre.trim()),
            num_recommendations: document.getElementById('numRecommendations').value,
            use_deep_learning: document.getElementById('useDeepLearning').checked
        };

        fetch('/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody),
        })
        .then(response => response.json())
        .then(data => {
            loader.classList.add('hidden');
            recommendedTracks = data;
            displayRecommendations(data);
            createPlaylistSection.classList.remove('hidden');
            updateAudioFeaturesChart(data);
        })
        .catch(error => {
            console.error('Error:', error);
            loader.classList.add('hidden');
        });
    }

    function displayRecommendations(tracks) {
        recommendationsContainer.innerHTML = '';
        tracks.forEach(track => {
            const trackElement = document.createElement('div');
            trackElement.className = 'bg-white rounded-lg shadow-md p-4';
            trackElement.innerHTML = `
                <img src="${track.album.images[1].url}" alt="${track.name}" class="w-full h-48 object-cover mb-4">
                <h3 class="font-bold">${track.name}</h3>
                <p>${track.artists.map(artist => artist.name).join(', ')}</p>
                <audio controls class="w-full mt-2">
                    <source src="${track.preview_url}" type="audio/mpeg">
                    Your browser does not support the audio element.
                </audio>
            `;
            recommendationsContainer.appendChild(trackElement);
        });
    }

    function updateAudioFeaturesChart(tracks) {
        const features = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'liveness'];
        const averageFeatures = features.reduce((acc, feature) => {
            acc[feature] = tracks.reduce((sum, track) => sum + track[feature], 0) / tracks.length;
            return acc;
        }, {});

        const chartData = {
            labels: features,
            datasets: [{
                label: 'Average Audio Features',
                data: features.map(feature => averageFeatures[feature]),
                fill: true,
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgb(54, 162, 235)',
                pointBackgroundColor: 'rgb(54, 162, 235)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgb(54, 162, 235)'
            }]
        };

        if (radarChart) {
            radarChart.destroy();
        }

        const ctx = document.getElementById('radarChart').getContext('2d');
        if (!ctx) {
  console.error('Failed to get context for radar chart');
}

const fetchRecommendations = async () => {

  const payload = {
    numRecommendations: document.getElementById('numRecommendations').value,
    useDeepLearning: document.getElementById('useDeepLearning').checked,
    featureWeights: {
      danceability: document.getElementById('danceability').value,
      energy: document.getElementById('energy').value,
      valence: document.getElementById('valence').value,
    },
    excludeArtists: document.getElementById('excludeArtists').value.split(',').map(artist => artist.trim()),
    excludeGenres: document.getElementById('excludeGenres').value.split(',').map(genre => genre.trim())
  };

  // Show loader
  document.getElementById('loader').classList.remove('hidden');

  try {
    const response = await fetch('/api/get-recommendations', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    const data = await response.json();

    // Check if data structure is as expected
    if (!data || typeof data.danceability === 'undefined') {
      console.error('Unexpected data structure:', data);
      // Hide loader
      document.getElementById('loader').classList.add('hidden');
      return;
    }

    // Update the chart with the new recommendations
    updateChart(data);

    // Hide loader
    document.getElementById('loader').classList.add('hidden');
  } catch (error) {
    console.error('Error fetching recommendations:', error);
    // Hide loader
    document.getElementById('loader').classList.add('hidden');
  }
};

const updateChart = (data) => {
  if (!ctx) return;

  const chartData = {
    labels: ['Danceability', 'Energy', 'Valence'],
    datasets: [
      {
        label: 'Features',
        data: [data.danceability, data.energy, data.valence],
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1
      }
    ]
  };

  new Chart(ctx, {
    type: 'radar',
    data: chartData,
    options: {
      scale: {
        ticks: {
          beginAtZero: true
        }
      }
    }
  });

  // Show the chart
  document.getElementById('audioFeaturesChart').classList.remove('hidden');
};

// Event listeners for the sliders
document.getElementById('danceability').addEventListener('input', (event) => {
  document.getElementById('danceability-value').innerText = event.target.value;
});

document.getElementById('energy').addEventListener('input', (event) => {
  document.getElementById('energy-value').innerText = event.target.value;
});

document.getElementById('valence').addEventListener('input', (event) => {
  document.getElementById('valence-value').innerText = event.target.value;
});

// Event listener for the get recommendations button
document.getElementById('getRecommendationsBtn').addEventListener('click', fetchRecommendations);

// Populate the playlist dropdown
const fetchPlaylists = async () => {
  try {
    const response = await fetch('/api/get-playlists');
    const playlists = await response.json();

    const playlistSelect = document.getElementById('playlistSelect');
    playlists.forEach(playlist => {
      const option = document.createElement('option');
      option.value = playlist.id;
      option.textContent = playlist.name;
      playlistSelect.appendChild(option);
    });
  } catch (error) {
    console.error('Error fetching playlists:', error);
  }
};

// Initial data fetch
fetchPlaylists();
        radarChart = new Chart(ctx, {
            type: 'radar',
            data: chartData,
            options: {
                elements: {
                    line: {
                        borderWidth: 3
                    }
                },
                scale: {
                    ticks: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });

        audioFeaturesChart.classList.remove('hidden');
    }

    createPlaylistBtn.addEventListener('click', createPlaylist);

    function createPlaylist() {
        const playlistName = document.getElementById('playlistName').value;
        const trackUris = recommendedTracks.map(track => track.uri);

        fetch('/create-playlist', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                name: playlistName,
                tracks: trackUris
            }),
        })
        .then(response => response.json())
        .then(data => {
            alert(`Playlist "${playlistName}" created successfully!`);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while creating the playlist.');
        });
    }

    // Initialize Spotify Web Playback SDK
    window.onSpotifyWebPlaybackSDKReady = () => {
        fetch('/get-spotify-token')
            .then(response => response.json())
            .then(data => {
                const token = data.token;
                const player = new Spotify.Player({
                    name: 'Song Recommender Web Player',
                    getOAuthToken: cb => { cb(token); }
                });

                // Error handling
                player.addListener('initialization_error', ({ message }) => { console.error(message); });
                player.addListener('authentication_error', ({ message }) => { console.error(message); });
                player.addListener('account_error', ({ message }) => { console.error(message); });
                player.addListener('playback_error', ({ message }) => { console.error(message); });

                // Playback status updates
                player.addListener('player_state_changed', state => { console.log(state); });

                // Ready
                player.addListener('ready', ({ device_id }) => {
                    console.log('Ready with Device ID', device_id);
                });

                // Not Ready
                player.addListener('not_ready', ({ device_id }) => {
                    console.log('Device ID has gone offline', device_id);
                });

                // Connect to the player!
                player.connect();
            })
            .catch(error => {
                console.error('Error getting Spotify token:', error);
            });
    };
});