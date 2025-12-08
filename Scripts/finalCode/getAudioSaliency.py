import numpy as np
import librosa
from scipy.integrate import dblquad

def compute_audio_at_direction(W: np.ndarray, X: np.ndarray, 
                               Y: np.ndarray, Z: np.ndarray,
                               top_left: tuple, bottom_right: tuple,
                               center_time: float, sampleRate: int,
                               window_sec: float = 0.1) -> np.ndarray:
    """
    Compute audio waveform for a tile and around a specific time.
    
    Parameters:
        center_time: time in seconds to center the window
        window_sec: half-length of window (seconds) to extract
    """
    top_left_lat, top_left_lon = top_left
    bottom_right_lat, bottom_right_lon = bottom_right
    
    # Convert bounds to radians
    lat_min_rad = np.radians(bottom_right_lat)
    lat_max_rad = np.radians(top_left_lat)
    lon_min_rad = np.radians(top_left_lon)
    lon_max_rad = np.radians(bottom_right_lon)
    
    # Compute patch area (steradians)
    area = (lon_max_rad - lon_min_rad) * (np.sin(lat_max_rad) - np.sin(lat_min_rad))
    
    # Integrate Y_W over the region
    integral_Y_W, _ = dblquad(lambda lon, lat: np.cos(lat), lat_min_rad, lat_max_rad,
                              lambda _: lon_min_rad, lambda _: lon_max_rad)
    integral_Y_X, _ = dblquad(lambda lon, lat: np.cos(lat)*np.cos(lon)*np.cos(lat), lat_min_rad, lat_max_rad,
                              lambda _: lon_min_rad, lambda _: lon_max_rad)
    integral_Y_Y, _ = dblquad(lambda lon, lat: np.cos(lat)*np.sin(lon)*np.cos(lat), lat_min_rad, lat_max_rad,
                              lambda _: lon_min_rad, lambda _: lon_max_rad)
    integral_Y_Z, _ = dblquad(lambda lon, lat: np.sin(lat)*np.cos(lat), lat_min_rad, lat_max_rad,
                              lambda _: lon_min_rad, lambda _: lon_max_rad)
    
    # Convert time to sample indices
    center_sample = int(center_time * sampleRate)
    half_window_samples = int(window_sec * sampleRate)
    start = max(center_sample - half_window_samples, 0)
    end = min(center_sample + half_window_samples, len(W))
    
    # Extract the waveform slice
    W_slice = W[start:end]
    X_slice = X[start:end]
    Y_slice = Y[start:end]
    Z_slice = Z[start:end]
    
    # Reconstruct waveform for this tile
    wave = (integral_Y_W * W_slice + integral_Y_X * X_slice +
            integral_Y_Y * Y_slice + integral_Y_Z * Z_slice) / area
    
    return wave

def computeHNR(frame):
    """
    Compute HNR for a single frame using autocorrelation.
    HNR = 10 * log10(energy_harmonic / energy_noise)
    """
    frame = frame - np.mean(frame)  # remove DC
    if np.all(frame == 0):
        return 0.0

    # FFT-based autocorrelation
    autocorr = np.fft.irfft(np.fft.rfft(frame) * np.conj(np.fft.rfft(frame)))
    autocorr = autocorr / np.max(np.abs(autocorr))  # normalize

    # Harmonic energy = max autocorr (excluding lag 0)
    harmonic_energy = np.max(autocorr[1:])
    # Noise energy = lag 0 minus harmonic energy
    noise_energy = autocorr[0] - harmonic_energy
    if noise_energy <= 0:
        return 40.0  # cap to a reasonable max
    return 10 * np.log10(harmonic_energy / noise_energy)

def processWave(wave, sampleRate):
    windowSize = 2048
    hopSize = 100
    
    # converts this to a Short-Time Fourier Transform. Tells you how much eergy has at each frequency over time.
    # does this by going through windows. Length of each window defined by n_fft. Then, shifts window to right by length
    # hop length. at each window, computes how much of each frequency is present.
    # final value is 2D array of rows being each frequency, columns being time (which is now the windows), so value being amplitude/energy for that time and frequency
    stftWave = np.abs(librosa.stft(wave, n_fft=windowSize, hop_length=hopSize))
    # when we get the mel, that just converts all the frequencies to 128 possible onces, which are moreso frequencies humans can hear. So compressing
    # the frequencies from a large number of frequencies to a smaller number, in this case n_mels amount
    mel = librosa.feature.melspectrogram(S = stftWave, sr= sampleRate, n_mels = 128)
    # converts from power scaling of audio to decibel scaling, cause humans perceive in moreso logarithm of audio (so higher sounds kinda taper off to us)
    logMel = librosa.power_to_db(mel, ref=np.max)

    # gets overall frame energy, including amplitude
    volumeNorm = np.mean((logMel + 80), axis=0)
    # gets the contrast in energy between frequencies within a specific frequency band, so where some frequencies bands may have parts of high energy frequencies, while other parts are low energy
    contrast = librosa.feature.spectral_contrast(S = stftWave, sr=sampleRate)
    # combines the difference frequency bands to get a average contrast for that time frame
    contrast = np.mean(contrast, axis=0)
    # basically gets how much the sound chagnes over time. Does this by getting differnece over time fimes with np.diff, squaring that value, and getting its sum
    temporal_novelty = np.sum(np.diff(logMel, axis=1) ** 2, axis=0)
    # do this to add an extra value cause rn, the length is T - 1, since you're getting difference between frames. So add 1 to get it to T length
    temporal_novelty = np.insert(temporal_novelty, 0, 0)

    # this gets how noise like a sound is, whether it's tonal or liek white noise. A tonal sound is one that just
    # stands out, like through sharp peaks.
    spectral_flatness = librosa.feature.spectral_flatness(y=wave, n_fft=windowSize, hop_length= hopSize)
    # gets the average frequency weighted by amplitude, how "bright" the sound is, sees if the audio tends to have more high frequency or low frequency sounds
    centroid = librosa.feature.spectral_centroid(S=stftWave, sr=sampleRate)
    # indicates the range of frequencies present, so if the frequencies are more concentrated or spread out
    bandwidth = librosa.feature.spectral_bandwidth(S=stftWave, sr=sampleRate)



    # Compute HNR per frame. HNR is how harmonic the sound is, if its harmonic, with a pattern, or more noisy.
    # different from spectral flatness in that it measures if its harmonic, as opposed to tonal. Basically if there's like
    # a repeating pattern that identifies the town
    hnr_values = []
    num_frames = stftWave.shape[1]
    for i in range(num_frames):
        frame = wave[i*hopSize : i*hopSize + windowSize]
        if len(frame) < 2:
            continue
        hnr_values.append(computeHNR(frame))
    hnr_values = np.array(hnr_values)
    
    return np.array([np.mean(volumeNorm), np.mean(contrast), np.mean(temporal_novelty), np.mean(hnr_values), np.mean(spectral_flatness), np.mean(centroid), np.mean(bandwidth)])

def precompute_integrals(tile_size_deg=20):
    """
    Precompute integrals for coarse tiles (done once, cached for all frames).
    
    Parameters:
        tile_size_deg: size of each tile in degrees (default 20x20)
    
    Returns:
        Dictionary mapping (lat, lon) tile coordinates to integral values and area
    """
    print(f"Precomputing integrals for {tile_size_deg}° tiles...")
    
    tile_cache = {}
    
    # Generate all tile coordinates
    latitudes = list(range(90, -90, -tile_size_deg))
    longitudes = list(range(-180, 180, tile_size_deg))
        
    for topLeftLat in latitudes:
        for topLeftLon in longitudes:            
            bottom_right_lat, bottom_right_lon = (topLeftLat - tile_size_deg, topLeftLon + tile_size_deg)
            
            # Convert bounds to radians
            lat_min_rad = np.radians(bottom_right_lat)
            lat_max_rad = np.radians(topLeftLat)
            lon_min_rad = np.radians(topLeftLon)
            lon_max_rad = np.radians(bottom_right_lon)
            
            # Compute patch area
            area = (lon_max_rad - lon_min_rad) * (np.sin(lat_max_rad) - np.sin(lat_min_rad))
            
            # Compute integrals
            integral_Y_W, _ = dblquad(
                lambda lon, lat: np.cos(lat), lat_min_rad, lat_max_rad,
                lambda _: lon_min_rad, lambda _: lon_max_rad)
            integral_Y_X, _ = dblquad(
                lambda lon, lat: np.cos(lat)*np.cos(lon)*np.cos(lat), lat_min_rad, lat_max_rad,
                lambda _: lon_min_rad, lambda _: lon_max_rad)
            integral_Y_Y, _ = dblquad(
                lambda lon, lat: np.cos(lat)*np.sin(lon)*np.cos(lat), lat_min_rad, lat_max_rad,
                lambda _: lon_min_rad, lambda _: lon_max_rad)
            integral_Y_Z, _ = dblquad(
                lambda lon, lat: np.sin(lat)*np.cos(lat), lat_min_rad, lat_max_rad,
                lambda _: lon_min_rad, lambda _: lon_max_rad)
            
            tile_cache[(topLeftLat, topLeftLon)] = (
                area,
                integral_Y_W,
                integral_Y_X,
                integral_Y_Y,
                integral_Y_Z
            )
    
    print("Integral precomputation complete!")
    return tile_cache


def get_tile_for_pixel(lat_pixel, lon_pixel, erp_height, erp_width, tile_size_deg=20):
    """
    Get the tile coordinates for a given pixel in the ERP map.
    """
    # Convert pixel to lat/lon
    lat = 90 - (lat_pixel / erp_height) * 180
    lon = -180 + (lon_pixel / erp_width) * 360
    
    # Find which tile this belongs to
    tile_lat = int(np.floor(lat / tile_size_deg)) * tile_size_deg
    tile_lon = int(np.floor(lon / tile_size_deg)) * tile_size_deg
    
    return (tile_lat, tile_lon)


def compute_audio_saliency_heatmap_vectorized(W, X, Y, Z, audio_samplerate, 
                                              frame_idx, video_fps, 
                                              erp_height, erp_width, 
                                              tile_cache,  sample_every_n_frames, 
                                              numHeatmaps, tile_size_deg=20):
    """
    Compute audio saliency heatmap for a given frame using precomputed tile integrals.
    Each tile's saliency is computed once and replicated to all pixels in that tile.
    Extracts audio from 2.5 frames before to 2.5 frames after the current frame.
    Returns a 2D array of shape (erp_height, erp_width).
    """
    # Time in seconds corresponding to this video frame
    time_sec = frame_idx / video_fps
    
    # Window: 2.5 frames before and 2.5 frames after = 5 frames total
    frameWindow = sample_every_n_frames / 2
    window_sec = frameWindow / video_fps
    
    # Convert time to sample indices
    center_sample = int(time_sec * audio_samplerate)
    half_window_samples = int(window_sec * audio_samplerate)
    start = max(center_sample - half_window_samples, 0)
    end = min(center_sample + half_window_samples, len(W))
    
    # Extract waveform slices
    W_slice = W[start:end]
    X_slice = X[start:end]
    Y_slice = Y[start:end]
    Z_slice = Z[start:end]
    
    # Initialize output saliency map
    saliency_map = np.zeros((numHeatmaps, erp_height, erp_width))
    
    numLatTiles = 180 // tile_size_deg
    numLonTiles = 360 // tile_size_deg   

    # Calculate pixels per tile
    pixels_per_tile_lat = erp_height // numLatTiles  # 9 tiles in latitude
    pixels_per_tile_lon = erp_width // numLonTiles   # 18 tiles in longitude
    
    # Iterate through each tile
    for lat_tile in range(numLatTiles):
        for lon_tile in range(numLonTiles):
            # Get tile coordinates
            tile_lat = 90 - lat_tile * tile_size_deg
            tile_lon = -180 + lon_tile * tile_size_deg
            tile_coords = (tile_lat, tile_lon)
            
            if tile_coords in tile_cache:
                area, integral_Y_W, integral_Y_X, integral_Y_Y, integral_Y_Z = tile_cache[tile_coords]
                
                # Reconstruct waveform for this tile
                wave = (integral_Y_W * W_slice + 
                       integral_Y_X * X_slice +
                       integral_Y_Y * Y_slice + 
                       integral_Y_Z * Z_slice) / area
                
                if len(wave) > 0:
                    saliency_values = processWave(wave, audio_samplerate)
                else:
                    saliency_values = np.zeros(numHeatmaps)
                
                # Fill all pixels in this tile with the same saliency value
                y_start = lat_tile * pixels_per_tile_lat
                y_end = (lat_tile + 1) * pixels_per_tile_lat
                x_start = lon_tile * pixels_per_tile_lon
                x_end = (lon_tile + 1) * pixels_per_tile_lon
                
                # weirdly works cause of python's funky mapping
                saliency_map[:, y_start:y_end, x_start:x_end] = saliency_values[:, np.newaxis, np.newaxis]
            else:
                raise IndexError("IDk how we got this.")
    
    return saliency_map