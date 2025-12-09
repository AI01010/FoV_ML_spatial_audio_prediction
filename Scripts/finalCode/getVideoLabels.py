import numpy as np
import pandas as pd
def filterDf(csv_path, participant_id, video_id):
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    # Filter for specific participant and video
    filtered_df = df[df['video'] == int(video_id)]

    if len(filtered_df) == 0:
        raise ValueError(f"No data found for participant {participant_id} and video {video_id}")

    # Sort by timestamp to ensure correct ordering
    filtered_df = filtered_df.sort_values('t').reset_index(drop=True)
    all_participants = filtered_df['id'].unique()
    return (filtered_df, all_participants)

# Function to convert (u, v) coordinates to tile index
def uv_to_tile_index(u, v, rows, cols):
    """Convert normalized coordinates (u, v) to tile index"""
    col = int(u * cols)
    row = int(v * rows)

    # Clamp to valid range
    col = min(col, cols - 1)
    row = min(row, rows - 1)

    # Calculate tile index (row-major order)
    tile_index = row * cols + col
    return tile_index

def getModeTileIndex(targetTime, labelDf, participants, rows, cols):
    participant_tiles = []

    # Get tile index for each participant at this timestamp
    for pid in participants:
        participant_data = labelDf[labelDf['id'] == pid]

        if len(participant_data) > 0:
            # Find closest timestamp for this participant
            idx = (participant_data['t'] - targetTime).abs().idxmin()
            u = participant_data.loc[idx, 'u']
            v = participant_data.loc[idx, 'v']
            tile_idx = uv_to_tile_index(u, v, rows, cols)
            participant_tiles.append(tile_idx)

    # Calculate mode (most common tile index)
    if len(participant_tiles) > 0:
        # Use bincount to find mode efficiently
        counts = np.bincount(participant_tiles)
        mode_tile = np.argmax(counts)
        return mode_tile
    return 0
