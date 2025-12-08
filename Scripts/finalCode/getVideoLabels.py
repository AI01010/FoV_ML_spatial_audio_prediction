import pandas as pd
def filterDf(csv_path, participant_id, video_id):
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    # Filter for specific participant and video
    filtered_df = df[(df['id'] == participant_id) & (df['video'] == video_id)]

    if len(filtered_df) == 0:
        raise ValueError(f"No data found for participant {participant_id} and video {video_id}")

    # Check stereo mode
    stereo_mode = filtered_df['stereo'].iloc[0]

    # Sort by timestamp to ensure correct ordering
    filtered_df = filtered_df.sort_values('t').reset_index(drop=True)
    return filterDf

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
