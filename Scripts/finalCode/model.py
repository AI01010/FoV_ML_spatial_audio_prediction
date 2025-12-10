import torch, torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

class SaliencyTileDataset(Dataset):
    """Memory-efficient dataset for saliency heatmaps"""

    def __init__(self, heatmaps, tile_indices):
        """
        Args:
            heatmaps: numpy array of shape (N_frames, NUM_HEATMAPS, H, W)
            tile_indices: numpy array of shape (N_frames,) with tile indices [0-143]
        """
        self.tile_indices = tile_indices.astype(np.int64)

    def __len__(self):
        totalLength = 0
        for()
        return len(self.tile_indices)

    def __getitem__(self, idx):
        heatmap = torch.from_numpy(self.heatmaps[idx])
        tile_idx = torch.tensor(self.tile_indices[idx], dtype=torch.long)
        return heatmap, tile_idx


class HeatmapFusionCNN(nn.Module):
    """Lightweight CNN for fusing saliency heatmaps"""

    def __init__(self, num_heatmaps=9, num_tiles=144, dropout=0.3):
        super(HeatmapFusionCNN, self).__init__()

        # Heatmap fusion with 1x1 conv
        self.fusion = nn.Conv2d(num_heatmaps, 8, kernel_size=1)

        # Lightweight feature extraction
        self.conv1 = nn.Conv2d(8, 32, kernel_size=5, stride=4, padding=2)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Adaptive pooling
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # Classifier
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_tiles)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Fuse heatmaps
        x = self.fusion(x)
        x = self.relu(x)

        # Feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        # Pool and classify
        x = self.pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def tile_coords_to_index(x, y, tiles_x):
    """Convert tile coordinates to linear index"""
    return y * tiles_x + x


def tile_index_to_coords(idx, tiles_x):
    """Convert linear index to tile coordinates"""
    y = idx // tiles_x
    x = idx % tiles_x
    return x, y


def tile_distance(pred_idx, true_idx, tiles_x):
    """Calculate tile distance"""
    px, py = tile_index_to_coords(pred_idx, tiles_x)
    tx, ty = tile_index_to_coords(true_idx, tiles_x)
    return np.sqrt((px - tx)**2 + (py - ty)**2)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for heatmaps, tile_indices in dataloader:
        heatmaps = heatmaps.to(device)
        tile_indices = tile_indices.to(device)

        optimizer.zero_grad()
        outputs = model(heatmaps)
        loss = criterion(outputs, tile_indices)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += tile_indices.size(0)
        correct += predicted.eq(tile_indices).sum().item()

    return total_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    tile_distances = []

    with torch.no_grad():
        for heatmaps, tile_indices in dataloader:
            heatmaps = heatmaps.to(device)
            tile_indices = tile_indices.to(device)

            outputs = model(heatmaps)
            loss = criterion(outputs, tile_indices)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += tile_indices.size(0)
            correct += predicted.eq(tile_indices).sum().item()

            for pred, true in zip(predicted.cpu().numpy(), tile_indices.cpu().numpy()):
                tile_distances.append(tile_distance(pred, true))

    avg_distance = np.mean(tile_distances)
    return total_loss / len(dataloader), 100. * correct / total, avg_distance

def flatten_video_data(heatmaps_4d, tile_indices_1d, numHeatmaps, frameHeight, frameWidth):
    """
    Flatten video data from (videos, frames, ...) to (total_frames, ...)

    Args:
        heatmaps_4d: shape (num_videos, frames_per_video, num_heatmaps, H, W)
        tile_indices_1d: shape (num_videos, frames_per_video)

    Returns:
        heatmaps: shape (total_frames, num_heatmaps, H, W)
        tile_indices: shape (total_frames,)
    """
    num_videos, frames_per_video = tile_indices_1d.shape
    total_frames = num_videos * frames_per_video

    # Reshape heatmaps: (videos, frames, heatmaps, H, W) -> (total_frames, heatmaps, H, W)
    heatmaps = heatmaps_4d.reshape(total_frames, numHeatmaps, frameHeight, frameWidth)

    # Flatten tile indices: (videos, frames) -> (total_frames,)
    tile_indices = tile_indices_1d.reshape(total_frames)

    return heatmaps, tile_indices