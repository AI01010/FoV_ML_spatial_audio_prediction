import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import time
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

class SaliencyTileDataset(Dataset):
    """Memory-efficient dataset for saliency heatmaps"""

    def __init__(self, heatmaps_set, tile_indices_set):
        """
        Args:
            heatmaps: numpy array of shape (N_frames, NUM_HEATMAPS, H, W)
            tile_indices: numpy array of shape (N_frames,) with tile indices [0-143]
        """
        self.heatmaps_set = heatmaps_set
        self.tile_indices_set = tile_indices_set

    def __len__(self):
        totalLength = 0
        for set in self.tile_indices_set:
            totalLength += len(set)
        return totalLength

    def __getitem__(self, idx):
        for i in range(len(self.tile_indices_set)):
            if(idx < len(self.tile_indices_set[i])):
                heatmap = torch.from_numpy(self.heatmaps_set[i][idx].copy()).float()
                tile_idx = torch.tensor(self.tile_indices_set[i][idx], dtype=torch.long)
                return heatmap, tile_idx
            idx -= len(self.tile_indices_set[i])
        print("THIS SHOULD NEVER PRINT")
        return None, None
    
class TileCoordinateLoss(nn.Module):
    """
    Euclidean distance loss in tile coordinate space
    Instead of treating tiles as classes, measure spatial distance
    """
    def __init__(self, tiles_x=16, tiles_y=9):
        super(TileCoordinateLoss, self).__init__()

        self.tiles_x = tiles_x
        self.tiles_y = tiles_y

        # Coordinates of every class (constant)
        predictedXCoords = torch.arange(tiles_x * tiles_y, device=device) % tiles_x
        predictedYCoords = torch.arange(tiles_x * tiles_y, device=device) // tiles_x

        predictedXCoords = predictedXCoords.unsqueeze(0)        # (1, C)
        predictedYCoords = predictedYCoords.unsqueeze(0)        # (1, C)

        self.predictedXCoords = predictedXCoords
        self.predictedYCoords = predictedYCoords

    def forward(self, predictedTilesBatch, actualTilesBatch):
        vectorizedProbabilities = F.softmax(predictedTilesBatch, dim = 1)

        # Coordinates of true tile per batch element
        true_x = (actualTilesBatch % self.tiles_x).unsqueeze(1)              # (B, 1)
        true_y = (actualTilesBatch // self.tiles_x).unsqueeze(1)             # (B, 1)

        dx = torch.abs(true_x - self.predictedXCoords)
        dx = torch.minimum(dx, self.tiles_x - dx)

        dy = true_y - self.predictedYCoords

        vectorizedDistances = torch.sqrt(dx**2 + dy**2)

        finalLossPerBatchPerClass = vectorizedDistances * vectorizedProbabilities

        summatedLoss = torch.sum(finalLossPerBatchPerClass, dim = 1)

        return summatedLoss.mean()
        
class HybridTileLoss(nn.Module):
    """
    Combined loss: CrossEntropy for classification + Euclidean for spatial awareness
    """
    def __init__(self, tiles_x=16, tiles_y=9, ce_weight=0.3, coord_weight=0.7):
        super(HybridTileLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.coord_loss = TileCoordinateLoss(tiles_x, tiles_y)
        self.ce_weight = ce_weight
        self.coord_weight = coord_weight

    

    def forward(self, logits, target_indices):
        """
        Args:
            logits: (batch_size, num_tiles) - model outputs
            target_indices: (batch_size,) - ground truth tile indices

        Returns:
            total_loss: weighted combination of CE and coordinate loss
            ce_loss: cross entropy component
            coord_loss: coordinate distance component
        """
        ce = self.ce_loss(logits, target_indices)
        coord = self.coord_loss(logits, target_indices)

        total = self.ce_weight * ce + self.coord_weight * coord

        # return total
        return ce

class HeatmapFusionCNN(nn.Module):
    """Lightweight CNN for fusing saliency heatmaps"""

    def __init__(self, num_heatmaps=9, numCols = 16, numRows = 9, theDevice = "cuda", dropout=0.3):
        super(HeatmapFusionCNN, self).__init__()

        self.numCols = numCols
        self.numRows = numRows
        self.numTiles = numCols * numRows
        self.device = theDevice
        self.criterion = HybridTileLoss(tiles_x=self.numCols, tiles_y=self.numRows,
                                   ce_weight=0.3, coord_weight=0.7).to(self.device)

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
        self.fc2 = nn.Linear(256, self.numTiles)

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
    
    def tile_coords_to_index(self, x, y):
        """Convert tile coordinates to linear index"""
        return y * self.numCols + x


    def tile_index_to_coords(self, idx):
        """Convert linear index to tile coordinates"""
        y = idx // self.numCols
        x = idx % self.numCols
        return x, y
        
    def train_epoch(self, dataloader, optimizer):
        """Train for one epoch"""
        self.train()
        total_loss = 0
        correct = 0
        total = 0


        batchNum = 0

        for heatmaps, tile_indices in dataloader:
            start_time = time.time()
            heatmaps = heatmaps.to(self.device)
            tile_indices = tile_indices.to(self.device)

            optimizer.zero_grad()
            outputs = self(heatmaps)
            loss = self.criterion(outputs, tile_indices)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += tile_indices.size(0)
            correct += predicted.eq(tile_indices).sum().item()

            epoch_time = time.time() - start_time
            print(f"Batch {batchNum+1}/{len(dataloader)}: {epoch_time:.1f}s")
            batchNum += 1


        return total_loss / len(dataloader), 100. * correct / total
    
    def tile_distance(self, pred_idx, true_idx):
        """Calculate tile distance"""
        px, py = self.tile_index_to_coords(pred_idx)
        tx, ty = self.tile_index_to_coords(true_idx)
        return np.sqrt((px - tx)**2 + (py - ty)**2)

    def validate(self, dataloader):
        """Validate the model"""
        self.eval()
        total_loss = 0
        correct = 0
        total = 0
        tile_distances = []

        with torch.no_grad():
            for heatmaps, tile_indices in dataloader:
                heatmaps = heatmaps.to(self.device)
                tile_indices = tile_indices.to(self.device)

                outputs = self(heatmaps)
                loss = self.criterion(outputs, tile_indices)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += tile_indices.size(0)
                correct += predicted.eq(tile_indices).sum().item()

                probs = torch.softmax(outputs, dim=1)  # softmax along the class dimension
                print(f"Predicted stuff is {probs}, actual is {tile_indices}")
                

                for pred, true in zip(predicted.cpu().numpy(), tile_indices.cpu().numpy()):
                    tile_distances.append(self.tile_distance(pred, true))

        avg_distance = np.mean(tile_distances)
        return total_loss / len(dataloader), 100. * correct / total, avg_distance


    def trainModel(self, train_loader, val_loader):
        optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

        # Training
        num_epochs = 40
        results = []

        print("Training started...\n")
        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()

            train_loss, train_acc = self.train_epoch(train_loader, optimizer)
            val_loss, val_acc, avg_tile_dist = self.validate(val_loader)

            scheduler.step(val_loss)
            epoch_time = time.time() - epoch_start

            results.append([
                epoch + 1,
                f"{train_loss:.4f}",
                f"{train_acc:.1f}%",
                f"{val_loss:.4f}",
                f"{val_acc:.1f}%",
                f"{avg_tile_dist:.2f}",
                f"{epoch_time:.1f}s"
            ])

            print(f"Epoch {epoch+1}/{num_epochs}: "
                    f"Train {train_acc:.1f}% | Val {val_acc:.1f}% | "
                    f"Dist {avg_tile_dist:.2f} | {epoch_time:.1f}s")

            # Clear cache periodically
            if self.device == "cuda":
                torch.cuda.empty_cache()

        total_time = time.time() - start_time
        print(f"Total time was: {total_time}")

        # torch.save(self.state_dict(), 'best_model.pth')