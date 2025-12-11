import numpy as np
import cv2

import sys
import os

# Path to the U-2-Net-Repo folder relative to cwd
repo_path = os.path.join(os.getcwd(), "../../U-2-Net-Repo")  # e.g., c:\Users\mahd\Documents\FOV Prediction\U-2-Net-Repo

# Append it to sys.path
sys.path.append(repo_path)

print(f"Repo path was: {repo_path}")

from model.u2net import U2NETP
import torch
from torchvision import transforms
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

MODEL_PATH = "../../ModelFiles/u2netp.pth"

# Then load model as before
u2netp = U2NETP(3, 1).to(device)
u2netp.load_state_dict(torch.load(MODEL_PATH, map_location=device))
u2netp.eval()
print("U2 Net Model ready!")
""
transform_u2 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((320, 320)),
    transforms.ToTensor()
])

# Path to the U-2-Net-Repo folder relative to cwd
repo_path = os.path.join(os.getcwd(), "../../Hani-Raft-Repo/core")  # e.g., c:\Users\mahd\Documents\FOV Prediction\U-2-Net-Repo

# Append it to sys.path
sys.path.append(repo_path)

import argparse
from matplotlib import cm
try:
    from raft import RAFT
    from utils.utils import InputPadder
    print("Successfully imported core RAFT modules from your fork.")
except ImportError as e:
    print(f"FATAL ERROR: Could not import RAFT modules. Check your fork structure: {e}")
    sys.exit(1)


# ---------------------------
# Load CPU-only RAFT model
# ---------------------------
# These arguments trigger the 'small' model and disable CUDA-specific features.
# The custom logic in your raft.py will ensure the CPUCostVolume is used here.
args = argparse.Namespace(
    small=True,
    mixed_precision=False,
    alternate_corr=False, # Set to False to use the primary CorrBlock path (which you patched)
    dropout=0,
    dropout2=0,
)

weights_path = "../../ModelFiles/raft-small.pth"

print(f"Loading RAFT {device} model...")
# Initialize model on device
model = RAFT(args).to(device)

# Clean state_dict and load weights
checkpoint = torch.load(weights_path, map_location=device)
state_dict = checkpoint.get("state_dict", checkpoint)
clean_state = {}
for k, v in state_dict.items():
    new_k = k.replace("module.", "")
    if new_k in model.state_dict():
        clean_state[new_k] = v

model.load_state_dict(clean_state, strict=False)
model.eval()
print(f"RAFT model loaded on {device} successfully.")



def compute_flow(img1, img2, model):
    """Computes optical flow using the loaded RAFT model across various scales."""
    import matplotlib.pyplot as plt

    original_h, original_w = img1.shape[:2]
    # Use smaller scales for better CPU stability and speed
    scales = [1.0, 0.75, 0.5, 0.33, 0.25, 0.1, .01]

    for s in scales:
        try:
            print(f"\nTrying scale {s}...")
            # Ensure dimensions are at least 32x32 and multiples of 8 (padding will handle multiples of 8)
            new_h, new_w = max(32, int(original_h * s)), max(32, int(original_w * s))
            im1 = cv2.resize(img1, (new_w, new_h))
            im2 = cv2.resize(img2, (new_w, new_h))

            # Convert to tensor and normalize [0, 1]
            t1 = torch.from_numpy(im1/255.).permute(2,0,1).float().unsqueeze(0).to(device)
            t2 = torch.from_numpy(im2/255.).permute(2,0,1).float().unsqueeze(0).to(device)

            # Pad to multiples of 8 (required by RAFT architecture)
            padder = InputPadder(t1.shape)
            t1, t2 = padder.pad(t1, t2)

            print(f"Padded tensor shapes: t1={t1.shape}, t2={t2.shape}")

            # Run RAFT
            with torch.no_grad():
                # Setting iters=12 is faster for inference than the default 32
                _, flow_up = model(t1, t2, iters=12, test_mode=True)

            # Post-process flow: Resize back to original resolution (H, W)
            flow = flow_up[0].permute(1,2,0).cpu().numpy()

            # The flow values need to be scaled correctly after resizing
            flow_x = cv2.resize(flow[...,0], (original_w, original_h), interpolation=cv2.INTER_LINEAR) * (original_w / flow.shape[1])
            flow_y = cv2.resize(flow[...,1], (original_w, original_h), interpolation=cv2.INTER_LINEAR) * (original_h / flow.shape[0])
            flow_final = np.stack([flow_x, flow_y], axis=-1)

            print(f"✓ Flow computed successfully at scale {s}")
            return flow_final

        except Exception as e:
            print(f"Failed at scale {s} with error: {e}")

    raise RuntimeError("Flow failed at all scales")

def flow_to_magnitude_map(flow):
    """
    Computes only the magnitude (speed) of the optical flow.
    Returns a normalized single-channel array.
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    rad = np.sqrt(u**2 + v**2)
    flow_max = np.max(rad)
    if flow_max > 0:
        rad /= flow_max

    return rad.astype(np.float32)

def generate_flow_heatmap(flow):
    """
    Returns the heatmap image (RGB uint8 array) for the flow magnitude.
    Does NOT display the heatmap.
    """
    magnitude_map = flow_to_magnitude_map(flow)

    # Apply 'jet' colormap using matplotlib
    colormap = cm.get_cmap('jet')
    heatmap = colormap(magnitude_map)  # RGBA float32 in [0,1]

    # Convert RGBA float [0,1] → RGB uint8 [0,255]
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)

    return heatmap


def process_image(file):
    erp = cv2.cvtColor(file, cv2.COLOR_BGR2RGB)
    H, W, _ = erp.shape

    # STEP 4: Run U²-NetP (Global)
    input_full = transform_u2(erp).unsqueeze(0).to(device)
    with torch.no_grad():
        d1, *_ = u2netp(input_full)
        pred_full = F.interpolate(d1, size=(H, W), mode="bilinear", align_corners=False)
        saliency_full = pred_full.squeeze().cpu().numpy()

    saliency_full_resized = (saliency_full - saliency_full.min()) / (saliency_full.max() - saliency_full.min() + 1e-8)

    return saliency_full_resized


def compute_video_saliency_heatmap_vectorized(prevFrame, currentFrame, frame_idx, video_fps,
                                              erp_height, erp_width,
                                              tile_cache,  sample_every_n_frames,
                                              numHeatmaps, tile_size_deg=20):
    """
    Compute video saliency heatmap for a given frame using ERP tile.
    Saliency is computed once on full ERP per frame.
    For motion/optical flow saliency use the current and previous frame (if exists)
    Returns a 2D array of shape (erp_height, erp_width).
    """
    # Time in seconds corresponding to this video frame

    # Initialize output saliency map
    saliency_map = np.zeros((numHeatmaps, erp_height, erp_width))

    # Example usage SalNet:
    saliency_map[0] = process_image(currentFrame)

    # Example usage FlowNet:
    flow = compute_flow(prevFrame, currentFrame, model)
    magnitude = flow_to_magnitude_map(flow)
    saliency_map[1] = magnitude

    return saliency_map
