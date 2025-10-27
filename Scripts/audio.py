import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

class EquirectangularAudioMapper:
    """
    Maps equirectangular 360° video to spherical coordinates and computes
    audio energy at each pixel location using ambisonic audio.
    
    This aligns the audio spatial information with the video pixels.
    """
    
    def __init__(self, sample_rate: int = 48000, frame_duration_ms: float = 33.33):
        """
        Initialize the mapper.
        
        Args:
            sample_rate: Audio sample rate in Hz
            frame_duration_ms: Duration of each frame in milliseconds
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
    
    def erp_to_spherical(self, u: np.ndarray, v: np.ndarray, 
                        W: int, H: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert equirectangular pixel coordinates to spherical coordinates.
        
        Args:
            u: Horizontal pixel coordinates (can be array)
            v: Vertical pixel coordinates (can be array)
            W: Width of equirectangular image
            H: Height of equirectangular image
            
        Returns:
            lambda_: Longitude in radians [-π, π] (azimuth)
            phi: Latitude in radians [-π/2, π/2] (elevation)
        """
        # Convert pixel coordinates to spherical angles
        lambda_ = 2 * np.pi * (u / W - 0.5)  # Longitude (azimuth)
        phi = np.pi * (0.5 - v / H)           # Latitude (elevation)
        
        return lambda_, phi
    
    def spherical_to_cartesian(self, lambda_: np.ndarray, 
                               phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert spherical coordinates to 3D Cartesian coordinates on unit sphere.
        
        Args:
            lambda_: Longitude (azimuth) in radians
            phi: Latitude (elevation) in radians
            
        Returns:
            x, y, z: Cartesian coordinates on unit sphere
        """
        x = np.cos(phi) * np.cos(lambda_)
        y = np.sin(phi)
        z = np.cos(phi) * np.sin(lambda_)
        
        return x, y, z
    
    def compute_audio_at_direction(self, W: np.ndarray, X: np.ndarray, 
                                   Y: np.ndarray, Z: np.ndarray,
                                   theta: float, phi: float) -> float:
        """
        Compute audio waveform/energy at a specific spherical direction.
        Uses the spherical harmonic reconstruction formula:
        s(θ, φ, t) = W(t)Y_W(θ,φ) + X(t)Y_X(θ,φ) + Y(t)Y_Y(θ,φ) + Z(t)Y_Z(θ,φ)
        
        Args:
            W, X, Y, Z: Ambisonic channel waveforms for this time frame
            theta: Azimuth angle (longitude) in radians
            phi: Elevation angle (latitude) in radians
            
        Returns:
            energy: Audio energy at this direction
        """
        # Spherical harmonic basis functions (First-Order Ambisonics)
        # Using ACN/SN3D convention
        Y_W = 1.0                                    # Y_0^0 (omnidirectional)
        Y_X = np.cos(phi) * np.cos(theta)           # Y_1^1 (front-back)
        Y_Y = np.cos(phi) * np.sin(theta)           # Y_1^-1 (left-right)
        Y_Z = np.sin(phi)                            # Y_1^0 (up-down)
        
        # Reconstruct audio signal at this direction
        signal = W * Y_W + X * Y_X + Y * Y_Y + Z * Y_Z
        
        # Compute energy (RMS)
        energy = np.sqrt(np.mean(signal ** 2))
        
        return energy
    
    def create_audio_energy_map(self, ambisonics: np.ndarray, 
                                frame_idx: int,
                                video_width: int, 
                                video_height: int,
                                sample_every: int = 5) -> np.ndarray:
        """
        Create an audio energy map aligned with equirectangular video frame.
        
        Args:
            ambisonics: Array of shape [samples, 4] (W, X, Y, Z channels)
            frame_idx: Which video frame to process
            video_width: Width of the equirectangular video
            video_height: Height of the equirectangular video
            sample_every: Sample every N degrees (e.g., 5 means 5°×5° grid)
                         Lower = higher resolution but slower
            
        Returns:
            energy_map: Array of shape [H//sample_every, W//sample_every] 
                       containing audio energy at each sampled pixel location
        """
        # Extract audio frame
        start = frame_idx * self.frame_size
        end = start + self.frame_size
        
        if end > ambisonics.shape[0]:
            raise ValueError(f"Frame {frame_idx} exceeds audio length")
        
        W_audio = ambisonics[start:end, 0]
        X_audio = ambisonics[start:end, 1]
        Y_audio = ambisonics[start:end, 2]
        Z_audio = ambisonics[start:end, 3]
        
        # Sample every N degrees
        u_samples = np.arange(0, video_width, sample_every)
        v_samples = np.arange(0, video_height, sample_every)
        
        # Create output energy map
        out_height = len(v_samples)
        out_width = len(u_samples)
        energy_map = np.zeros((out_height, out_width))
        
        # For each sampled pixel location
        for i, v in enumerate(v_samples):
            for j, u in enumerate(u_samples):
                # Convert pixel to spherical coordinates
                lambda_, phi = self.erp_to_spherical(u, v, video_width, video_height)
                
                # Compute audio energy at this direction
                energy = self.compute_audio_at_direction(
                    W_audio, X_audio, Y_audio, Z_audio,
                    theta=lambda_, phi=phi
                )
                
                energy_map[i, j] = energy
        
        return energy_map
    
    def create_full_resolution_audio_map(self, ambisonics: np.ndarray,
                                         frame_idx: int,
                                         video_width: int,
                                         video_height: int) -> np.ndarray:
        """
        Create audio energy map at full video resolution.
        WARNING: This can be slow for high-resolution videos!
        
        Args:
            ambisonics: Array of shape [samples, 4] (W, X, Y, Z channels)
            frame_idx: Which video frame to process
            video_width: Width of the equirectangular video
            video_height: Height of the equirectangular video
            
        Returns:
            energy_map: Array of shape [video_height, video_width]
        """
        # Extract audio frame
        start = frame_idx * self.frame_size
        end = start + self.frame_size
        
        W_audio = ambisonics[start:end, 0]
        X_audio = ambisonics[start:end, 1]
        Y_audio = ambisonics[start:end, 2]
        Z_audio = ambisonics[start:end, 3]
        
        # Pre-compute all pixel coordinates
        u_coords, v_coords = np.meshgrid(
            np.arange(video_width), 
            np.arange(video_height)
        )
        
        # Convert all pixels to spherical at once
        lambda_all, phi_all = self.erp_to_spherical(
            u_coords, v_coords, video_width, video_height
        )
        
        # Compute spherical harmonics for all directions
        Y_W = 1.0
        Y_X = np.cos(phi_all) * np.cos(lambda_all)
        Y_Y = np.cos(phi_all) * np.sin(lambda_all)
        Y_Z = np.sin(phi_all)
        
        # Reconstruct audio at all directions
        # Use mean of audio channels for vectorized computation
        W_mean = np.mean(W_audio)
        X_mean = np.mean(X_audio)
        Y_mean = np.mean(Y_audio)
        Z_mean = np.mean(Z_audio)
        
        energy_map = np.sqrt(
            (W_mean * Y_W) ** 2 + 
            (X_mean * Y_X) ** 2 + 
            (Y_mean * Y_Y) ** 2 + 
            (Z_mean * Y_Z) ** 2
        )
        
        return energy_map
    
    def create_audio_map_sequence(self, ambisonics: np.ndarray,
                                 video_width: int,
                                 video_height: int,
                                 sample_every: int = 5) -> np.ndarray:
        """
        Create audio energy maps for all frames in the video.
        
        Args:
            ambisonics: Array of shape [samples, 4] (W, X, Y, Z channels)
            video_width: Width of equirectangular video
            video_height: Height of equirectangular video
            sample_every: Sample every N degrees
            
        Returns:
            audio_sequence: Array of shape [n_frames, H//sample_every, W//sample_every]
        """
        n_samples = ambisonics.shape[0]
        n_frames = n_samples // self.frame_size
        
        audio_maps = []
        
        print(f"Processing {n_frames} frames...")
        for frame_idx in range(n_frames):
            if frame_idx % 10 == 0:
                print(f"  Frame {frame_idx}/{n_frames}")
            
            energy_map = self.create_audio_energy_map(
                ambisonics, frame_idx, video_width, video_height, sample_every
            )
            audio_maps.append(energy_map)
        
        return np.array(audio_maps)
    
    def visualize_audio_video_alignment(self, video_frame: np.ndarray,
                                       audio_map: np.ndarray,
                                       alpha: float = 0.5) -> None:
        """
        Visualize audio energy map overlaid on video frame.
        
        Args:
            video_frame: Equirectangular video frame [H, W, 3]
            audio_map: Audio energy map (can be lower resolution)
            alpha: Transparency of audio overlay (0-1)
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Video frame
        axes[0].imshow(video_frame)
        axes[0].set_title("Video Frame (Equirectangular)")
        axes[0].axis('off')
        
        # Audio energy map
        audio_db = 20 * np.log10(audio_map + 1e-10)
        im = axes[1].imshow(audio_db, cmap='hot', aspect='auto')
        axes[1].set_title("Audio Energy Map (dB)")
        axes[1].set_xlabel("Longitude (0° to 360°)")
        axes[1].set_ylabel("Latitude (90° to -90°)")
        plt.colorbar(im, ax=axes[1])
        
        # Overlay
        axes[2].imshow(video_frame)
        
        # Resize audio map to match video if needed
        from scipy.ndimage import zoom
        if audio_map.shape != video_frame.shape[:2]:
            scale_y = video_frame.shape[0] / audio_map.shape[0]
            scale_x = video_frame.shape[1] / audio_map.shape[1]
            audio_map_resized = zoom(audio_map, (scale_y, scale_x), order=1)
        else:
            audio_map_resized = audio_map
        
        audio_normalized = (audio_map_resized - audio_map_resized.min()) / \
                          (audio_map_resized.max() - audio_map_resized.min())
        axes[2].imshow(audio_normalized, cmap='hot', alpha=alpha)
        axes[2].set_title("Audio-Visual Overlay")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Simulate ambisonic audio (1 second, sound from right side)
    sample_rate = 48000
    duration = 1.0
    n_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, n_samples)
    
    # 440 Hz tone from 90° (right side), 0° elevation
    source_azimuth = np.radians(90)
    source_elevation = np.radians(0)
    
    tone = np.sin(2 * np.pi * 440 * t)
    W = tone * 1.0
    X = tone * np.cos(source_azimuth) * np.cos(source_elevation)
    Y = tone * np.sin(source_azimuth) * np.cos(source_elevation)
    Z = tone * np.sin(source_elevation)
    
    ambisonics = np.column_stack([W, X, Y, Z])
    
    # Simulate equirectangular video dimensions
    video_width = 3840   # 4K width
    video_height = 1920  # 4K height
    
    # Initialize mapper
    mapper = EquirectangularAudioMapper(sample_rate=sample_rate)
    
    # Create audio energy map for first frame (sampled every 5°)
    print("Creating audio energy map (5° sampling)...")
    audio_map = mapper.create_audio_energy_map(
        ambisonics, 
        frame_idx=0,
        video_width=video_width,
        video_height=video_height,
        sample_every=20  # Sample every 20 pixels for demo
    )
    
    print(f"Audio map shape: {audio_map.shape}")
    print(f"Energy range: {audio_map.min():.6f} to {audio_map.max():.6f}")
    
    # Visualize
    plt.figure(figsize=(12, 6))
    audio_db = 20 * np.log10(audio_map + 1e-10)
    plt.imshow(audio_db, cmap='hot', aspect='auto', origin='upper')
    plt.colorbar(label='Energy (dB)')
    plt.title('Audio Energy Map - Sound from Right (90°)')
    plt.xlabel('Longitude (0° left to 360° right)')
    plt.ylabel('Latitude (90° top to -90° bottom)')
    
    # Mark where sound should be (90° azimuth, 0° elevation)
    # 90° azimuth = 3/4 across width, 0° elevation = middle height
    marker_x = audio_map.shape[1] * 0.75
    marker_y = audio_map.shape[0] * 0.5
    plt.plot(marker_x, marker_y, 'b*', markersize=20, label='Expected source (90°, 0°)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\n✓ Audio energy map created successfully!")
    print(f"  - Map resolution: {audio_map.shape}")
    print(f"  - Peak energy location: row {np.unravel_index(audio_map.argmax(), audio_map.shape)}")