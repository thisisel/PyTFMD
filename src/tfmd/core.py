import numpy as np
from dataclasses import dataclass
from scipy.signal import ShortTimeFFT, get_window
from scipy.ndimage import label, binary_dilation
from sklearn.cluster import KMeans


@dataclass
class TFMDOptions:
    """
    Configuration options for the TFMD algorithm.

    Attributes:
        window_length (int): STFT window length (G in the paper).
        alpha (float): Shape parameter for the Gaussian window.
        overlap_ratio (float): STFT window overlap ratio (rho).
        beta (float): Expansion factor for Iterative Competitive Dilation (ICD).
        sigma (float): Minimum component size threshold as a ratio of the spectrogram area.
    """
    window_length: int = 128
    alpha: float = 2.5
    overlap_ratio: float = 0.90
    beta: float = 0.5
    sigma: float = 1e-3

class TFMD:
    """
    Implements the Time-Frequency Mode Decomposition (TFMD) algorithm.

    This class provides a Python implementation of the TFMD algorithm based on the
    2025 paper by Zhou et al., "Time-Frequency Mode Decomposition: A Morphological
    Segmentation Framework for Signal Analysis."

    The process follows a class-based API similar to scikit-learn: initialize the
    decomposer with parameters, then use the `decompose` method on a signal.
    """
    def __init__(self, options: TFMDOptions = TFMDOptions()):
        self.opts = options
        # Attributes to store intermediate results for inspection
        self.spectrogram_ = None
        self.core_mask_ = None
        self.labeled_mask_ = None
        self.final_masks_ = None

    def decompose(self, signal: np.ndarray, fs: int):
        """
        Decomposes a signal into its constituent modes using the TFMD algorithm.

        Args:
            signal (np.ndarray): The input signal as a 1D NumPy array.
            fs (int): The sampling frequency of the signal.

        Returns:
            A tuple containing:
            - list[np.ndarray]: A list of the decomposed signal modes.
            - np.ndarray: The reconstructed signal, formed by summing the modes.
        """
        signal = np.asarray(signal)
        if signal.ndim != 1:
            raise ValueError("Input signal must be a 1D array.")
        n_samples = len(signal)

        # --- 1. STFT ---
        # Configure the Short-Time Fourier Transform
        win = get_window(('gaussian', self.opts.alpha), self.opts.window_length)
        hop = int(self.opts.window_length * (1 - self.opts.overlap_ratio))
        sft = ShortTimeFFT(win, hop=hop, fs=fs, scale_to='psd')
        
        S = sft.stft(signal) # Spectrogram
        magnitude = np.abs(S)
        self.spectrogram_ = magnitude

        # --- 2. K-means Clustering ---
        # Identify high-energy regions in the spectrogram
        kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto')
        labels = kmeans.fit_predict(magnitude.ravel().reshape(-1, 1))
        labels = labels.reshape(magnitude.shape)

        # The signal core corresponds to the cluster with the higher mean energy
        if kmeans.cluster_centers_[0] < kmeans.cluster_centers_[1]:
            signal_id = 1
        else:
            signal_id = 0
        self.core_mask_ = (labels == signal_id)

        # --- 3. Connected-Component Labeling (CCL) ---
        labeled_mask, n_features = label(self.core_mask_) # type: ignore
        self.labeled_mask_ = labeled_mask

        # --- 4. Size Filtering ---
        min_size = round(labeled_mask.size * self.opts.sigma)
        filtered_labels = np.copy(labeled_mask)
        component_sizes = np.bincount(filtered_labels.ravel())
        
        # Remove labels for components that are too small
        for i in range(1, n_features + 1):
            if component_sizes[i] < min_size:
                filtered_labels[filtered_labels == i] = 0
        
        # Re-label the mask to ensure consecutive numbering
        self.labeled_mask_, n_modes = label(filtered_labels > 0) # type: ignore
        
        if n_modes == 0:
            print("TFMD: No modes found after filtering.")
            return [], np.zeros(n_samples)

        # --- 5. Iterative Competitive Dilation (ICD) ---
        # Calculate expansion radius for each component
        r = np.zeros(n_modes, dtype=int)
        for i in range(1, n_modes + 1):
            area = np.sum(self.labeled_mask_ == i)
            r[i-1] = max(1, round(self.opts.beta * np.sqrt(area / np.pi)))

        L_expand = np.copy(self.labeled_mask_)
        blocked = np.zeros_like(L_expand, dtype=bool)
        
        for iter_num in range(1, np.max(r) + 1 if len(r) > 0 else 1):
            claims = np.zeros_like(L_expand)
            for i in range(1, n_modes + 1):
                if iter_num <= r[i-1]:
                    mask_i = (L_expand == i)
                    # Dilate and find the frontier
                    dilated = binary_dilation(mask_i)
                    frontier = dilated & ~mask_i & (L_expand == 0) & ~blocked
                    
                    if np.any(frontier):
                        # Identify contested and new claims
                        contested = frontier & (claims > 0)
                        new_claims = frontier & (claims == 0)
                        claims[new_claims] = i
                        claims[contested] = -1 # Mark contested pixels

            blocked[claims == -1] = True
            accept = (claims > 0) & ~blocked
            if not np.any(accept):
                break
            L_expand[accept] = claims[accept]

        # --- 6. ISTFT (Mode Reconstruction) ---
        components = []
        self.final_masks_ = []
        for i in range(1, n_modes + 1):
            mask = (L_expand == i)
            self.final_masks_.append(mask)
            
            # Apply mask to the original complex spectrogram
            S_masked = S * mask
            
            # Reconstruct the mode via inverse STFT
            recon_mode = sft.istft(S_masked)
            
            # Truncate to original signal length
            components.append(recon_mode[:n_samples])

        # Sum modes to get the final reconstructed signal
        reconstructed_signal = np.sum(components, axis=0) if components else np.zeros(n_samples)
        
        print(f"TFMD: Extracted {len(components)} modes from signal (N={n_samples}, fs={fs} Hz)")
        
        return components, reconstructed_signal