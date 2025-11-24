import numpy as np
from typing import Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    from tfmd.core import TFMD

def decompose_with_padding(signal: np.ndarray, 
                           fs: int, 
                           decomposer: 'TFMD') -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Decomposes a signal using TFMD with boundary padding to reduce edge artifacts.

    Standard STFT processing can introduce artifacts at the start and end of a signal
    due to windowing. For short segments like EEG trials, these artifacts can 
    significantly impact the quality of the decomposition. This function applies
    reflection padding before decomposition and crops the results afterwards.

    Args:
        signal (np.ndarray): The input signal (1D array).
        fs (int): Sampling frequency in Hz.
        decomposer (TFMD): An instance of the TFMD class configured with desired options.

    Returns:
        Tuple[List[np.ndarray], np.ndarray]: 
            - A list of decomposed modes (cropped to original length).
            - The reconstructed signal (cropped to original length).
    """
    # 1. Determine padding length
    # Using half the window length is standard to ensure the first/last sample
    # is the center of at least one window.
    pad_len = decomposer.opts.window_length // 2

    # 2. Apply Padding
    # 'reflect' mode mirrors the signal (e.g., CBA|ABC...|CBA). 
    # This maintains continuity of the signal value at the boundary, 
    # reducing spectral leakage compared to zero-padding.
    padded_signal = np.pad(signal, pad_len, mode='reflect')

    # 3. Perform Decomposition
    # The decomposer works on the longer, padded signal.
    padded_modes, padded_recon = decomposer.decompose(padded_signal, fs)

    # 4. Crop Results
    # Remove the padding to return signals of the original length.
    if padded_modes:
        modes = [m[pad_len:-pad_len] for m in padded_modes]
    else:
        modes = []
        
    reconstructed = padded_recon[pad_len:-pad_len]

    return modes, reconstructed