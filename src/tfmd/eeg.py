import numpy as np
import mne
from pathlib import Path
from dataclasses import dataclass

@dataclass
class BCICompetitionIV2a:
    """
    A data loader for the BCI Competition IV Dataset 2a (GitHub .npz version).

    This class handles the conversion of raw NumPy arrays into MNE-Python objects,
    injecting the necessary metadata (channel names, sampling rates, types) that
    is missing from the .npz files.

    Attributes:
        file_path (str | Path): Path to the .npz file (e.g., 'A01T.npz').
    """
    file_path: str | Path

    # Standard montage for BCI Competition IV 2a
    # 22 EEG channels + 3 EOG channels
    CH_NAMES = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 
        'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 
        'P2', 'POz', 
        'EOG-left', 'EOG-central', 'EOG-right'
    ]
    
    CH_TYPES = ['eeg'] * 22 + ['eog'] * 3
    FS = 250  # Sampling frequency is fixed at 250 Hz

    # Event codes mapping
    EVENT_ID = {
        'Left Hand': 769,
        'Right Hand': 770,
        'Foot': 771,
        'Tongue': 772,
        'Eye movements': 1072,
        'Rejected trial': 1023,
        'Start of Trial': 768
    }

    def load(self) -> tuple[mne.io.RawArray, np.ndarray]:
        """
        Loads the data and returns an MNE Raw object and an events array.

        Returns:
            raw (mne.io.RawArray): The continuous raw data with metadata.
            events (np.ndarray): The MNE-compatible events array (N, 3).
                                 [sample_index, 0, event_id]
        """
        path = Path(self.file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        # 1. Load the .npz file
        data = np.load(path)
        
        # 2. Extract Signal
        # The .npz signal 's' is shape (n_samples, n_channels).
        # MNE expects (n_channels, n_samples).
        if 's' not in data:
            raise KeyError("Invalid .npz file: missing key 's' (signal).")
        
        raw_signal = data['s'].T
        
        # Validation
        if raw_signal.shape[0] != len(self.CH_NAMES):
            raise ValueError(
                f"Channel mismatch. Expected {len(self.CH_NAMES)}, "
                f"found {raw_signal.shape[0]} in file."
            )

        # 3. Create MNE Info and Raw Object
        info = mne.create_info(
            ch_names=self.CH_NAMES,
            sfreq=self.FS,
            ch_types=self.CH_TYPES # type: ignore
        )
        
        # Set standard montage (approximate locations for visualization)
        montage = mne.channels.make_standard_montage('standard_1020')
        info.set_montage(montage, on_missing='ignore') # EOGs might be missing in standard montage
        
        raw = mne.io.RawArray(raw_signal, info)

        # 4. Extract and Format Events
        # .npz has 'etyp' (types) and 'epos' (positions)
        # MNE events array is a column stack of [pos, 0, type]
        etyp = data['etyp'].flatten()
        epos = data['epos'].flatten()
        
        # Filter out 0s or NaNs if any exist in the raw arrays
        valid_mask = ~np.isnan(epos)
        etyp = etyp[valid_mask].astype(int)
        epos = epos[valid_mask].astype(int)

        events = np.column_stack((
            epos,
            np.zeros_like(epos, dtype=int),
            etyp
        ))

        return raw, events