import pytest
import numpy as np
from tfmd.core import TFMD, TFMDOptions
from tfmd.signal_generator import generate_signal

# Tolerances are set slightly higher than the paper's reported values
# to account for minor implementation differences.
# Paper values: [2.62e-2, 4.11e-2, 4.98e-2, 3.28e-2, 6.28e-2, 5.34e-2]
ERROR_TOLERANCES = {
    1: 0.04,
    2: 0.06,
    3: 0.07,
    4: 0.05,
    5: 0.08, # Higher tolerance for the most complex signal
    6: 0.06,
}

@pytest.mark.parametrize("case_idx", range(1, 7))
def test_tfmd_on_synthetic_signals(case_idx):
    """
    Validates the TFMD implementation against the 6 synthetic signal cases
    from the original paper under noise-free conditions.
    """
    # 1. Generate Signal
    fs = 1000
    data = generate_signal(case_idx, fs=fs)
    signal = data['clean']
    
    # For Case 5, the paper uses a slightly different alpha
    if case_idx == 5:
        options = TFMDOptions(alpha=2.0)
    else:
        options = TFMDOptions()

    # 2. Decompose
    decomposer = TFMD(options=options)
    modes, reconstructed_signal = decomposer.decompose(signal, fs)

    # 3. Assertions
    # a) Check if the correct number of modes were found
    assert len(modes) == data['num_gt'], f"Case {case_idx} ({data['name']}): Incorrect number of modes found."

    # b) Check if the total reconstruction error is within tolerance
    if data['num_gt'] > 0:
        error = np.linalg.norm(signal - reconstructed_signal) / np.linalg.norm(signal)
        tolerance = ERROR_TOLERANCES[case_idx]
        assert error < tolerance, f"Case {case_idx} ({data['name']}): Reconstruction error {error:.4f} exceeds tolerance {tolerance}."