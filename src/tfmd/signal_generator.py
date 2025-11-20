import numpy as np
from scipy.signal import chirp
from scipy.signal.windows import tukey

def generate_signal(case_idx: int, fs: int = 1000) -> dict:
    """
    Generates synthetic test signals based on the TFMD paper.

    Args:
        case_idx (int): The signal case to generate (1-6).
        fs (int): The sampling frequency in Hz.

    Returns:
        dict: A dictionary containing the signal data, including the clean composite
              signal, ground truth components, time vector, and metadata.
    """
    T_dur = 1.0
    name = ""
    components = []

    if case_idx == 1:  # Frequency-Separated Chirps
        name = 'Frequency-Separated Chirps'
        t = np.arange(0, T_dur, 1/fs)
        c1 = 1.0 * chirp(t, f0=20, f1=70, t1=T_dur, method='linear')
        c2 = 0.9 * chirp(t, f0=130, f1=180, t1=T_dur, method='quadratic')
        components = [c1, c2]

    elif case_idx == 2:  # Sinusoidal FM
        name = 'Sinusoidal FM'
        t = np.arange(0, T_dur, 1/fs)
        c1 = 1.2 * np.cos(2 * np.pi * 100 * t + 15 * np.sin(2 * np.pi * 2 * t))
        c2 = 1.0 * np.cos(2 * np.pi * 250 * t + 5 * np.sin(2 * np.pi * 5 * t))
        components = [c1, c2]

    elif case_idx == 3:  # Four Components Mix
        name = 'Four Components Mix'
        t = np.arange(0, T_dur, 1/fs)
        N = len(t)
        c1 = 1.0 * chirp(t, f0=10, f1=40, t1=T_dur, method='linear')
        c2 = 0.9 * np.sin(2 * np.pi * 100 * t)
        
        c3 = np.zeros(N)
        idx3 = (t >= 0) & (t <= 0.7)
        t3 = t[idx3]
        c3[idx3] = 1.1 * np.cos(2 * np.pi * 350 * t3 + 5 * np.sin(2 * np.pi * 6 * t3))

        c4 = np.zeros(N)
        idx4 = (t >= 0.6) & (t <= 0.9)
        c4[idx4] = 1.2 * np.sin(2 * np.pi * 200 * t[idx4]) * tukey(sum(idx4), 0.25)
        components = [c1, c2, c3, c4]

    elif case_idx == 4:  # Chirp and AM Tone
        name = 'Chirp and AM Tone'
        t = np.arange(0, T_dur, 1/fs)
        c1 = 1.0 * chirp(t, f0=20, f1=80, t1=T_dur, method='linear')
        c2 = 1.1 * (0.8 + 0.4 * np.cos(2 * np.pi * 2 * t)) * np.sin(2 * np.pi * 200 * t)
        components = [c1, c2]

    elif case_idx == 5:  # Generalized Nonlinear (7 components)
        name = 'Generalized Nonlinear'
        T_dur = 3.0
        t = np.arange(0, T_dur, 1/fs)
        N = len(t)

        c1 = np.cos(2 * np.pi * (170 * t + 20 * t**2 + 3 * np.cos(3 * np.pi * t)))
        
        c2 = np.zeros(N)
        idx2 = t <= 1.5
        c2[idx2] = np.cos(2 * np.pi * (75 * t[idx2] + 20 * t[idx2]**2))

        c3 = np.zeros(N)
        idx3 = t >= 1.0
        c3[idx3] = np.cos(2 * np.pi * (10 * t[idx3] + 20 * t[idx3]**2 + 3 * np.cos(3 * np.pi * t[idx3])))
        
        # Helper for dispersive components
        def ifft_dispersive(ratio_start, func):
            Nf = N // 2 + 1
            spec_full = np.zeros(N, dtype=np.complex128)
            idx_start = int(ratio_start * Nf)
            f = np.arange(idx_start, Nf) / T_dur
            spec_pos = func(f)
            spec_full[idx_start:Nf] = spec_pos
            spec_full[N - Nf + 2 : N - idx_start + 1] = np.conj(np.flip(spec_pos[1:]))
            return np.real(np.fft.ifft(spec_full))

        c4 = ifft_dispersive(1/2, lambda f: 30 * np.exp(-1j * 2 * np.pi * (0.4 * f + 2 * np.cos(2 * np.pi * f / 100))))
        c5 = ifft_dispersive(3/5, lambda f: 30 * np.exp(-1j * 2 * np.pi * (0.8 * f + 0.0005 * f**2)))
        c6 = ifft_dispersive(7/10, lambda f: 30 * np.exp(-1j * 2 * np.pi * (1.8 * f + 2 * np.cos(2 * np.pi * f / 100))))
        c7 = ifft_dispersive(8/10, lambda f: 30 * np.exp(-1j * 2 * np.pi * (2.2 * f + 0.0005 * f**2)))
        
        components = [c1, c2, c3, c4, c5, c6, c7]

    elif case_idx == 6:  # Two Simple Tones
        name = 'Two Simple Tones'
        t = np.arange(0, T_dur, 1/fs)
        c1 = 1.0 * np.sin(2 * np.pi * 100 * t)
        c2 = 0.8 * np.sin(2 * np.pi * 200 * t)
        components = [c1, c2]
        
    else:
        raise ValueError(f'Invalid case_idx: {case_idx} (must be 1-6)')

    clean_signal = np.sum(components, axis=0)

    return {
        'clean': clean_signal,
        'components_gt': components,
        't': t,
        'fs': fs,
        'name': name,
        'num_gt': len(components),
        'T_dur': T_dur
    }