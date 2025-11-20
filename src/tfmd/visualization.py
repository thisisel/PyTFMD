import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from typing import Literal


def plot_decomposition(
    original_signal: np.ndarray,
    t: np.ndarray,
    modes: list[np.ndarray],
    reconstructed_signal: np.ndarray,
    title: str = "TFMD Results",
    env: Literal["notebook", "module"] = "module",
):
    """
    Generates a standard plot visualizing the results of a TFMD run.

    Args:
        original_signal (np.ndarray): The input signal.
        t (np.ndarray): The time vector corresponding to the signal.
        modes (list[np.ndarray]): The list of decomposed modes.
        reconstructed_signal (np.ndarray): The signal reconstructed from the modes.
        title (str, optional): The main title for the plot.
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(title, fontsize=16)

    # 1. Original Signal
    axes[0].plot(t, original_signal, label="Original Signal")
    axes[0].set_title("Original Signal")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Decomposed Modes
    if modes:
        for i, mode in enumerate(modes):
            axes[1].plot(t, mode, label=f"Mode {i+1}")
        axes[1].set_title("Decomposed Modes")
        axes[1].legend(loc="upper right")
    else:
        axes[1].set_title("No Modes Found")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # 3. Reconstructed Signal vs. Original
    axes[2].plot(t, reconstructed_signal, label="Reconstructed Signal", color="r")
    axes[2].plot(
        t,
        original_signal,
        label="Original Signal",
        color="k",
        linestyle="--",
        alpha=0.7,
    )
    axes[2].set_title("Reconstructed vs. Original Signal")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # type: ignore
    # Save the figure to a file to be displayed later.
    if env == "notebook":
        plt.show()
    else:
        plt.savefig("decomposition_plot.png")
        plt.close(fig)  # Close the figure to prevent it from displaying in the output


def plot_spectrogram(
    spectrogram: np.ndarray,
    fs: int,
    hop_length: int,
    ax: plt.Axes = None, # type: ignore
    title: str = "Spectrogram",
):
    """
    Plots a magnitude spectrogram with correct time and frequency axes.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    n_freqs, n_frames = spectrogram.shape

    # Construct axes
    # Frequency axis goes from 0 to fs/2
    f_axis = np.linspace(0, fs / 2, n_freqs)
    # Time axis
    t_axis = np.arange(n_frames) * hop_length / fs

    # Use pcolormesh for accurate grid representation
    # Using log scale for intensity often reveals hidden details
    pcm = ax.pcolormesh(t_axis, f_axis, spectrogram, shading="gouraud", cmap="jet")

    ax.set_title(title)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    plt.colorbar(pcm, ax=ax, label="Magnitude")
    return ax


def plot_pipeline_steps(decomposer, fs: int):
    """
    Visualizes the internal state of a TFMD instance after decomposition.
    This is the 'Dashboard' for debugging segmentation failures.

    Args:
        decomposer: A TFMD instance that has already run .decompose()
        fs: Sampling frequency
    """
    if decomposer.spectrogram_ is None:
        raise ValueError("Decomposer has not been run yet.")

    opts = decomposer.opts
    hop = int(opts.window_length * (1 - opts.overlap_ratio))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    fig.suptitle("TFMD Pipeline Inspection", fontsize=16)

    # 1. Raw Spectrogram
    plot_spectrogram(
        decomposer.spectrogram_, fs, hop, ax=axes[0, 0], title="1. Raw STFT Spectrogram"
    )

    # 2. K-Means Core Mask (Binary)
    # We use the same axis logic but plot the boolean mask
    n_freqs, n_frames = decomposer.spectrogram_.shape
    f_axis = np.linspace(0, fs / 2, n_freqs)
    t_axis = np.arange(n_frames) * hop / fs

    axes[0, 1].pcolormesh(
        t_axis, f_axis, decomposer.core_mask_, cmap="gray_r", shading="nearest"
    )
    axes[0, 1].set_title("2. K-Means Signal Core (Clusters=2)")

    # 3. Labeled Components (Post-CCL, Pre-Filter)
    # We use a distinctive colormap to show different integer labels
    cmap_labels = plt.get_cmap("tab20", np.max(decomposer.labeled_mask_) + 1)
    axes[1, 0].pcolormesh(
        t_axis, f_axis, decomposer.labeled_mask_, cmap=cmap_labels, shading="nearest"
    )
    axes[1, 0].set_title("3. Labeled Components (CCL)")

    # 4. Final Expanded Masks (Post-ICD)
    # Merge the list of binary masks into a single label matrix for visualization
    final_composite = np.zeros_like(decomposer.spectrogram_, dtype=int)
    if decomposer.final_masks_:
        for idx, mask in enumerate(decomposer.final_masks_):
            final_composite[mask] = idx + 1

    axes[1, 1].pcolormesh(
        t_axis, f_axis, final_composite, cmap=cmap_labels, shading="nearest"
    )
    axes[1, 1].set_title(f"4. Final Modes after ICD (N={len(decomposer.final_masks_)})")

    plt.tight_layout()
    plt.show()


def plot_instantaneous_frequency(t: np.ndarray, modes: list[np.ndarray], fs: int):
    """
    Calculates and plots the Instantaneous Frequency (IF) of extracted modes using the Hilbert transform.
    Essential for validating chirp tracking.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, mode in enumerate(modes):
        analytic_signal = hilbert(mode)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal)) # type: ignore
        # IF is derivative of phase / (2*pi)
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs

        # Pad to match time vector length
        instantaneous_frequency = np.concatenate(
            ([instantaneous_frequency[0]], instantaneous_frequency)
        )

        # Simple moving average smoothing for display
        window_size = int(fs * 0.01)  # 10ms window
        if window_size > 1:
            instantaneous_frequency = np.convolve(
                instantaneous_frequency, np.ones(window_size) / window_size, mode="same"
            )

        ax.plot(t, instantaneous_frequency, label=f"Mode {i+1} IF")

    ax.set_title("Instantaneous Frequency Trajectories")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.show()
