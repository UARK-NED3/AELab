"""
plot_wfs.py
===========
Load a .WFS streaming file, plot the raw waveform, and compute/display
the spectrogram.

Usage
-----
    python3 plot_wfs.py STREAM20260308-030707-760.wfs
    python3 plot_wfs.py STREAM20260308-030707-760.wfs --max-records 500
    python3 plot_wfs.py STREAM20260308-030707-760.wfs --max-freq 100e3 --vmin -100 --vmax -20
    python3 plot_wfs.py STREAM20260308-030707-760.wfs --no-show --save-png

The script concatenates all waveform records into one continuous time-series
and produces two figures:
  1. Raw waveform (volts vs time) — rendered as a min/max pixel envelope
  2. Spectrogram (power in dB, frequency vs time) — computed at display resolution

--max-freq, --vmin, --vmax are all optional. When omitted the full frequency
range and matplotlib's auto colorbar limits are used.
"""

import argparse
import sys
import time as _time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from numpy.lib.stride_tricks import as_strided
from scipy.signal import get_window

from decode_wfs import load_continuous, decode_wfs


# ---------------------------------------------------------------------------
# Display-resolution spectrogram
# ---------------------------------------------------------------------------

def display_spectrogram(raw, sampling_frequency, nperseg, n_time_bins):
    """
    Compute a spectrogram with exactly *n_time_bins* output columns —
    one per horizontal pixel in the figure.

    Rather than using the standard overlap-add approach (which produces
    ~N/step time bins regardless of display resolution), the signal is
    divided into n_time_bins equal-length chunks and one windowed FFT is
    computed per chunk. For a 1-billion-sample signal this reduces the
    number of FFTs from ~250,000 to ~2,400, cutting both compute time and
    peak memory use by two orders of magnitude.

    Parameters
    ----------
    raw : np.ndarray, shape (N,)
        The full concatenated signal as float64.
    sampling_frequency : float
        Sample rate in Hz.
    nperseg : int
        FFT window length in samples. Each chunk must be at least this long;
        if chunks are shorter (very short files), nperseg is clamped down.
    n_time_bins : int
        Number of output time columns (= horizontal pixels in the figure).

    Returns
    -------
    frequencies : np.ndarray, shape (n_freq,)
    times       : np.ndarray, shape (n_time_bins,)  — center of each chunk (s)
    Sxx         : np.ndarray, shape (n_freq, n_time_bins)  — power spectrum
    """
    n = len(raw)
    chunk_size = max(1, n // n_time_bins)

    # Clamp nperseg so it never exceeds the chunk size
    nperseg = min(nperseg, chunk_size)

    win      = get_window("hann", nperseg)
    win_norm = (win ** 2).sum() * sampling_frequency  # 'spectrum' scaling

    n_bins_actual = (n - nperseg) // chunk_size + 1

    # Zero-copy strided view: shape (n_bins_actual, nperseg)
    # Each row starts chunk_size samples after the previous one.
    stride = raw.strides[0]
    segments = as_strided(
        raw,
        shape=(n_bins_actual, nperseg),
        strides=(chunk_size * stride, stride),
        writeable=False,
    )

    # Apply window and batch-FFT all segments at once
    windowed = segments * win[np.newaxis, :]
    ffts     = np.fft.rfft(windowed, axis=1)          # (n_bins_actual, n_freq)
    Sxx      = (np.abs(ffts) ** 2 / win_norm).T       # (n_freq, n_bins_actual)

    frequencies = np.fft.rfftfreq(nperseg, d=1.0 / sampling_frequency)
    times       = (np.arange(n_bins_actual) * chunk_size + nperseg / 2) / sampling_frequency

    return frequencies, times, Sxx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot waveform + spectrogram from a .WFS file."
    )
    parser.add_argument("wfs_file",
                        help="Path to the .wfs file")
    parser.add_argument("--channel", type=int, default=None,
                        help="AE channel number to plot (default: all channels concatenated)")
    parser.add_argument("--max-records", type=int, default=None,
                        help="Maximum number of waveform records to load")
    parser.add_argument("--sample-rate", type=float, default=None,
                        help="Override sample rate in Hz (default: read from file)")
    parser.add_argument("--max-freq", type=float, default=None,
                        help="Upper frequency limit for spectrogram in Hz "
                             "(default: no limit — show full Nyquist range)")
    parser.add_argument("--nperseg", type=int, default=None,
                        help="FFT window size (default: 2 × samples-per-record)")
    parser.add_argument("--vmin", type=float, default=None,
                        help="Spectrogram colorbar minimum in dB "
                             "(default: matplotlib auto)")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Spectrogram colorbar maximum in dB "
                             "(default: matplotlib auto)")
    parser.add_argument("--title", type=str, default=None,
                        help="Custom plot title prefix")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not open interactive plot windows")
    parser.add_argument("--save-png", action="store_true",
                        help="Save figures as PNG files next to the .wfs file")
    args = parser.parse_args()

    wfs_path = Path(args.wfs_file)
    if not wfs_path.exists():
        sys.exit(f"File not found: {wfs_path}")

    # ------------------------------------------------------------------ #
    #  1. Load data                                                         #
    # ------------------------------------------------------------------ #
    raw, t, sr = load_continuous(
        wfs_path,
        channel=args.channel,
        max_records=args.max_records,
        sample_rate_hz=args.sample_rate,
    )

    sampling_frequency = float(sr)
    print(f"Sampling frequency: {sampling_frequency:.2f} Hz")

    print('sample_rate_hz=', sr)
    print('index\ttime_s\tsignal_v')
    [print(f'{i}\t{t[i]:.9f}\t{raw[i]:.9f}') for i in range(min(10, len(raw)))]

    stem       = wfs_path.stem
    title_base = args.title or stem
    fig_dpi    = 150
    label_size = 14
    tick_size  = 12

    # ------------------------------------------------------------------ #
    #  2. Waveform plot — min/max pixel envelope                            #
    # ------------------------------------------------------------------ #
    fig_wave, ax_wave = plt.subplots(figsize=(16, 4), dpi=fig_dpi)

    px_wide  = int(fig_wave.get_figwidth()  * fig_dpi)
    bin_size = max(1, len(raw) // px_wide)
    n_use    = (len(raw) // bin_size) * bin_size
    grid     = raw[:n_use].reshape(-1, bin_size)
    t_bins   = t[:n_use].reshape(-1, bin_size)[:, 0]

    env_min = grid.min(axis=1)
    env_max = grid.max(axis=1)

    ax_wave.fill_between(t_bins, env_min, env_max,
                         color="steelblue", alpha=0.8, linewidth=0)
    ax_wave.set_xlabel("Time (s)", fontsize=label_size)
    ax_wave.set_ylabel("Volts (V)", fontsize=label_size)
    ax_wave.set_title(f"Waveform — {title_base}", fontsize=16, pad=12)
    ax_wave.tick_params(axis="both", which="major", labelsize=tick_size)
    ax_wave.set_xlim(t[0], t[-1])
    plt.tight_layout()

    if args.save_png:
        out = wfs_path.with_name(f"{stem}_waveform.png")
        fig_wave.savefig(out, dpi=fig_dpi)
        print(f"Saved: {out}")

    # ------------------------------------------------------------------ #
    #  3. Spectrogram — display-resolution FFT                             #
    # ------------------------------------------------------------------ #
    samples_per_record = len(decode_wfs(wfs_path, max_records=1).waveforms[0].samples)
    nperseg    = args.nperseg or samples_per_record * 2
    px_wide    = int(16 * fig_dpi)
    px_tall    = int(7  * fig_dpi)

    print(f"Spectrogram  nperseg={nperseg:,}  n_time_bins={px_wide:,}")

    _t0 = _time.perf_counter()
    frequencies, times, Sxx = display_spectrogram(
        raw, sampling_frequency, nperseg, n_time_bins=px_wide
    )
    print(f"  display_spectrogram : {_time.perf_counter() - _t0:.2f}s  "
          f"→ Sxx shape {Sxx.shape}")

    _t0 = _time.perf_counter()
    Sxx_log = 10 * np.log10(Sxx)
    print(f"  10*log10(Sxx)       : {_time.perf_counter() - _t0:.2f}s")

    # Frequency mask — only applied when --max-freq is provided
    if args.max_freq is not None:
        freq_mask          = frequencies <= args.max_freq
        masked_frequencies = frequencies[freq_mask]
        masked_Sxx_log     = Sxx_log[freq_mask, :]
    else:
        masked_frequencies = frequencies
        masked_Sxx_log     = Sxx_log

    # Downsample frequency axis to display pixel height
    f_step             = max(1, masked_Sxx_log.shape[0] // px_tall)
    masked_Sxx_log     = masked_Sxx_log[::f_step, :]
    masked_frequencies = masked_frequencies[::f_step]

    print(f"  final image size    : {masked_Sxx_log.shape[0]}×{masked_Sxx_log.shape[1]} "
          f"(freq bins × time bins)")
    if args.vmin is not None or args.vmax is not None:
        print(f"  colorbar range      : vmin={args.vmin}  vmax={args.vmax}")

    _t0 = _time.perf_counter()
    fig_spec, ax1 = plt.subplots(figsize=(16, 7), dpi=fig_dpi)

    spectrogram_image = ax1.imshow(
        masked_Sxx_log,
        extent=[
            times[0],
            times[-1],
            masked_frequencies[-1] / 1e3,
            masked_frequencies[0]  / 1e3,
        ],
        interpolation="bilinear",
        cmap="viridis",
        aspect="auto",
        vmin=args.vmin,
        vmax=args.vmax,
    )
    ax1.invert_yaxis()
    ax1.set_ylabel("Frequency (kHz)", fontsize=label_size)
    ax1.set_xlabel("Time (s)",        fontsize=label_size)
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.1f}")
    )
    plt.subplots_adjust(right=0.85)

    cbar = fig_spec.colorbar(spectrogram_image, ax=[ax1], fraction=0.05, pad=0.04)
    cbar.set_label("Power (dB)", fontsize=label_size)
    cbar.ax.tick_params(labelsize=tick_size)
    plt.tick_params(axis="both", which="major", labelsize=tick_size)
    plt.title(f"Spectrogram — {title_base}", fontsize=16, pad=15)
    print(f"  imshow + colorbar   : {_time.perf_counter() - _t0:.2f}s")

    if args.save_png:
        out = wfs_path.with_name(f"{stem}_spectrogram.png")
        fig_spec.savefig(out, dpi=fig_dpi)
        print(f"Saved: {out}")

    if not args.no_show:
        _t0 = _time.perf_counter()
        plt.show()
        print(f"  plt.show            : {_time.perf_counter() - _t0:.2f}s")


if __name__ == "__main__":
    main()
