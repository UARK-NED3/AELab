# EasyAE analysis

[`wfs_spectrogram.ipynb`](wfs_spectrogram.ipynb) loads a single channel from a
`.wfs` recording and displays max-pooled and mean-pooled spectrograms. Older
analysis notebooks are retained alongside it.

Install the required packages in the notebook's Python environment:

```sh
pip install decode-wfs numpy scipy matplotlib
```

Run the notebook from top to bottom. Enter the recording filepath when prompted;
set the channel, optional time interval, analysis parameters, and plotting
parameters in the configuration cell. No recording is bundled with this notebook.

All three functions are defined before configuration and loading:

- `select_time_range`: selects `[start_time, end_time)` while preserving the
  decoder's recording time coordinates.
- `compute_spectrogram`: uses SciPy periodograms in batches and returns frequency
  centers, display-column time edges, and a dictionary of linear spectral-density
  matrices. Request `modes=("mean", "max")` or either mode individually.
- `plot_spectrogram`: displays an already-computed dB matrix using `imshow()` and
  returns the figure and axes.

Both pooling modes use identical overlapping windows and display columns. Mean
pooling averages linear values; max pooling takes the highest value independently
at each frequency. Convert the final matrices to dB after computation. Frequency
limits are shared, while the two plots have independent color limits. Rerun only
the plotting cells after changing appearance settings.

Batch size controls spectral working memory, not analysis resolution. The decoded
waveform remains in RAM. Zero padding includes the selected interval's boundaries
by assuming zero values outside it; `boundary=None` keeps complete windows only.
Max-pooled columns can combine peaks from different times, and their levels depend
on the number of contributing windows. Density units are V²/Hz for voltage input.
