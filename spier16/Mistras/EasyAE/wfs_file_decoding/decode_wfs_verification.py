import pandas as pd
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.animation as animation
import scipy
from scipy import stats
from scipy.signal import welch, stft, spectrogram
from scipy.stats import chi2
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import os
from datetime import datetime, timedelta
import re
import pywt
from numpy.fft import rfft, rfftfreq
import copy
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle


mpl.rcParams['agg.path.chunksize'] = 10000000

TIME_ATOL = 1e-12
# AEwin64 ASCII exports round signal values to 8 decimal places in volts.
# Half an LSB at that printed precision is enough to treat the values as equal.
SIGNAL_ATOL = 5e-7
DIAGNOSTIC_HALF_WINDOW = 6


def first_mismatch_index(a, b, atol, rtol=0.0, chunk_size=1_000_000):
    """
    Return the index of the first element pair that differs by more than the
    supplied tolerance, or None when all compared elements match.
    """
    n = min(len(a), len(b))
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        matches = np.isclose(a[start:stop], b[start:stop], atol=atol, rtol=rtol)
        if not np.all(matches):
            return start + int(np.flatnonzero(~matches)[0])
    return None


# 1) Ask for folder_path
while True:
    folder_path = input("Enter the absolute path …").strip().strip('"')
    if os.path.isdir(folder_path):
        break
    print("Invalid directory. Try again.")

print(f"Using folder: {folder_path}")

# Build & sort CSV-file list up front
def sort_key(fn):
    base = os.path.splitext(fn)[0]
    parts = base.split('_')
    try:
        return int(parts[-2])
    except:
        return 0

csv_files = sorted(
    [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')],
    key=sort_key
)

# Try loading the pickle, otherwise build & cache
base_name     = os.path.basename(folder_path)
pickle_file   = os.path.join(folder_path, base_name + '_combined.pkl')

if os.path.isfile(pickle_file):
    print("Loading cached array…")
    with open(pickle_file, 'rb') as pf:
        loaded_item = pickle.load(pf)

    if isinstance(loaded_item, dict):
        # already in the new format
        combined_data      = loaded_item['combined_data']
        sample_interval    = loaded_item['sample_interval']
        stream_start_time  = loaded_item['start_time']
        wfm_stream_metadata = {
            'combined_data': combined_data,
            'sample_interval': sample_interval,
            'start_time': stream_start_time,
        }

    elif isinstance(loaded_item, np.ndarray):
        # --- NEW: bring the old array forward into the new-dict format ---
        combined_data = loaded_item

        if csv_files:
            # re-extract sample_interval & start_time from first CSV
            first_csv = os.path.join(folder_path, csv_files[0])
            with open(first_csv, 'r') as f:
                header = [next(f) for _ in range(12)]
            sample_interval = float(header[3].split(':')[-1].strip())
            stream_start_time_str = header[2].split(' ')[-1].strip()
            stream_start_time = datetime.strptime(stream_start_time_str, "%H:%M:%S")
        else:
            raise ValueError("No CSV files found in the specified folder.")

        # build the dict
        wfm_stream_metadata = {
            'combined_data': combined_data,
            'sample_interval': sample_interval,
            'start_time': stream_start_time,
        }

        # save it right back
        print("Upgrading pickle to new format…")
        with open(pickle_file, 'wb') as pf:
            pickle.dump(wfm_stream_metadata, pf)
        print("Re‐saved pickle with metadata.")
else:
    # Always pull sample_interval from first csv file
    first_csv = os.path.join(folder_path, csv_files[0])
    with open(first_csv, 'r') as f:
        header = [next(f) for _ in range(12)]
    sample_interval = float(header[3].split(':')[-1].strip())
    print("Sample interval (s):", sample_interval)
    stream_start_time_str = header[2].split(' ')[-1].strip()
    stream_start_time = datetime.strptime(stream_start_time_str, "%H:%M:%S")
    print("Stream start time:", stream_start_time)
    wfm_stream_metadata = {
        'sample_interval': sample_interval,
        'start_time': stream_start_time,
    }

    data_list = []
    for fn in tqdm(csv_files, desc="Reading CSVs"):
        path = os.path.join(folder_path, fn)
        arr  = np.loadtxt(path, delimiter=',', skiprows=12)
        data_list.append(arr)
    combined_data = np.vstack(data_list)
    wfm_stream_metadata['combined_data'] = combined_data
    print("Saving cache…")
    with open(pickle_file, 'wb') as pf:
        pickle.dump(wfm_stream_metadata, pf)

print("combined_data shape:", combined_data.shape)







from pathlib import Path
from decode_wfs import load_continuous

wfs_path = Path(input("Enter the path to the WFS file: ").strip().strip('"'))

channel = int(input("Enter the channel number to compare (or press Enter for default): ") or 1)

raw, t, sr = load_continuous(
    wfs_path,
    channel=channel
)

if combined_data.ndim != 2 or combined_data.shape[1] < 2:
    raise ValueError(
        "combined_data is expected to be a 2-D array with at least two columns: "
        "[time, signal]."
    )

csv_t = combined_data[:, 0]
csv_raw = combined_data[:, 1]

print("\nComparison summary")
print("------------------")
print(f"CSV array shape        : {combined_data.shape}")
print(f"WFS signal length      : {len(raw):,}")
print(f"WFS time length        : {len(t):,}")

same_length = len(csv_t) == len(t) and len(csv_raw) == len(raw)
print(f"Lengths match          : {same_length}")

compare_len = min(len(csv_t), len(t), len(csv_raw), len(raw))
csv_trimmed = combined_data.shape[0] - compare_len
wfs_trimmed = len(raw) - compare_len

if not same_length:
    print(f"Trimmed CSV samples     : {csv_trimmed:,}")
    print(f"Trimmed WFS samples     : {wfs_trimmed:,}")
    print(f"Comparing first         : {compare_len:,} samples")

csv_t_cmp = csv_t[:compare_len]
csv_raw_cmp = csv_raw[:compare_len]
t_cmp = t[:compare_len]
raw_cmp = raw[:compare_len]

first_time_bad = first_mismatch_index(csv_t_cmp, t_cmp, atol=TIME_ATOL, rtol=0.0)
first_signal_bad = first_mismatch_index(csv_raw_cmp, raw_cmp, atol=SIGNAL_ATOL, rtol=0.0)

time_match = first_time_bad is None
signal_match = first_signal_bad is None

print(f"Time arrays match      : {time_match}")
print(f"Signal arrays match    : {signal_match}")

if time_match and signal_match and same_length:
    print("\nResult: stitched CSV output matches load_continuous() exactly.")
elif time_match and signal_match:
    print("\nResult: overlapping samples match exactly after trimming the longer array.")
else:
    print("\nResult: arrays do not match, even after trimming.")
    if first_time_bad is not None:
        print("First time mismatch:")
        print(
            f"  index={first_time_bad:,}  "
            f"csv_t={csv_t_cmp[first_time_bad]:.16g}  "
            f"wfs_t={t_cmp[first_time_bad]:.16g}  "
            f"abs_diff={abs(csv_t_cmp[first_time_bad] - t_cmp[first_time_bad]):.16g}"
        )
    if first_signal_bad is not None:
        print("First signal mismatch:")
        print(
            f"  index={first_signal_bad:,}  "
            f"csv_signal={csv_raw_cmp[first_signal_bad]:.16g}  "
            f"wfs_signal={raw_cmp[first_signal_bad]:.16g}  "
            f"abs_diff={abs(csv_raw_cmp[first_signal_bad] - raw_cmp[first_signal_bad]):.16g}"
        )

        start = max(0, first_signal_bad - DIAGNOSTIC_HALF_WINDOW)
        stop = min(compare_len, first_signal_bad + DIAGNOSTIC_HALF_WINDOW + 1)
        print("\nSignal diagnostic around first mismatch:")
        print("index\ttime_s\tcsv_signal\twfs_signal\tabs_diff")
        for i in range(start, stop):
            print(
                f"{i:,}\t"
                f"{t_cmp[i]:.9f}\t"
                f"{csv_raw_cmp[i]:.8f}\t"
                f"{raw_cmp[i]:.11f}\t"
                f"{abs(csv_raw_cmp[i] - raw_cmp[i]):.11f}"
            )

        print("\nSeam-shift diagnostic (WFS shifted by -1, 0, +1 sample):")
        print("index\ttime_s\tcsv_signal\twfs(-1)\tdiff(-1)\twfs(0)\tdiff(0)\twfs(+1)\tdiff(+1)")
        for i in range(start, stop):
            row = [
                f"{i:,}",
                f"{t_cmp[i]:.9f}",
                f"{csv_raw_cmp[i]:.8f}",
            ]

            for shift in (-1, 0, 1):
                j = i + shift
                if 0 <= j < compare_len:
                    wfs_val = raw_cmp[j]
                    diff_val = abs(csv_raw_cmp[i] - wfs_val)
                    row.extend([f"{wfs_val:.11f}", f"{diff_val:.11f}"])
                else:
                    row.extend(["NA", "NA"])

            print("\t".join(row))
