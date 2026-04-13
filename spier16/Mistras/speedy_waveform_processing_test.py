from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
import glob
import numpy as np

def extract_time_from_filename(filename):
    # Returns the time the csv was recorded in microseconds
    parts = filename.split('_')
    return int(parts[-1].strip('.csv'))

directory = input("Enter the directory containing CSV files: ").strip()
if not os.path.isdir(directory):
    raise ValueError(f"The provided directory '{directory}' does not exist.")

# csv_files = glob.glob(os.path.join(directory, "*.csv"))
# if not csv_files:
#     raise ValueError(f"No CSV files found in the provided directory '{directory}'.")
ordered_csv_list = sorted(
    [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.csv')],
    key=lambda path: extract_time_from_filename(os.path.basename(path))
)

def load_file(fname):
    return np.loadtxt(fname, delimiter=',', skiprows=12)

with ThreadPoolExecutor() as executor:
    data_list = list(tqdm(executor.map(load_file, ordered_csv_list),
                            total=len(ordered_csv_list),
                            desc="Loading CSV files"))

all_data = np.stack(data_list, axis=0)
print("Loaded data shape:", all_data.shape)

np.set_printoptions(threshold=20)  # Show up to 100 elements before truncating.
print(all_data)
