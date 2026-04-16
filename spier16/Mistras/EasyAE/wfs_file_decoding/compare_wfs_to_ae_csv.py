"""
Compare AEwin64 CSV stream exports against decode_wfs.py output.

This is a diagnostic workbench for reverse-engineering WFS stitching issues.
It intentionally reads only a requested CSV window and only a requested number
of WFS records by default, so it is safe to run on very large AE streams.

Example
-------
python compare_wfs_to_ae_csv.py --sample-count 200000 --max-records 256

To write a side-by-side sample table:
python compare_wfs_to_ae_csv.py --side-by-side-out wfs_csv_side_by_side.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from decode_wfs import (
    CID_OFFSET,
    MSG_ID_DOCUMENTED,
    MSG_ID_STREAMING,
    SUBID_HW_SETUP,
    SUBID_STREAM_START,
    SUBID_WAVEFORM,
    VOLTS_PER_COUNT,
    WAVEFORM_HEADER_BYTES,
    WFSFile,
    WaveformRecord,
    _parse_hardware_setup,
    _parse_stream_start_sample_index,
    _parse_waveform_start_sample_index,
)


DEFAULT_WFS = (
    r"\\HULAB-2\Files\0_Ishraq\New Pool Boiling Video\Boiling-398"
    r"\STREAM20260308-030707-760.wfs"
)
DEFAULT_WFMS_DIR = (
    r"\\HULAB-2\Files\0_Ishraq\New Pool Boiling Video\Boiling-398\wfms"
)

TIME_ATOL = 1e-12
SIGNAL_ATOL = 5e-7


@dataclass
class CsvWindow:
    time_s: np.ndarray
    voltage_v: np.ndarray
    sample_interval_s: float | None
    files_read: list[str]
    skipped_rows: int


@dataclass
class StrategyResult:
    name: str
    signal: np.ndarray
    best_lag: int
    compared: int
    mean_abs: float
    rmse: float
    max_abs: float
    within_tol_pct: float
    corr: float
    first_bad: int | None


@dataclass
class WfsWindow:
    signal: np.ndarray
    sample_rate_hz: int
    stream_start_sample_index: int | None
    records_seen: int
    records_used: int
    first_record_start: int | None
    last_record_start: int | None
    channels: set[int]
    hardware_setup_hex: str | None
    pretrigger_samples: int | None


@dataclass
class BinAccumulator:
    start_sample: int
    bin_samples: int
    n_bins: int
    count: np.ndarray
    sumsq: np.ndarray
    min_value: np.ndarray
    max_value: np.ndarray

    @classmethod
    def create(cls, start_sample: int, sample_count: int, bin_samples: int) -> "BinAccumulator":
        n_bins = int(math.ceil(sample_count / bin_samples))
        return cls(
            start_sample=start_sample,
            bin_samples=bin_samples,
            n_bins=n_bins,
            count=np.zeros(n_bins, dtype=np.int64),
            sumsq=np.zeros(n_bins, dtype=np.float64),
            min_value=np.full(n_bins, np.inf, dtype=np.float64),
            max_value=np.full(n_bins, -np.inf, dtype=np.float64),
        )

    def add(self, global_start_sample: int, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        offset = global_start_sample - self.start_sample
        pos = 0
        while pos < len(values):
            absolute = offset + pos
            if absolute < 0:
                skip = min(len(values) - pos, -absolute)
                pos += skip
                continue
            bin_idx = absolute // self.bin_samples
            if bin_idx >= self.n_bins:
                break
            in_bin = absolute % self.bin_samples
            take = min(len(values) - pos, self.bin_samples - in_bin)
            chunk = values[pos : pos + take]
            finite = np.isfinite(chunk)
            if np.any(finite):
                chunk = chunk[finite]
                self.count[bin_idx] += len(chunk)
                self.sumsq[bin_idx] += float(np.sum(chunk * chunk))
                self.min_value[bin_idx] = min(self.min_value[bin_idx], float(np.min(chunk)))
                self.max_value[bin_idx] = max(self.max_value[bin_idx], float(np.max(chunk)))
            pos += take

    def rms(self) -> np.ndarray:
        out = np.full(self.n_bins, np.nan, dtype=np.float64)
        valid = self.count > 0
        out[valid] = np.sqrt(self.sumsq[valid] / self.count[valid])
        return out


def sort_key(filename: str) -> tuple[int, str]:
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_")
    try:
        return int(parts[-2]), filename
    except (ValueError, IndexError):
        return 0, filename


def list_csv_files(folder: Path) -> list[Path]:
    files = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() == ".csv"],
        key=lambda p: sort_key(p.name),
    )
    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")
    return files


def parse_csv_sample_range(path: Path) -> tuple[int, int] | None:
    parts = path.stem.split("_")
    if len(parts) < 3:
        return None
    try:
        start = int(parts[-2])
        end = int(parts[-1])
    except ValueError:
        return None
    return start, end


def read_csv_header(path: Path) -> tuple[list[str], float | None]:
    with path.open("r", newline="") as f:
        header = [next(f).rstrip("\n") for _ in range(12)]

    sample_interval = None
    if len(header) > 3 and ":" in header[3]:
        try:
            sample_interval = float(header[3].split(":")[-1].strip())
        except ValueError:
            sample_interval = None
    return header, sample_interval


def read_csv_time_origin(folder: Path) -> tuple[float, float]:
    files = list_csv_files(folder)
    _, sample_interval = read_csv_header(files[0])
    first_row = np.loadtxt(files[0], delimiter=",", skiprows=12, max_rows=1)
    if first_row.ndim > 1:
        first_time = float(first_row[0, 0])
    else:
        first_time = float(first_row[0])
    if sample_interval is None:
        second_rows = np.loadtxt(files[0], delimiter=",", skiprows=12, max_rows=2)
        sample_interval = float(second_rows[1, 0] - second_rows[0, 0])
    return first_time, float(sample_interval)


def count_csv_data_rows(path: Path) -> int:
    with path.open("r", newline="") as f:
        for _ in range(12):
            next(f, None)
        return sum(1 for _ in f)


def load_csv_window(folder: Path, start_sample: int, sample_count: int) -> CsvWindow:
    files = list_csv_files(folder)
    _, sample_interval = read_csv_header(files[0])

    remaining_skip = int(start_sample)
    remaining_take = int(sample_count)
    chunks: list[np.ndarray] = []
    files_read: list[str] = []
    skipped_rows = 0

    for csv_path in files:
        if remaining_take <= 0:
            break

        row_count = None
        if remaining_skip > 0:
            row_count = count_csv_data_rows(csv_path)
            if remaining_skip >= row_count:
                remaining_skip -= row_count
                skipped_rows += row_count
                continue

        arr = np.loadtxt(csv_path, delimiter=",", skiprows=12)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < 2:
            raise ValueError(f"Expected at least two columns in {csv_path}")

        if remaining_skip > 0:
            arr = arr[remaining_skip:]
            skipped_rows += remaining_skip
            remaining_skip = 0

        if len(arr) > remaining_take:
            arr = arr[:remaining_take]

        chunks.append(arr[:, :2])
        files_read.append(str(csv_path))
        remaining_take -= len(arr)

    if not chunks:
        raise ValueError(
            f"Could not read any CSV samples from {folder} at start_sample={start_sample}"
        )

    combined = np.vstack(chunks)
    if len(combined) < sample_count:
        print(
            f"WARNING: requested {sample_count:,} CSV samples but only read "
            f"{len(combined):,}."
        )

    return CsvWindow(
        time_s=combined[:, 0].astype(np.float64, copy=False),
        voltage_v=combined[:, 1].astype(np.float64, copy=False),
        sample_interval_s=sample_interval,
        files_read=files_read,
        skipped_rows=skipped_rows,
    )


def iter_csv_window_chunks(
    folder: Path,
    start_sample: int,
    sample_count: int,
    max_rows_per_chunk: int = 1_000_000,
) -> Iterable[tuple[int, np.ndarray, np.ndarray, str]]:
    files = list_csv_files(folder)
    end_sample = start_sample + sample_count
    for csv_path in files:
        file_range = parse_csv_sample_range(csv_path)
        if file_range is not None:
            file_start, file_end = file_range
            if file_end < start_sample:
                continue
            if file_start >= end_sample:
                break
            local_skip = max(0, start_sample - file_start)
            local_take = min(file_end + 1, end_sample) - (file_start + local_skip)
        else:
            row_count = count_csv_data_rows(csv_path)
            # Fall back to cumulative row scanning only when filenames do not
            # expose sample ranges.
            local_skip = 0
            local_take = row_count

        if local_take <= 0:
            continue

        rows_left = int(local_take)
        rows_skip = int(local_skip)
        global_row = (file_range[0] + rows_skip) if file_range is not None else start_sample
        while rows_left > 0:
            take = min(rows_left, max_rows_per_chunk)
            arr = np.loadtxt(
                csv_path,
                delimiter=",",
                skiprows=12 + rows_skip,
                max_rows=take,
            )
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            yield (
                global_row,
                arr[:, 0].astype(np.float64, copy=False),
                arr[:, 1].astype(np.float64, copy=False),
                str(csv_path),
            )
            rows_left -= take
            rows_skip += take
            global_row += take


def scan_csv_rms(
    folder: Path,
    start_sample: int,
    sample_count: int,
    bin_samples: int,
) -> tuple[BinAccumulator, list[str]]:
    acc = BinAccumulator.create(start_sample, sample_count, bin_samples)
    files_read: list[str] = []
    for global_start, _time_s, voltage_v, filename in iter_csv_window_chunks(
        folder, start_sample, sample_count
    ):
        acc.add(global_start, voltage_v)
        if not files_read or files_read[-1] != filename:
            files_read.append(filename)
    return acc, files_read


def decode_wfs_prefix(path: Path, max_records: int | None) -> WFSFile:
    """
    Streaming-prefix version of decode_wfs.decode_wfs().

    decode_wfs.py currently reads the entire WFS file into memory before it can
    stop at max_records. This local copy reads message-by-message so diagnostics
    can start from the front of multi-GB stream files.
    """
    result = WFSFile(path=path)
    n_waveforms = 0

    with path.open("rb") as f:
        while True:
            if max_records is not None and n_waveforms >= max_records:
                break

            len_bytes = f.read(2)
            if len(len_bytes) < 2:
                break

            msg_len = struct.unpack("<H", len_bytes)[0]
            if msg_len == 0:
                continue

            body = f.read(msg_len)
            if len(body) < msg_len:
                print("WARNING: truncated WFS message at end of file.")
                break
            if len(body) < 2:
                continue

            msg_id = body[0]
            sub_id = body[1]
            if msg_id not in (MSG_ID_STREAMING, MSG_ID_DOCUMENTED):
                continue

            if sub_id == SUBID_HW_SETUP:
                result.hardware_setup = _parse_hardware_setup(body)
                result.sample_rate_hz = result.hardware_setup.sample_rate_hz
                continue

            if sub_id == SUBID_STREAM_START:
                result.stream_start_sample_index = _parse_stream_start_sample_index(body)
                continue

            if sub_id == SUBID_WAVEFORM:
                if msg_len <= WAVEFORM_HEADER_BYTES:
                    continue

                n_samples = (msg_len - WAVEFORM_HEADER_BYTES) // 2
                sample_bytes = body[
                    WAVEFORM_HEADER_BYTES : WAVEFORM_HEADER_BYTES + n_samples * 2
                ]
                raw_counts = np.frombuffer(sample_bytes, dtype="<i2")
                samples = raw_counts.astype(np.float64) * VOLTS_PER_COUNT
                pretrigger_samples = (
                    result.hardware_setup.pretrigger_samples
                    if result.hardware_setup is not None
                    else None
                )
                result.waveforms.append(
                    WaveformRecord(
                        channel=body[CID_OFFSET] if len(body) > CID_OFFSET else 0,
                        samples=samples,
                        sample_rate_hz=result.sample_rate_hz,
                        pretrigger_samples=pretrigger_samples,
                        start_sample_index=_parse_waveform_start_sample_index(body),
                    )
                )
                n_waveforms += 1

    pretrigger_samples = (
        result.hardware_setup.pretrigger_samples
        if result.hardware_setup is not None
        else None
    )
    for rec in result.waveforms:
        if rec.sample_rate_hz is None:
            rec.sample_rate_hz = result.sample_rate_hz
        if rec.pretrigger_samples is None:
            rec.pretrigger_samples = pretrigger_samples

    return result


def decode_wfs_sample_window(
    path: Path,
    channel: int,
    start_sample: int,
    sample_count: int,
    pointer_scale: float = 2.0,
    sample_rate_hz: int | None = None,
) -> WfsWindow:
    """
    Decode only the WFS records that overlap one continuous stream window.

    start_sample and sample_count use the same sample index convention as
    decode_wfs.load_continuous(): sample 0 is the first sample after applying
    the AEwin64 stream-start trim.
    """
    end_sample = start_sample + sample_count
    signal = np.full(sample_count, np.nan, dtype=np.float64)
    channels: set[int] = set()
    stream_start_sample_index = None
    hardware_setup_hex = None
    pretrigger_samples = None
    sr = sample_rate_hz
    records_seen = 0
    records_used = 0
    first_record_start = None
    last_record_start = None
    last_end = 0

    with path.open("rb") as f:
        while True:
            len_bytes = f.read(2)
            if len(len_bytes) < 2:
                break

            msg_len = struct.unpack("<H", len_bytes)[0]
            if msg_len == 0:
                continue

            body = f.read(msg_len)
            if len(body) < msg_len:
                print("WARNING: truncated WFS message at end of file.")
                break
            if len(body) < 2:
                continue

            msg_id = body[0]
            sub_id = body[1]
            if msg_id not in (MSG_ID_STREAMING, MSG_ID_DOCUMENTED):
                continue

            if sub_id == SUBID_HW_SETUP:
                hs = _parse_hardware_setup(body)
                sr = sample_rate_hz or hs.sample_rate_hz
                pretrigger_samples = hs.pretrigger_samples
                hardware_setup_hex = hs.raw_hex
                continue

            if sub_id == SUBID_STREAM_START:
                stream_start_sample_index = _parse_stream_start_sample_index(body)
                continue

            if sub_id != SUBID_WAVEFORM or msg_len <= WAVEFORM_HEADER_BYTES:
                continue

            cid = body[CID_OFFSET] if len(body) > CID_OFFSET else 0
            channels.add(cid)
            if cid != channel:
                continue

            if stream_start_sample_index is None:
                raise ValueError("Saw waveform data before stream-start sample index.")

            raw_pointer = _parse_waveform_start_sample_index(body)
            if raw_pointer is None:
                continue
            record_start = int(round((raw_pointer / 2.0) * pointer_scale)) - stream_start_sample_index

            n_samples = (msg_len - WAVEFORM_HEADER_BYTES) // 2
            src_start = max(0, -record_start)
            placed_start = max(0, record_start)

            overlap = max(0, last_end - placed_start)
            src_start += overlap
            placed_start += overlap

            placed_len = n_samples - src_start
            placed_end = placed_start + max(0, placed_len)
            last_end = max(last_end, placed_end)
            records_seen += 1

            if placed_end <= start_sample:
                continue
            if placed_start >= end_sample:
                break

            sample_bytes = body[
                WAVEFORM_HEADER_BYTES : WAVEFORM_HEADER_BYTES + n_samples * 2
            ]
            raw_counts = np.frombuffer(sample_bytes, dtype="<i2")
            samples = raw_counts.astype(np.float64) * VOLTS_PER_COUNT

            copy_start = max(placed_start, start_sample)
            copy_end = min(placed_end, end_sample)
            source_start = src_start + (copy_start - placed_start)
            dest_start = copy_start - start_sample
            dest_end = copy_end - start_sample
            signal[dest_start:dest_end] = samples[source_start : source_start + (dest_end - dest_start)]

            records_used += 1
            if first_record_start is None:
                first_record_start = record_start
            last_record_start = record_start

    if sr is None:
        sr = 1_000_000

    return WfsWindow(
        signal=signal,
        sample_rate_hz=int(sr),
        stream_start_sample_index=stream_start_sample_index,
        records_seen=records_seen,
        records_used=records_used,
        first_record_start=first_record_start,
        last_record_start=last_record_start,
        channels=channels,
        hardware_setup_hex=hardware_setup_hex,
        pretrigger_samples=pretrigger_samples,
    )


def scan_wfs_rms(
    path: Path,
    channel: int,
    start_sample: int,
    sample_count: int,
    bin_samples: int,
    pointer_scale: float = 2.0,
    sample_rate_hz: int | None = None,
) -> tuple[BinAccumulator, WfsWindow]:
    end_sample = start_sample + sample_count
    acc = BinAccumulator.create(start_sample, sample_count, bin_samples)
    channels: set[int] = set()
    stream_start_sample_index = None
    hardware_setup_hex = None
    pretrigger_samples = None
    sr = sample_rate_hz
    records_seen = 0
    records_used = 0
    first_record_start = None
    last_record_start = None
    last_end = 0

    with path.open("rb") as f:
        while True:
            len_bytes = f.read(2)
            if len(len_bytes) < 2:
                break

            msg_len = struct.unpack("<H", len_bytes)[0]
            if msg_len == 0:
                continue

            body = f.read(msg_len)
            if len(body) < msg_len:
                print("WARNING: truncated WFS message at end of file.")
                break
            if len(body) < 2:
                continue

            msg_id = body[0]
            sub_id = body[1]
            if msg_id not in (MSG_ID_STREAMING, MSG_ID_DOCUMENTED):
                continue

            if sub_id == SUBID_HW_SETUP:
                hs = _parse_hardware_setup(body)
                sr = sample_rate_hz or hs.sample_rate_hz
                pretrigger_samples = hs.pretrigger_samples
                hardware_setup_hex = hs.raw_hex
                continue

            if sub_id == SUBID_STREAM_START:
                stream_start_sample_index = _parse_stream_start_sample_index(body)
                continue

            if sub_id != SUBID_WAVEFORM or msg_len <= WAVEFORM_HEADER_BYTES:
                continue

            cid = body[CID_OFFSET] if len(body) > CID_OFFSET else 0
            channels.add(cid)
            if cid != channel:
                continue

            if stream_start_sample_index is None:
                raise ValueError("Saw waveform data before stream-start sample index.")

            raw_pointer = _parse_waveform_start_sample_index(body)
            if raw_pointer is None:
                continue
            record_start = int(round((raw_pointer / 2.0) * pointer_scale)) - stream_start_sample_index

            n_samples = (msg_len - WAVEFORM_HEADER_BYTES) // 2
            src_start = max(0, -record_start)
            placed_start = max(0, record_start)

            overlap = max(0, last_end - placed_start)
            src_start += overlap
            placed_start += overlap

            placed_len = n_samples - src_start
            placed_end = placed_start + max(0, placed_len)
            last_end = max(last_end, placed_end)
            records_seen += 1

            if placed_end <= start_sample:
                continue
            if placed_start >= end_sample:
                break

            sample_bytes = body[
                WAVEFORM_HEADER_BYTES : WAVEFORM_HEADER_BYTES + n_samples * 2
            ]
            raw_counts = np.frombuffer(sample_bytes, dtype="<i2")
            samples = raw_counts.astype(np.float64) * VOLTS_PER_COUNT

            copy_start = max(placed_start, start_sample)
            copy_end = min(placed_end, end_sample)
            source_start = src_start + (copy_start - placed_start)
            chunk = samples[source_start : source_start + (copy_end - copy_start)]
            acc.add(copy_start, chunk)

            records_used += 1
            if first_record_start is None:
                first_record_start = record_start
            last_record_start = record_start

    if sr is None:
        sr = 1_000_000

    meta = WfsWindow(
        signal=np.empty(0, dtype=np.float64),
        sample_rate_hz=int(sr),
        stream_start_sample_index=stream_start_sample_index,
        records_seen=records_seen,
        records_used=records_used,
        first_record_start=first_record_start,
        last_record_start=last_record_start,
        channels=channels,
        hardware_setup_hex=hardware_setup_hex,
        pretrigger_samples=pretrigger_samples,
    )
    return acc, meta


def scan_wfs_sequential_rms(
    path: Path,
    channel: int,
    start_sample: int,
    sample_count: int,
    bin_samples: int,
    sample_rate_hz: int | None = None,
) -> tuple[BinAccumulator, WfsWindow]:
    """
    Scan WFS RMS by concatenating waveform records in file order.

    This applies only the initial stream-start modulo trim, then ignores later
    per-record pointer jumps. It is useful for testing whether body[24:28]
    should be treated as an absolute placement field or just metadata.
    """
    end_sample = start_sample + sample_count
    acc = BinAccumulator.create(start_sample, sample_count, bin_samples)
    channels: set[int] = set()
    stream_start_sample_index = None
    hardware_setup_hex = None
    pretrigger_samples = None
    sr = sample_rate_hz
    records_seen = 0
    records_used = 0
    first_record_start = None
    last_record_start = None
    record_len = None
    seq_pos = 0

    with path.open("rb") as f:
        while True:
            len_bytes = f.read(2)
            if len(len_bytes) < 2:
                break

            msg_len = struct.unpack("<H", len_bytes)[0]
            if msg_len == 0:
                continue

            body = f.read(msg_len)
            if len(body) < msg_len:
                print("WARNING: truncated WFS message at end of file.")
                break
            if len(body) < 2:
                continue

            msg_id = body[0]
            sub_id = body[1]
            if msg_id not in (MSG_ID_STREAMING, MSG_ID_DOCUMENTED):
                continue

            if sub_id == SUBID_HW_SETUP:
                hs = _parse_hardware_setup(body)
                sr = sample_rate_hz or hs.sample_rate_hz
                pretrigger_samples = hs.pretrigger_samples
                hardware_setup_hex = hs.raw_hex
                continue

            if sub_id == SUBID_STREAM_START:
                stream_start_sample_index = _parse_stream_start_sample_index(body)
                continue

            if sub_id != SUBID_WAVEFORM or msg_len <= WAVEFORM_HEADER_BYTES:
                continue

            cid = body[CID_OFFSET] if len(body) > CID_OFFSET else 0
            channels.add(cid)
            if cid != channel:
                continue

            n_samples = (msg_len - WAVEFORM_HEADER_BYTES) // 2
            if record_len is None:
                record_len = n_samples
                if stream_start_sample_index is not None and record_len > 0:
                    seq_pos = -(stream_start_sample_index % record_len)

            placed_start = seq_pos
            placed_end = seq_pos + n_samples
            records_seen += 1

            if placed_end > start_sample and placed_start < end_sample:
                sample_bytes = body[
                    WAVEFORM_HEADER_BYTES : WAVEFORM_HEADER_BYTES + n_samples * 2
                ]
                raw_counts = np.frombuffer(sample_bytes, dtype="<i2")
                samples = raw_counts.astype(np.float64) * VOLTS_PER_COUNT
                copy_start = max(placed_start, start_sample)
                copy_end = min(placed_end, end_sample)
                source_start = copy_start - placed_start
                acc.add(copy_start, samples[source_start : source_start + (copy_end - copy_start)])
                records_used += 1
                if first_record_start is None:
                    first_record_start = placed_start
                last_record_start = placed_start

            seq_pos += n_samples
            if placed_start >= end_sample:
                break

    if sr is None:
        sr = 1_000_000

    meta = WfsWindow(
        signal=np.empty(0, dtype=np.float64),
        sample_rate_hz=int(sr),
        stream_start_sample_index=stream_start_sample_index,
        records_seen=records_seen,
        records_used=records_used,
        first_record_start=first_record_start,
        last_record_start=last_record_start,
        channels=channels,
        hardware_setup_hex=hardware_setup_hex,
        pretrigger_samples=pretrigger_samples,
    )
    return acc, meta


def channel_records(wfs: WFSFile, channel: int) -> list[WaveformRecord]:
    records = [r for r in wfs.waveforms if r.channel == channel]
    if not records:
        raise ValueError(f"No WFS waveform records found for channel {channel}")
    return records


def reconstruct_concat(records: list[WaveformRecord]) -> np.ndarray:
    return np.concatenate([r.samples for r in records]).astype(np.float64, copy=False)


def reconstruct_concat_with_stream_mod(
    records: list[WaveformRecord],
    stream_start_sample_index: int | None,
) -> np.ndarray:
    raw = reconstruct_concat(records)
    if stream_start_sample_index is None or not records or len(records[0].samples) == 0:
        return raw
    offset = stream_start_sample_index % len(records[0].samples)
    return raw[offset:]


def reconstruct_positioned(
    records: list[WaveformRecord],
    stream_start_sample_index: int | None,
    pointer_scale: float,
) -> np.ndarray | None:
    if stream_start_sample_index is None:
        return None
    if any(r.start_sample_index is None for r in records):
        return None

    spans = []
    last_end = 0
    for rec in records:
        # decode_wfs.py currently stores 2 * the raw uint32 at body[24:28].
        pointer_raw = rec.start_sample_index / 2.0
        start = int(round(pointer_raw * pointer_scale)) - stream_start_sample_index
        src_start = max(0, -start)
        placed_start = max(0, start)

        overlap = max(0, last_end - placed_start)
        src_start += overlap
        placed_start += overlap

        placed_len = len(rec.samples) - src_start
        placed_end = placed_start + max(0, placed_len)
        spans.append((rec, src_start, placed_start, placed_end))
        last_end = max(last_end, placed_end)

    if not spans:
        return None
    total_len = max(end for _, _, _, end in spans)
    raw = np.zeros(total_len, dtype=np.float64)
    for rec, src_start, placed_start, placed_end in spans:
        if src_start >= len(rec.samples):
            continue
        raw[placed_start:placed_end] = rec.samples[src_start:]
    return raw


def candidate_lags(max_lag: int) -> list[int]:
    max_lag = abs(int(max_lag))
    if max_lag <= 512:
        return list(range(-max_lag, max_lag + 1))

    lags = set(range(-512, 513))
    step = max(1, max_lag // 512)
    lags.update(range(-max_lag, max_lag + 1, step))
    for value in (1024, 2048, 4096, 8192, 16384):
        if value <= max_lag:
            lags.add(value)
            lags.add(-value)
    return sorted(lags)


def aligned_views(a: np.ndarray, b: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag >= 0:
        n = min(len(a), len(b) - lag)
        if n <= 0:
            return a[:0], b[:0]
        return a[:n], b[lag : lag + n]

    n = min(len(a) + lag, len(b))
    if n <= 0:
        return a[:0], b[:0]
    return a[-lag : -lag + n], b[:n]


def first_mismatch_index(
    a: np.ndarray,
    b: np.ndarray,
    atol: float = SIGNAL_ATOL,
    chunk_size: int = 1_000_000,
) -> int | None:
    n = min(len(a), len(b))
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        ok = np.isclose(a[start:stop], b[start:stop], atol=atol, rtol=0.0)
        if not np.all(ok):
            return start + int(np.flatnonzero(~ok)[0])
    return None


def score_one(csv_signal: np.ndarray, wfs_signal: np.ndarray, lag: int) -> tuple:
    a, b = aligned_views(csv_signal, wfs_signal, lag)
    if len(a) == 0:
        return 0, math.inf, math.inf, math.inf, 0.0, math.nan, None

    diff = a - b
    abs_diff = np.abs(diff)
    mean_abs = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    max_abs = float(np.max(abs_diff))
    within_tol_pct = float(100.0 * np.mean(abs_diff <= SIGNAL_ATOL))
    if len(a) > 1 and np.std(a) > 0 and np.std(b) > 0:
        corr = float(np.corrcoef(a, b)[0, 1])
    else:
        corr = math.nan
    first_bad = first_mismatch_index(a, b)
    return len(a), mean_abs, rmse, max_abs, within_tol_pct, corr, first_bad


def best_strategy_result(
    name: str,
    csv_signal: np.ndarray,
    wfs_signal: np.ndarray,
    max_lag: int,
) -> StrategyResult:
    best = None
    for lag in candidate_lags(max_lag):
        compared, mean_abs, rmse, max_abs, within_tol_pct, corr, first_bad = score_one(
            csv_signal, wfs_signal, lag
        )
        row = (mean_abs, rmse, -within_tol_pct, -compared, lag)
        if best is None or row < best[0]:
            best = (
                row,
                StrategyResult(
                    name=name,
                    signal=wfs_signal,
                    best_lag=lag,
                    compared=compared,
                    mean_abs=mean_abs,
                    rmse=rmse,
                    max_abs=max_abs,
                    within_tol_pct=within_tol_pct,
                    corr=corr,
                    first_bad=first_bad,
                ),
            )
    assert best is not None
    return best[1]


def print_wfs_summary(wfs: WFSFile, records: list[WaveformRecord]) -> None:
    lengths = [len(r.samples) for r in records]
    starts = [r.start_sample_index for r in records if r.start_sample_index is not None]
    print("\nWFS summary")
    print("-----------")
    print(f"File                      : {wfs.path}")
    print(f"Decoded records           : {len(wfs.waveforms):,}")
    print(f"Channels                  : {sorted({r.channel for r in wfs.waveforms})}")
    print(f"Selected channel records  : {len(records):,}")
    print(f"Sample rate               : {wfs.sample_rate_hz}")
    print(f"Stream start sample index : {wfs.stream_start_sample_index}")
    print(f"Record length min/max     : {min(lengths):,} / {max(lengths):,}")
    if starts:
        print(f"First record start index  : {starts[0]}")
        print(f"First 8 start deltas      : {np.diff(starts[:9]).tolist()}")
    if wfs.hardware_setup is not None:
        hs = wfs.hardware_setup
        print(f"Hardware ADT              : {hs.adt_code} ({hs.adt_description})")
        print(f"Hardware pretrigger       : {hs.pretrigger_samples}")
        print(f"Hardware raw hex          : {hs.raw_hex}")


def print_csv_summary(window: CsvWindow, folder: Path) -> None:
    print("\nCSV summary")
    print("-----------")
    print(f"Folder                    : {folder}")
    print(f"Samples loaded            : {len(window.voltage_v):,}")
    print(f"Skipped rows              : {window.skipped_rows:,}")
    print(f"Sample interval header    : {window.sample_interval_s}")
    if len(window.time_s) > 1:
        print(f"Sample interval observed  : {np.mean(np.diff(window.time_s[:1000]))}")
    print(f"First time / last time    : {window.time_s[0]} / {window.time_s[-1]}")
    print(f"Files read                : {len(window.files_read)}")
    for path in window.files_read[:5]:
        print(f"  {path}")
    if len(window.files_read) > 5:
        print("  ...")


def build_strategies(
    records: list[WaveformRecord],
    stream_start_sample_index: int | None,
) -> dict[str, np.ndarray]:
    strategies: dict[str, np.ndarray] = {
        "concat_records": reconstruct_concat(records),
        "concat_trim_stream_mod_record_len": reconstruct_concat_with_stream_mod(
            records, stream_start_sample_index
        ),
    }
    for scale in (1.0, 2.0, 4.0):
        positioned = reconstruct_positioned(records, stream_start_sample_index, scale)
        if positioned is not None:
            strategies[f"positioned_pointer_scale_{scale:g}"] = positioned
    return strategies


def write_side_by_side(
    output_path: Path,
    csv_time: np.ndarray,
    csv_signal: np.ndarray,
    wfs_signal: np.ndarray,
    lag: int,
    start_sample: int,
    limit: int,
) -> None:
    csv_view, wfs_view = aligned_views(csv_signal, wfs_signal, lag)
    if lag >= 0:
        csv_start = 0
        wfs_start = lag
    else:
        csv_start = -lag
        wfs_start = 0

    n = min(limit, len(csv_view))
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "row",
                "csv_global_index",
                "wfs_index",
                "csv_time_s",
                "csv_voltage_v",
                "wfs_voltage_v",
                "diff_csv_minus_wfs_v",
                "abs_diff_v",
            ]
        )
        for row in range(n):
            csv_i = csv_start + row
            wfs_i = wfs_start + row
            diff = csv_signal[csv_i] - wfs_signal[wfs_i]
            writer.writerow(
                [
                    row,
                    start_sample + csv_i,
                    wfs_i,
                    csv_time[csv_i],
                    csv_signal[csv_i],
                    wfs_signal[wfs_i],
                    diff,
                    abs(diff),
                ]
            )


def print_side_by_side(
    csv_time: np.ndarray,
    csv_signal: np.ndarray,
    wfs_signal: np.ndarray,
    lag: int,
    start_sample: int,
    rows: int,
) -> None:
    csv_view, _ = aligned_views(csv_signal, wfs_signal, lag)
    if lag >= 0:
        csv_start = 0
        wfs_start = lag
    else:
        csv_start = -lag
        wfs_start = 0

    n = min(rows, len(csv_view))
    print("\nSide-by-side samples for best strategy")
    print("--------------------------------------")
    print("row\tcsv_global\twfs_idx\tcsv_time_s\tcsv_v\twfs_v\tdiff")
    for row in range(n):
        csv_i = csv_start + row
        wfs_i = wfs_start + row
        diff = csv_signal[csv_i] - wfs_signal[wfs_i]
        print(
            f"{row}\t{start_sample + csv_i:,}\t{wfs_i:,}\t"
            f"{csv_time[csv_i]:.9f}\t{csv_signal[csv_i]:.11f}\t"
            f"{wfs_signal[wfs_i]:.11f}\t{diff:.11f}"
        )


def print_rms_scan(
    csv_acc: BinAccumulator,
    wfs_acc: BinAccumulator,
    time_origin_s: float,
    sample_rate_hz: float,
    top_n: int = 20,
) -> None:
    csv_rms = csv_acc.rms()
    wfs_rms = wfs_acc.rms()
    diff = csv_rms - wfs_rms
    ratio = csv_rms / np.maximum(wfs_rms, np.finfo(float).tiny)
    bin_start_times = (
        time_origin_s
        + (csv_acc.start_sample + np.arange(csv_acc.n_bins) * csv_acc.bin_samples)
        / sample_rate_hz
    )
    bin_end_times = bin_start_times + csv_acc.bin_samples / sample_rate_hz

    valid = np.isfinite(csv_rms) & np.isfinite(wfs_rms)
    if not np.any(valid):
        print("No overlapping finite RMS bins to report.")
        return

    print("\nRMS scan summary")
    print("----------------")
    print(f"Bins                     : {csv_acc.n_bins:,}")
    print(f"Bin width                : {csv_acc.bin_samples / sample_rate_hz:.6g} s")
    print(f"CSV RMS min/median/max   : {np.nanmin(csv_rms):.12g} / {np.nanmedian(csv_rms):.12g} / {np.nanmax(csv_rms):.12g}")
    print(f"WFS RMS min/median/max   : {np.nanmin(wfs_rms):.12g} / {np.nanmedian(wfs_rms):.12g} / {np.nanmax(wfs_rms):.12g}")
    print(f"RMS diff max abs         : {np.nanmax(np.abs(diff)):.12g}")

    quiet_order = np.argsort(np.where(valid, csv_rms, np.inf))[:top_n]
    print("\nQuietest CSV RMS bins")
    print("---------------------")
    print("rank\tstart_s\tend_s\tcsv_rms\twfs_rms\tratio\tcsv_min\tcsv_max\twfs_min\twfs_max\tcount")
    for rank, idx in enumerate(quiet_order, start=1):
        print(
            f"{rank}\t{bin_start_times[idx]:.6f}\t{bin_end_times[idx]:.6f}\t"
            f"{csv_rms[idx]:.12g}\t{wfs_rms[idx]:.12g}\t{ratio[idx]:.9g}\t"
            f"{csv_acc.min_value[idx]:.11f}\t{csv_acc.max_value[idx]:.11f}\t"
            f"{wfs_acc.min_value[idx]:.11f}\t{wfs_acc.max_value[idx]:.11f}\t"
            f"{csv_acc.count[idx]:,}"
        )

    mismatch_order = np.argsort(np.where(valid, np.abs(diff), -np.inf))[::-1][:top_n]
    print("\nLargest CSV/WFS RMS disagreements")
    print("---------------------------------")
    print("rank\tstart_s\tend_s\tcsv_rms\twfs_rms\tdiff\tratio\tcount")
    for rank, idx in enumerate(mismatch_order, start=1):
        print(
            f"{rank}\t{bin_start_times[idx]:.6f}\t{bin_end_times[idx]:.6f}\t"
            f"{csv_rms[idx]:.12g}\t{wfs_rms[idx]:.12g}\t{diff[idx]:.12g}\t"
            f"{ratio[idx]:.9g}\t{csv_acc.count[idx]:,}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare AEwin64 WFS decode output to exported AE-stream CSV files."
    )
    parser.add_argument("--wfs", default=DEFAULT_WFS, help="Path to the .wfs file.")
    parser.add_argument(
        "--wfms-dir",
        default=DEFAULT_WFMS_DIR,
        help="Folder containing AEwin64 CSV stream exports.",
    )
    parser.add_argument("--channel", type=int, default=1, help="AE channel to compare.")
    parser.add_argument(
        "--start-sample",
        type=int,
        default=0,
        help="CSV global sample index where comparison starts.",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="CSV time in seconds where comparison starts. Overrides --start-sample.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=200_000,
        help="Number of CSV samples to compare.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration in seconds to compare/scan. Overrides --sample-count.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=512,
        help="Maximum WFS waveform records to decode from the front of the file.",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=8192,
        help="Maximum sample lag to search when aligning WFS to CSV.",
    )
    parser.add_argument(
        "--print-rows",
        type=int,
        default=25,
        help="Number of side-by-side rows to print.",
    )
    parser.add_argument(
        "--side-by-side-out",
        type=Path,
        default=None,
        help="Optional CSV path for the side-by-side table.",
    )
    parser.add_argument(
        "--side-by-side-rows",
        type=int,
        default=10_000,
        help="Rows to write when --side-by-side-out is set.",
    )
    parser.add_argument(
        "--rms-bin-seconds",
        type=float,
        default=None,
        help=(
            "When set, scan the requested window in RMS bins instead of doing "
            "sample-by-sample alignment."
        ),
    )
    parser.add_argument(
        "--rms-top-n",
        type=int,
        default=20,
        help="Number of quiet/disagreement bins to print in RMS scan mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wfs_path = Path(args.wfs)
    wfms_dir = Path(args.wfms_dir)

    if not wfs_path.exists():
        raise FileNotFoundError(f"WFS file not found: {wfs_path}")
    if not wfms_dir.is_dir():
        raise NotADirectoryError(f"CSV folder not found: {wfms_dir}")

    time_origin_s, sample_interval_s = read_csv_time_origin(wfms_dir)
    sample_rate_hz = 1.0 / sample_interval_s
    start_sample = args.start_sample
    sample_count = args.sample_count
    if args.start_time is not None:
        start_sample = int(round((args.start_time - time_origin_s) / sample_interval_s))
    if args.duration is not None:
        sample_count = int(round(args.duration / sample_interval_s))

    if args.rms_bin_seconds is not None:
        bin_samples = max(1, int(round(args.rms_bin_seconds / sample_interval_s)))
        print(
            f"RMS scan window: start_sample={start_sample:,}, "
            f"sample_count={sample_count:,}, bin_samples={bin_samples:,}"
        )
        csv_acc, files_read = scan_csv_rms(wfms_dir, start_sample, sample_count, bin_samples)
        wfs_acc, wfs_meta = scan_wfs_rms(
            wfs_path,
            args.channel,
            start_sample,
            sample_count,
            bin_samples,
            pointer_scale=2.0,
            sample_rate_hz=int(round(sample_rate_hz)),
        )
        wfs_seq_acc, wfs_seq_meta = scan_wfs_sequential_rms(
            wfs_path,
            args.channel,
            start_sample,
            sample_count,
            bin_samples,
            sample_rate_hz=int(round(sample_rate_hz)),
        )
        print(f"CSV files read            : {len(files_read)}")
        if files_read:
            print(f"First CSV file            : {files_read[0]}")
            print(f"Last CSV file             : {files_read[-1]}")
        print("\nWFS window summary")
        print("------------------")
        print(f"Channels seen             : {sorted(wfs_meta.channels)}")
        print(f"Sample rate               : {wfs_meta.sample_rate_hz}")
        print(f"Stream start sample index : {wfs_meta.stream_start_sample_index}")
        print(f"Records scanned/used      : {wfs_meta.records_seen:,} / {wfs_meta.records_used:,}")
        print(f"First/last used start     : {wfs_meta.first_record_start} / {wfs_meta.last_record_start}")
        print(f"Pretrigger samples        : {wfs_meta.pretrigger_samples}")
        print("\nPointer-positioned WFS vs CSV")
        print("============================")
        print_rms_scan(
            csv_acc,
            wfs_acc,
            time_origin_s,
            sample_rate_hz,
            top_n=args.rms_top_n,
        )
        print("\nSequential WFS vs CSV")
        print("=====================")
        print(f"Sequential records scanned/used: {wfs_seq_meta.records_seen:,} / {wfs_seq_meta.records_used:,}")
        print_rms_scan(
            csv_acc,
            wfs_seq_acc,
            time_origin_s,
            sample_rate_hz,
            top_n=args.rms_top_n,
        )
        return

    csv_window = load_csv_window(wfms_dir, start_sample, sample_count)
    print_csv_summary(csv_window, wfms_dir)

    if start_sample > 0:
        wfs_window = decode_wfs_sample_window(
            wfs_path,
            args.channel,
            start_sample,
            sample_count,
            pointer_scale=2.0,
            sample_rate_hz=int(round(sample_rate_hz)),
        )
        print("\nWFS window summary")
        print("------------------")
        print(f"Channels seen             : {sorted(wfs_window.channels)}")
        print(f"Sample rate               : {wfs_window.sample_rate_hz}")
        print(f"Stream start sample index : {wfs_window.stream_start_sample_index}")
        print(f"Records scanned/used      : {wfs_window.records_seen:,} / {wfs_window.records_used:,}")
        print(f"First/last used start     : {wfs_window.first_record_start} / {wfs_window.last_record_start}")
        print(f"NaN samples in WFS window : {np.isnan(wfs_window.signal).sum():,}")
        strategies = {"windowed_positioned_pointer_scale_2": wfs_window.signal}
    else:
        wfs = decode_wfs_prefix(wfs_path, max_records=args.max_records)
        records = channel_records(wfs, args.channel)
        print_wfs_summary(wfs, records)
        strategies = build_strategies(records, wfs.stream_start_sample_index)

    results = [
        best_strategy_result(name, csv_window.voltage_v, signal, args.max_lag)
        for name, signal in strategies.items()
    ]
    results.sort(key=lambda r: (r.mean_abs, r.rmse, -r.within_tol_pct))

    print("\nStrategy alignment scores")
    print("-------------------------")
    print(
        "strategy\tlen\tbest_lag\tcompared\tmean_abs\t"
        "rmse\tmax_abs\twithin_tol_%\tcorr\tfirst_bad"
    )
    for result in results:
        first_bad = "None" if result.first_bad is None else f"{result.first_bad:,}"
        print(
            f"{result.name}\t{len(result.signal):,}\t{result.best_lag:+,}\t"
            f"{result.compared:,}\t{result.mean_abs:.12g}\t"
            f"{result.rmse:.12g}\t{result.max_abs:.12g}\t"
            f"{result.within_tol_pct:.6f}\t{result.corr:.9f}\t{first_bad}"
        )

    best = results[0]
    print(
        f"\nBest strategy: {best.name} with lag {best.best_lag:+,} "
        f"(mean abs diff {best.mean_abs:.12g} V)"
    )
    print_side_by_side(
        csv_window.time_s,
        csv_window.voltage_v,
        best.signal,
        best.best_lag,
        start_sample,
        args.print_rows,
    )

    if args.side_by_side_out is not None:
        write_side_by_side(
            args.side_by_side_out,
            csv_window.time_s,
            csv_window.voltage_v,
            best.signal,
            best.best_lag,
            start_sample,
            args.side_by_side_rows,
        )
        print(f"\nWrote side-by-side table: {args.side_by_side_out}")


if __name__ == "__main__":
    main()
