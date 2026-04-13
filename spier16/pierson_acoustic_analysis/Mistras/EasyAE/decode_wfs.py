"""
AEwin64 .WFS File Decoder
=========================
Decodes binary .WFS (Waveform Stream) files produced by AEwin64 acoustic
emission software into numpy arrays.

File format source: AEwin64 User Manual / AppendixII_Data files (MISTRAS Group Inc.)
Empirically verified against real WFS captures (note: manual uses ID=173/0xAD
but real streaming files use ID=174/0xAE with a slightly different layout).

.WFS Binary Structure
---------------------
All multi-byte integers are little-endian (least significant byte first).

Each "message" in the file is laid out as:
    [LEN : 2 bytes] [MESSAGE BODY : LEN bytes]

The first byte of the body is the Message ID.

Hardware-setup message (ID=174, Sub-ID=42):
    Body layout:
        [0]     ID      = 174 (0xAE)
        [1]     Sub-ID  = 42  (0x2A)
        [2]     MVERS   = 110
        [3]     0x00    (padding)
        [4]     ADT     = 2   (16-bit signed samples)
        [5]     SETS    = number of channel setups
        [6:10]  SRATE   = sample rate in Hz, 4-byte LE uint  (e.g. 1000000 = 1 MSPS)
        ...rest of channel-setup block

Waveform message (ID=174, Sub-ID=1):
    Body layout (28-byte fixed header, then samples to end of message):
        [0]     ID      = 174 (0xAE)
        [1]     Sub-ID  = 1   (0x01)
        [2]     version / MVERS (0x64 = 100, constant)
        [3]     0x00    (padding)
        [4:8]   TOT low 4 bytes (0xAAAAAAAA when not timestamped)
        [8]     CID     = channel number (1-based)
        [9:28]  metadata / padding bytes
        [28:]   N × 2-byte signed int16 samples  (N = (LEN - 28) // 2)

NOTE: The documentation specifies ID=173 (0xAD) and a slightly different header,
but observed streaming files use ID=174 (0xAE) with a 28-byte header and no
explicit N field — samples implicitly fill to the end of the message.
"""

import struct
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Message ID constants (empirical — streaming files use 174, not 173)
# ---------------------------------------------------------------------------
MSG_ID_STREAMING   = 174   # 0xAE — used in real WFS streaming files
MSG_ID_DOCUMENTED  = 173   # 0xAD — as written in the manual (legacy / non-streaming)

SUBID_WAVEFORM  = 1
SUBID_HW_SETUP  = 42
SUBID_STREAM_START = 174

# Fixed header size in waveform messages (empirically determined)
# Samples begin at byte 28 and run to the end of the message body.
WAVEFORM_HEADER_BYTES = 28

# Offset of CID (channel ID) inside the 28-byte waveform header body
CID_OFFSET = 8

# Reverse engineered from AEwin64 ASCII waveform exports:
# exported_volts = raw_adc_counts / 16384
VOLTS_PER_COUNT = 1.0 / 16384.0


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class WaveformRecord:
    """One captured waveform burst."""
    channel: int                   # AE channel number (CID), 1-based
    samples: np.ndarray            # float64 numpy array of volts
    sample_rate_hz: Optional[int] = None  # filled from hardware-setup message
    pretrigger_samples: Optional[int] = None  # negative when capture includes pretrigger
    start_sample_index: Optional[int] = None  # absolute stream sample index, when present

    @property
    def time_axis_s(self) -> Optional[np.ndarray]:
        """Time axis for each sample in seconds, or None if rate unknown."""
        if self.sample_rate_hz is None:
            return None
        dt = 1.0 / self.sample_rate_hz
        start = (self.pretrigger_samples or 0) * dt
        return start + np.arange(len(self.samples)) * dt


@dataclass
class HardwareSetup:
    """
    Parsed hardware-setup metadata from a WFS message (ID=174, Sub-ID=42).

    The first 10 bytes are understood from observed files:
      [0]  message ID
      [1]  sub-ID
      [2]  message version
      [3]  padding
      [4]  ADT/sample-format code
      [5]  number of channel setup blocks
      [6:10] sample rate in Hz (uint32 LE)

    The remaining bytes vary by system/configuration. We preserve them both as
    raw bytes and as unpacked little-endian words so they can be reverse
    engineered from real captures without losing information.
    """
    message_id: int
    sub_id: int
    message_version: int
    adt_code: int
    n_channel_setups: int
    sample_rate_hz: Optional[int]
    pretrigger_samples: Optional[int] = None
    raw_body: bytes = b""
    extra_bytes: bytes = b""
    extra_u16_le: List[int] = field(default_factory=list)
    extra_i16_le: List[int] = field(default_factory=list)
    extra_u32_le: List[int] = field(default_factory=list)
    extra_i32_le: List[int] = field(default_factory=list)

    @property
    def adt_description(self) -> str:
        """Human-readable best-effort description of the sample format code."""
        return {
            1: "8-bit samples",
            2: "16-bit signed samples",
            3: "32-bit samples",
        }.get(self.adt_code, f"unknown format code {self.adt_code}")

    @property
    def pretrigger_seconds(self) -> Optional[float]:
        """Pretrigger interval in seconds when both rate and offset are known."""
        if self.sample_rate_hz in (None, 0) or self.pretrigger_samples is None:
            return None
        return abs(self.pretrigger_samples) / self.sample_rate_hz

    @property
    def raw_hex(self) -> str:
        """Full hardware-setup message body as a hex string."""
        return self.raw_body.hex(" ")

    @property
    def extra_hex(self) -> str:
        """Hardware-setup bytes after the known 10-byte prefix as hex."""
        return self.extra_bytes.hex(" ")


@dataclass
class WFSFile:
    """Decoded contents of a .WFS file."""
    path: Path
    sample_rate_hz: Optional[int] = None
    hardware_setup: Optional[HardwareSetup] = None
    stream_start_sample_index: Optional[int] = None
    waveforms: List[WaveformRecord] = field(default_factory=list)

    def to_array(self, channel: Optional[int] = None) -> np.ndarray:
        """
        Stack all waveforms (optionally filtered by channel) into a 2-D
        numpy array of shape (n_records, n_samples).

        All records have the same length in streaming-mode files, so no
        zero-padding is needed. Mixed-length files are zero-padded to the
        longest record.

        Parameters
        ----------
        channel : int, optional
            If given, only include records from this AE channel number.

        Returns
        -------
        np.ndarray  shape (n_records, max_samples), dtype float64
        """
        records = self.waveforms
        if channel is not None:
            records = [r for r in records if r.channel == channel]
        if not records:
            return np.empty((0, 0), dtype=np.float64)
        max_len = max(len(r.samples) for r in records)
        out = np.zeros((len(records), max_len), dtype=np.float64)
        for i, rec in enumerate(records):
            out[i, : len(rec.samples)] = rec.samples
        return out

    def waveform_time_axis_s(self, channel: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Time axis for samples within each waveform record.

        This returns a single 1-D axis because WFS streaming captures use the
        same sampling interval and pretrigger offset for each waveform record.
        Mixed-length files are padded out to the longest record, matching
        :meth:`to_array`.
        """
        records = self.waveforms
        if channel is not None:
            records = [r for r in records if r.channel == channel]
        if not records or records[0].sample_rate_hz is None:
            return None

        max_len = max(len(r.samples) for r in records)
        dt = 1.0 / records[0].sample_rate_hz
        start = (records[0].pretrigger_samples or 0) * dt
        return start + np.arange(max_len) * dt

    def channels(self) -> List[int]:
        """Unique channel numbers present in this file."""
        return sorted({r.channel for r in self.waveforms})


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_sample_rate(body: bytes) -> Optional[int]:
    """
    Extract sample rate in Hz from a hardware-setup message body
    (ID=174, Sub-ID=42).

    The sample rate is stored as a 4-byte little-endian uint at body[6:10],
    in units of Hz (e.g. 1000000 = 1 MSPS).
    """
    if len(body) < 10:
        return None

    # Observed WFS files contain two rate-related fields:
    #   body[6:10]  = 1_000_000 in both tested captures
    #   body[12:16] = 1000 for a 1 MSPS file, 500 for a 500 kSPS file
    #
    # The latter matches AEwin64's actual export timing when interpreted as kHz.
    if len(body) >= 16:
        khz_rate = struct.unpack_from("<I", body, 12)[0]
        if khz_rate > 0:
            return khz_rate * 1000

    srate = struct.unpack_from("<I", body, 6)[0]
    return srate if srate > 0 else None


def _parse_hardware_setup(body: bytes) -> HardwareSetup:
    """
    Parse a hardware-setup message body (ID=174/173, Sub-ID=42).

    Only the first 10 bytes are currently understood with confidence.
    Remaining bytes are preserved as raw values and unpacked LE words.
    """
    extra_bytes = body[10:] if len(body) > 10 else b""

    extra_u16_le = [
        struct.unpack_from("<H", extra_bytes, i)[0]
        for i in range(0, len(extra_bytes) - 1, 2)
    ]
    extra_i16_le = [
        struct.unpack_from("<h", extra_bytes, i)[0]
        for i in range(0, len(extra_bytes) - 1, 2)
    ]
    extra_u32_le = [
        struct.unpack_from("<I", extra_bytes, i)[0]
        for i in range(0, len(extra_bytes) - 3, 4)
    ]
    extra_i32_le = [
        struct.unpack_from("<i", extra_bytes, i)[0]
        for i in range(0, len(extra_bytes) - 3, 4)
    ]

    return HardwareSetup(
        message_id=body[0] if len(body) > 0 else 0,
        sub_id=body[1] if len(body) > 1 else 0,
        message_version=body[2] if len(body) > 2 else 0,
        adt_code=body[4] if len(body) > 4 else 0,
        n_channel_setups=body[5] if len(body) > 5 else 0,
        sample_rate_hz=_parse_sample_rate(body),
        pretrigger_samples=(
            struct.unpack_from("<i", body, 18)[0] if len(body) >= 22 else None
        ),
        raw_body=bytes(body),
        extra_bytes=extra_bytes,
        extra_u16_le=extra_u16_le,
        extra_i16_le=extra_i16_le,
        extra_u32_le=extra_u32_le,
        extra_i32_le=extra_i32_le,
    )


def _parse_stream_start_sample_index(body: bytes) -> Optional[int]:
    """
    Extract the absolute stream-start sample pointer from the
    ID=174, Sub-ID=174 message that precedes waveform data in streaming files.

    Across known AEwin64 captures, body[13:17] is the exact absolute sample
    index used by the ASCII waveform exporter as its start position in the
    stored stream. The corresponding within-record trim is:

        stream_start_sample_index % samples_per_record
    """
    if len(body) < 17:
        return None
    return struct.unpack_from("<I", body, 13)[0]


def _parse_waveform_start_sample_index(body: bytes) -> Optional[int]:
    """
    Extract the absolute stream sample pointer for one waveform record.

    In observed multi-channel WFS files, bytes 24:28 contain a per-record
    pointer that advances by 2048 counts while each waveform contains 4096
    int16 samples. Multiplying by 2 converts the stored counter into the
    same sample-index units used by the stream-start message.
    """
    if len(body) < WAVEFORM_HEADER_BYTES:
        return None
    return 2 * struct.unpack_from("<I", body, 24)[0]


def decode_wfs(path: str | Path,
               max_records: Optional[int] = None) -> WFSFile:
    """
    Parse a .WFS streaming file and return a :class:`WFSFile` with all
    waveform records decoded as voltage-valued numpy arrays.

    Parameters
    ----------
    path : str or Path
        Path to the .wfs file.
    max_records : int, optional
        Stop after reading this many waveform records (useful for large files
        during development/testing).

    Returns
    -------
    WFSFile
        Decoded file object.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    data = path.read_bytes()

    result = WFSFile(path=path)
    pos = 0
    total = len(data)
    n_waveforms = 0

    while pos + 2 <= total:
        if max_records is not None and n_waveforms >= max_records:
            break

        msg_len = struct.unpack_from("<H", data, pos)[0]
        pos += 2

        if msg_len == 0:
            continue  # zero-length padding message

        if pos + msg_len > total:
            break     # truncated file — stop gracefully

        body = data[pos : pos + msg_len]
        pos += msg_len

        if len(body) < 2:
            continue

        msg_id  = body[0]
        sub_id  = body[1]

        # ------------------------------------------------------------------ #
        #  Hardware setup → extract sample rate                               #
        # ------------------------------------------------------------------ #
        if sub_id == SUBID_HW_SETUP and msg_id in (MSG_ID_STREAMING, MSG_ID_DOCUMENTED):
            result.hardware_setup = _parse_hardware_setup(body)
            srate = result.hardware_setup.sample_rate_hz
            if srate is not None:
                result.sample_rate_hz = srate
            continue

        # ------------------------------------------------------------------ #
        #  Stream-start pointer used by AEwin64 ASCII waveform export         #
        # ------------------------------------------------------------------ #
        if sub_id == SUBID_STREAM_START and msg_id in (MSG_ID_STREAMING, MSG_ID_DOCUMENTED):
            result.stream_start_sample_index = _parse_stream_start_sample_index(body)
            continue

        # ------------------------------------------------------------------ #
        #  Waveform data                                                       #
        # ------------------------------------------------------------------ #
        if sub_id == SUBID_WAVEFORM and msg_id in (MSG_ID_STREAMING, MSG_ID_DOCUMENTED):
            if msg_len <= WAVEFORM_HEADER_BYTES:
                continue  # no samples in this message

            cid = body[CID_OFFSET] if len(body) > CID_OFFSET else 0

            n_samples = (msg_len - WAVEFORM_HEADER_BYTES) // 2
            sample_bytes = body[WAVEFORM_HEADER_BYTES : WAVEFORM_HEADER_BYTES + n_samples * 2]
            raw_counts = np.frombuffer(sample_bytes, dtype="<i2")
            samples = raw_counts.astype(np.float64) * VOLTS_PER_COUNT
            pretrigger_samples = (
                result.hardware_setup.pretrigger_samples
                if result.hardware_setup is not None else None
            )
            start_sample_index = _parse_waveform_start_sample_index(body)

            result.waveforms.append(WaveformRecord(
                channel=cid,
                samples=samples,
                sample_rate_hz=result.sample_rate_hz,
                pretrigger_samples=pretrigger_samples,
                start_sample_index=start_sample_index,
            ))
            n_waveforms += 1

    # Back-fill metadata on any records parsed before the hw-setup message.
    pretrigger_samples = (
        result.hardware_setup.pretrigger_samples
        if result.hardware_setup is not None else None
    )
    if result.sample_rate_hz is not None or pretrigger_samples is not None:
        for rec in result.waveforms:
            if rec.sample_rate_hz is None:
                rec.sample_rate_hz = result.sample_rate_hz
            if rec.pretrigger_samples is None:
                rec.pretrigger_samples = pretrigger_samples

    return result


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def load_continuous(path: str | Path,
                    channel: int = 1,
                    max_records: Optional[int] = None,
                    sample_rate_hz: Optional[int] = None):
    """
    Decode a .WFS file and concatenate all records into a single continuous
    time-series suitable for spectral analysis.

    Parameters
    ----------
    path : str or Path
        Path to the .wfs file.
    channel : int
        Only include records from this AE channel number.
    max_records : int, optional
        Stop after reading this many waveform records.
    sample_rate_hz : int, optional
        Override the sample rate found in the file (Hz).

    Returns
    -------
    samples : np.ndarray  float64, shape (N,)
        Concatenated waveform samples in volts.
    time : np.ndarray  float64, shape (N,)
        Time axis in seconds. When the file contains a pretrigger window,
        the first sample starts at a negative time offset.
    sample_rate_hz : int
        Sample rate used (either read from the file or the override value).

    Raises
    ------
    SystemExit
        If no waveform records are found, or the requested channel is absent.
    """
    import sys

    wfs = decode_wfs(path, max_records=max_records)

    if not wfs.waveforms:
        sys.exit(f"No waveform records found in {path}")

    sr = sample_rate_hz or wfs.sample_rate_hz
    if sr is None:
        print("WARNING: sample rate not found in hardware-setup message — "
              "defaulting to 1,000,000 Hz. Pass sample_rate_hz to override.")
        sr = 1_000_000

    records = wfs.waveforms
    records = [r for r in records if r.channel == channel]
    if not records:
        sys.exit(f"No records found for channel {channel} in {path}")

    print(f"Loaded {len(records):,} records  |  sample rate: {sr:,} Hz"
          f"  |  samples per record: {len(records[0].samples):,}")

    pretrigger_samples = records[0].pretrigger_samples or 0

    have_record_positions = (
        wfs.stream_start_sample_index is not None
        and all(r.start_sample_index is not None for r in records)
    )
    if have_record_positions:
        front_trim = 0
        last_end = 0
        spans = []
        for rec in records:
            start = rec.start_sample_index - wfs.stream_start_sample_index
            src_start = max(0, -start)
            placed_start = max(0, start)

            overlap = max(0, last_end - placed_start)
            src_start += overlap
            placed_start += overlap

            placed_len = len(rec.samples) - src_start
            placed_end = placed_start + max(0, placed_len)
            spans.append((rec, src_start, placed_start, placed_end))
            last_end = max(last_end, placed_end)
            if start < 0:
                front_trim = max(front_trim, -start)

        total_len = max(end for _, _, _, end in spans)
        raw = np.zeros(total_len, dtype=np.float64)
        for rec, src_start, placed_start, placed_end in spans:
            if src_start >= len(rec.samples):
                continue
            raw[placed_start:placed_end] = rec.samples[src_start:]

        lead_gap = max(0, records[0].start_sample_index - wfs.stream_start_sample_index)
        if lead_gap:
            print(f"Inserted leading gap before first record: {lead_gap:,} samples")
        if front_trim:
            print(f"Trimmed leading samples before stream start: {front_trim:,} samples")
        print("Reconstructed continuous stream using waveform start pointers")
    else:
        raw = np.concatenate([r.samples for r in records]).astype(np.float64, copy=False)
        if wfs.stream_start_sample_index is not None and len(records[0].samples) > 0:
            start_offset = wfs.stream_start_sample_index % len(records[0].samples)
            if start_offset:
                raw = raw[start_offset:]
                print(f"Applied AEwin64 stream start offset: {start_offset:,} samples")
    t = np.linspace(pretrigger_samples / sr, (pretrigger_samples + len(raw) - 1) / sr, len(raw))

    print(f"Concatenated file records: {len(raw):,} samples")

    return raw, t, sr


def wfs_to_numpy(path: str | Path,
                 channel: Optional[int] = None,
                 max_records: Optional[int] = None,
                 return_time_axis: bool = False):
    """
    One-liner: decode a .WFS file and return waveform samples in volts.

    Shape: (n_waveforms, n_samples_per_waveform).

    Parameters
    ----------
    path : str or Path
        Path to the .wfs file.
    channel : int, optional
        If given, only return waveforms from that AE channel.
    max_records : int, optional
        Limit number of records read (useful for large files).
    return_time_axis : bool, optional
        When True, also return the per-waveform time axis in seconds. This
        axis includes the pretrigger offset when available.

    Returns
    -------
    np.ndarray  shape (n_waveforms, n_samples), dtype float64
        Waveform amplitudes in volts.
    np.ndarray  shape (n_samples,), dtype float64
        Returned only when ``return_time_axis=True``.
    """
    wfs = decode_wfs(path, max_records=max_records)
    samples = wfs.to_array(channel=channel)
    if return_time_axis:
        return samples, wfs.waveform_time_axis_s(channel=channel)
    return samples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Decode a .WFS streaming file and print summary info."
    )
    parser.add_argument("wfs_file", help="Path to the .wfs file")
    parser.add_argument(
        "--channel", type=int, default=None,
        help="Only show waveforms from this channel number"
    )
    parser.add_argument(
        "--max-records", type=int, default=None,
        help="Stop after reading this many waveform records"
    )
    args = parser.parse_args()

    wfs = decode_wfs(args.wfs_file, max_records=args.max_records)

    print(f"File          : {wfs.path.name}")
    if wfs.sample_rate_hz:
        print(f"Sample rate   : {wfs.sample_rate_hz:,} Hz  ({wfs.sample_rate_hz/1e6:.3f} MSPS)")
    else:
        print("Sample rate   : unknown (no hardware-setup message found)")
    if wfs.hardware_setup is not None:
        hs = wfs.hardware_setup
        print(f"Sample format : ADT={hs.adt_code}  ({hs.adt_description})")
        print(f"Setup blocks  : {hs.n_channel_setups}")
        print(f"HW msg ver    : {hs.message_version}")
        if hs.pretrigger_samples is not None:
            print(f"Pretrigger    : {hs.pretrigger_samples} samples")
            if hs.pretrigger_seconds is not None:
                print(f"Pretrigger s  : {hs.pretrigger_seconds:.8f} s")
        if hs.extra_bytes:
            print(f"HW extra hex  : {hs.extra_hex}")
            print(f"HW extra u16  : {hs.extra_u16_le}")
            print(f"HW extra i16  : {hs.extra_i16_le}")
            print(f"HW extra u32  : {hs.extra_u32_le}")
            print(f"HW extra i32  : {hs.extra_i32_le}")
    print(f"Total records : {len(wfs.waveforms):,}")
    print(f"Channels      : {wfs.channels()}")

    records = wfs.waveforms
    if args.channel is not None:
        records = [r for r in records if r.channel == args.channel]
        print(f"Records ch {args.channel} : {len(records):,}")
    print()

    for i, rec in enumerate(records[:5]):
        print(f"  [{i}] ch={rec.channel}  n={len(rec.samples):,} samples"
              f"  min={rec.samples.min():.8f} V  max={rec.samples.max():.8f} V"
              f"  std={rec.samples.std():.8f} V")

    if len(records) > 5:
        print(f"  ... ({len(records) - 5:,} more records)")

    arr = wfs.to_array(channel=args.channel)
    print(f"\nnumpy array shape : {arr.shape}  dtype={arr.dtype}")
    print("Done.")
