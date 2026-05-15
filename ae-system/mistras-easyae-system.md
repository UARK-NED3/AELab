# MISTRAS EasyAE Acoustic Emission System

## Components

| Role | Hardware or Software | Notes |
| --- | --- | --- |
| AE sensor | MISTRAS R3a low-frequency AE sensor | Passive AE sensor with a 30 kHz resonant response. |
| DAQ | MISTRAS EasyAE | Two-channel AE data acquisition and digital signal processing system. |
| Software | MISTRAS AEWin | Used to acquire and read back AE hit data and waveforms from the EasyAE system. |

## Manufacturer Notes

The [MISTRAS R3a product page](https://www.physicalacoustics.com/by-product/sensors/R3a-30-kHz-Low-Frequency-AE-Sensor) describes the R3a as a rugged low-frequency AE sensor with a machined stainless-steel cavity, ceramic face electrical isolation, SMA connector, and 30 kHz resonant response.

The [MISTRAS EasyAE product page](https://www.physicalacoustics.com/by-product/small-systems/easy-ae/) describes EasyAE as a compact two-channel AE DAQ and digital signal processing system using USB-C communication. It supports waveform streaming, AE feature extraction, AE signal processing, and waveform-based acquisition. The page also notes that AE hit data and waveforms are recorded and read back using AEWin control software.

## Lab Notes to Add

- Sensor mounting method and coupling material.
- Preamplifier settings, if used.
- AEWin acquisition settings, including threshold, sampling rate, hit definition time, hit lockout time, and peak definition time.
- File formats exported from AEWin and where matching analysis notebooks are stored.
- Calibration or pencil-lead break procedures used before experiments.

## Related Repository Areas

- `pd-ae/`
- `pd-immersion-ae/`
- `pool-boiling-ae/`
- `flow-boiling-ae/`
- `spier16/Mistras/EasyAE/`
