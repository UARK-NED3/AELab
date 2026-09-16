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

## Documented Configurations

| Experiment | Mounting | Threshold | Preamp | Analog filter | Sample rate | PDT / HDT / HLT | Reference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Flow boiling, ENRC 3414 microchannel loop | MHR15A magnetic hold-down on a magnetic sticker, on the test-section housing wall parallel to the heat sink | 31 dB fixed | 26 dB, passive | 5 kHz – 100 kHz | 1 MSPS | 50 / 200 / 300 µs | [flow-boiling-ae/docs/03](../flow-boiling-ae/docs/03-easyae-acquisition.md#31-acquisition-settings-in-the-stored-layout) |

The flow boiling SOP also documents the full acquisition and ASCII export workflow
([3.2–3.4](../flow-boiling-ae/docs/03-easyae-acquisition.md#32-acquiring-ae-data)), the AE
parameter definitions and their export column codes
([7](../flow-boiling-ae/docs/07-ae-features.md)), and the structure of the `.DTA`-derived hit and
time-driven ASCII files and the `.wfs`-derived waveform CSV segments
([5](../flow-boiling-ae/docs/05-raw-data-products.md)). Those pages apply to any EasyAE
acquisition in the lab, not only to flow boiling.

## Lab Notes to Add

- Sensor mounting method and coupling material for the pool boiling and partial discharge setups.
- Calibration or pencil-lead break procedures used before experiments.
- Definitions of the `FRQ-C` (frequency centroid) and `P-FRQ` (peak frequency) spectrum features
  from the AEWin documentation — see
  [flow-boiling-ae/docs/07.2](../flow-boiling-ae/docs/07-ae-features.md#72-export-column-mapping).

## Related Repository Areas

- `pd-ae/`
- `pd-immersion-ae/`
- `pool-boiling-ae/`
- `flow-boiling-ae/`
- `spier16/Mistras/EasyAE/`
