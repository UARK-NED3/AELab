# Hydrophone and NI DAQ System

## Components

| Role | Hardware or Software | Notes |
| --- | --- | --- |
| Hydrophone | High Tech HTI-96-MIN | Miniature hydrophone used for underwater or immersed acoustic measurements. |
| DAQ | NI 9230 C Series sound and vibration input module | Three-channel input module for IEPE and non-IEPE sensors. |
| Software | NI LabVIEW | Used for data acquisition and lab-specific logging workflows. |

## Manufacturer Notes

The [High Tech HTI-96-MIN product page](https://www.hightechincusa.com/products/hydrophones/hti96min.html) lists the HTI-96-Min hydrophone series with a 2 Hz to 30 kHz frequency response. The same page lists sensitivity values for versions with and without preamplifiers, including -201 dB re 1 V/uPa without preamplifier and configurable preamplified versions.

The [NI 9230 product page](https://www.ni.com/en/shop/hardware/sound-and-vibration/model-ni-9230) describes the NI 9230 as a 3-channel, 12.8 kS/s/channel, +/-30 V C Series sound and vibration input module. NI notes that it can measure IEPE and non-IEPE sensors, supports software-selectable AC/DC coupling, includes IEPE open/short detection and IEPE signal conditioning, and simultaneously measures input channels.

## Lab Notes to Add

- Hydrophone model variant, preamplifier configuration, and calibration sheet location.
- NI chassis or carrier used with the NI 9230.
- LabVIEW VI name, version, and saved data format.
- Sampling rate, voltage range, coupling mode, and channel mapping.
- Conversion from voltage to acoustic pressure for each calibration configuration.

## Related Repository Areas

- `spier16/Hydrophones/`
- `pd-immersion-ae/`
- `pool-boiling-ae/`
- `flow-boiling-ae/`
