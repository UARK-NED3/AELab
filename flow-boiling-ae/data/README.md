# Data Access

Do not commit large raw datasets here.

Use this folder for OSF links, dataset manifests, file naming notes, calibration metadata, and
small example CSV files needed for tests or tutorials.

The file types produced by a run are described in
[5. Raw Data Products](../docs/05-raw-data-products.md).

## Per-Run Upload Checklist

A run is only reusable if all three acquisition systems are archived together with the
information needed to align them.

- [ ] LabVIEW `.lvm`, with its header intact (`Delta_X` and the absolute start time).
- [ ] EasyAE `.DTA` and `.wfs`.
- [ ] EasyAE ASCII exports: `Hit.TXT` and `Time.TXT`.
- [ ] EasyAE streamed waveform CSV segments, complete, with the trailing sample index preserved
      in every file name.
- [ ] PSCS heater log CSV, with its `StartTime` and `Sampling` header rows.
- [ ] The AEwin layout file (`.lay64`) used for the run, or a record of any deviation from the
      settings in [3.1](../docs/03-easyae-acquisition.md#31-acquisition-settings-in-the-stored-layout).

## Manifest Fields

Record these for each run, alongside the standard fields in
[`CONTRIBUTING.md`](../../CONTRIBUTING.md#data-policy):

| Field | Note |
| --- | --- |
| Run ID | Matching the folder and file names, e.g. `FL-33` |
| Date | Acquisition date |
| Working fluid | DI water unless otherwise stated |
| Heat sink sample | Required, because the heat loss coefficients are sample-specific |
| Inlet setpoint | °C |
| Flow rate | L/min |
| Heater program | Steady-state level, or the PSCS External Timed Program steps for a transient |
| AE threshold and preamp gain | dB, from the AEwin layout |
| Deviations | Any departure from the SOP, including the reason |

## Calibration Metadata to Keep With the Data

- Pressure transducer offset (`P Offset (kPa)` on the LabVIEW front panel) as used for the run.
- Flow meter K-factor if it differs from 22,000 pulses/litre.
- Heat loss coefficients `a` and `b`, and the sample they were determined on.
- AE sensor coupling condition and mounting location.

Document the expected download location or script-based download step for each dataset.
