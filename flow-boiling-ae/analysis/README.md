# Analysis Code

Reusable flow boiling acoustic emission analysis code.

The processing steps that this folder should implement are specified in the SOP:

- Parsing: [5. Raw Data Products](../docs/05-raw-data-products.md)
- Reduction: [6. Data Reduction](../docs/06-data-reduction.md)
- AE features: [7. Acoustic Emission Features](../docs/07-ae-features.md)

## Expected Contents

| Component | Input | Output |
| --- | --- | --- |
| LabVIEW `.lvm` reader | `.lvm` | Time series with `Delta_X` and absolute start time read from the header |
| EasyAE hit reader | `Hit.TXT` | Hit table with the column mapping in [7.2](../docs/07-ae-features.md#72-export-column-mapping) |
| EasyAE time-driven reader | `Time.TXT` | Continuous RMS, THR, ABS-ENERGY, ASL record |
| Waveform segment loader | `STREAM*_<ch>_<start>.csv` | Concatenated waveform with sample interval and pre-amp gain read from the information header |
| PSCS log reader | `LOG-*.csv` | Heater voltage and current with `StartTime` and `Sampling` from the header |
| Time alignment | LabVIEW + PSCS records | `Q_load` on the flow-loop timeline, per [6.3](../docs/06-data-reduction.md#63-heater-power-supply-time-alignment) |
| Flow rate | `.lvm` pulse column | `Q_LPM`, per [6.2](../docs/06-data-reduction.md#62-flow-rate-from-the-pulse-train) |
| Thermal reduction | Aligned records | `Q_loss`, `Q_hf`, `q_base`, `m_dot`, `Q_sensible`, `Q_boil`, `R_th`, `R_eff`, `x_out` |
| Spectrograms | `dP` and AE waveform | Time–frequency maps with window length, hop, and window function recorded |

## Conventions

- Read acquisition parameters from file headers. Do not hard-code the sample interval, pre-amp
  gain, or K-factor where the file carries them.
- Keep the empirical heat loss coefficients `a` and `b` as named, documented inputs, not
  literals. They are valid only for the aluminum straight microchannel heat sink sample — see
  [6.5](../docs/06-data-reduction.md#65-thermal-characteristics).
- Carry units explicitly in variable names or in a units layer, and state the constant-property
  assumption wherever reduced heat split values are produced.
- Mask rather than plot the divergences: `R_th` as `Q_hf` approaches zero, and `x_out` where the
  outlet is subcooled.
- Keep personal paths, raw data, and generated result folders out of version control.

## Existing Code

`spier16/Mistras/EasyAE/` in this repository already contains EasyAE notebooks and `.wfs`
decoding utilities. Check there before writing a new parser, and migrate rather than duplicate.
