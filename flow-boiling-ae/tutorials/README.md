# Tutorials

Tutorial notebooks for flow boiling acoustic emission analysis.

Tutorials should be runnable from Colab, Jupyter, MATLAB Online, or another clearly documented
platform. Include setup cells, OSF data access instructions, and at least one output that
confirms the workflow ran correctly.

## Suggested Tutorials

Each maps to a section of the [SOP](../README.md#standard-operating-procedure), so a student can
read the procedure and then run the corresponding notebook.

| Tutorial | Goal | SOP reference |
| --- | --- | --- |
| Reading a run | Load the `.lvm`, `Hit.TXT`, `Time.TXT`, and PSCS log for one run and plot each raw signal | [5](../docs/05-raw-data-products.md) |
| Flow rate from pulses | Recover `Q_LPM` from the flow sensor pulse train and compare against the LabVIEW front-panel readout | [6.2](../docs/06-data-reduction.md#62-flow-rate-from-the-pulse-train) |
| Aligning heater power | Map the PSCS log onto the flow-loop timeline and reproduce the applied power trace | [6.3](../docs/06-data-reduction.md#63-heater-power-supply-time-alignment) |
| Heat split and quality | Compute `Q_loss`, `Q_hf`, `Q_sensible`, `Q_boil`, and `x_out`, including the subcooled-outlet screen | [6.5](../docs/06-data-reduction.md#65-thermal-characteristics), [6.6](../docs/06-data-reduction.md#66-outlet-vapor-quality) |
| AE hit features | Load the hit table, apply the column mapping, and plot features against heater power and baseplate temperature | [7](../docs/07-ae-features.md) |
| AE waveform spectrogram | Concatenate the streamed CSV segments and produce a spectrogram alongside the thermal record | [5.5](../docs/05-raw-data-products.md#55-easyae-streamed-waveform-csv), [7.3](../docs/07-ae-features.md#73-representative-results) |

Start from the notebooks in `spier16/Mistras/EasyAE/` rather than from scratch where one already
covers the step.
