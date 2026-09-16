# 8. Equipment and Part List

Parts used in the ENRC 3414 microchannel two-phase flow loop with AE sensing.

---

## 8.1 Instrumentation and Data Acquisition

| Part | Manufacturer | Part number |
| --- | --- | --- |
| Thermocouple (T-type) | DwyerOmega | TJ36-CPSS-116G-3-SB |
| Pressure transducer | DwyerOmega | PX2300-1BDI |
| Flowrate sensor | DwyerOmega | BV1000TRN025B |
| NI DAQ card | National Instruments | 9129, 9239 |
| NI cDAQ chassis | National Instruments | 9178 |
| AE DAQ | Physical Acoustics (MISTRAS) | EasyAE, Model 1288-5015 |
| Acoustic sensor | Physical Acoustics (MISTRAS) | R3a, 30 kHz low-frequency AE sensor |
| Magnetic hold-down | Physical Acoustics (MISTRAS) | MHR15A |
| Magnetic sticker | SALEX | X002VQXSR9 |
| Single-board computer | Raspberry Pi Foundation | Raspberry Pi 4 Model B |
| DAQ HAT | Measurement Computing Corporation | MCC 172 |
| Accelerometer (10 kHz) | PCB Piezotronics | TLD352A56 |
| Accelerometer (30 kHz) | PCB Piezotronics | 621C40 |

> See the open item in
> [1.5](01-facility-and-safety.md#15-signal-acquisition-architecture) about the NI 9129 / NI 9219
> discrepancy in the source document. Confirm against NI MAX on the acquisition PC.

## 8.2 Thermal and Fluid Handling

| Part | Manufacturer | Part number |
| --- | --- | --- |
| Pump | US Solar Pumps | D5 Solar Pump |
| Heater | *(not recorded in the source document)* | — |
| Heater power supply unit | BK Precision | 1685B |
| Pump power supply unit | BK Precision | 1550 |
| In-line heater | Watlow | FLC-178 |
| Heat sink | Custom machined | Aluminum |
| External chiller | TEYU S&A | CW-5200 |
| Organic material filter | Pentair | GS-10 |
| Fluid reservoir | Wilmad-LabGlass | LG-8079C-104 cylindrical jacketed reaction vessel, O-ring flange, 2 L |

## 8.3 Piping and Fittings

| Part | Manufacturer | Part number |
| --- | --- | --- |
| Tubing | McMaster | 89895K722 |
| 90° tube elbow | Swagelok | S-400-9 |
| Tee connector | Swagelok | SS-400-3-4-2 |
| Male NPT connector | Swagelok | S-400-1-4 |
| Metering valve | Swagelok | SS-SS4-VH |
| Pneumatic flow control valve | Zoro | 5TUL2 |
| Teflon tape | VOTMELL | 4 rolls, 1/2 in. (W) × 520 in. (L) |

## 8.4 Gaps to Close

The following entries are incomplete in the source document and should be filled in from the lab
records rather than inferred:

- Test-section ceramic heater manufacturer and part number. The facility description refers to it
  as "Ceramic Heater (Model Y)".
- Heat sink drawing number and channel geometry (channel width, depth, count, fin thickness).
  Only the overall heated footprint is documented, in
  [6.1](06-data-reduction.md#61-constants).
- Polycarbonate cover plate and PEEK layer sources.
