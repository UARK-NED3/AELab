# AELab

Research toolkits for acoustic emission sensing for phase change processes, flows, and partial discharge.

This repository is organized as a collaborative workspace for analysis code, tutorial notebooks, and reproducible examples. Large experimental datasets should be stored externally, such as on OSF, and referenced from notebooks or scripts by stable download links.

## Project Areas

| Folder | Focus |
| --- | --- |
| `pd-ae/` | Acoustic emission analysis for partial discharge experiments. |
| `pd-immersion-ae/` | Acoustic emission analysis for immersed partial discharge experiments. |
| `pool-boiling-ae/` | Acoustic emission analysis for pool boiling experiments. |
| `flow-boiling-ae/` | Acoustic emission analysis for flow boiling experiments. |
| `ae-system/` | Sensor, data acquisition, and software notes for acoustic sensing systems used in the lab. |
| `spier16/` | Existing notebooks, scripts, and reference materials from the initial repository. |

## Suggested Folder Pattern

Each project area follows the same lightweight structure:

- `analysis/`: reusable scripts, packages, helper functions, and processing workflows.
- `tutorials/`: Colab, Jupyter, MATLAB Live Script, or other tutorial notebooks.
- `data/`: small metadata files and instructions for accessing external datasets.

Avoid committing large raw data files, generated result archives, or local environment folders. Instead, document the dataset source, OSF link, expected file names, and any preprocessing steps needed to reproduce the analysis.

## Student Contribution Checklist

Before opening a pull request, please check that:

- The contribution belongs in the correct project folder.
- New notebooks can run from a fresh runtime or clearly list the required setup.
- Dataset access is documented with a public or lab-approved OSF link.
- Paths are relative or configurable, not hard-coded to a personal computer.
- Outputs, figures, and generated files are only committed when they are useful for review.
- The README in the project folder has been updated with the new analysis or tutorial.

## Tutorial Expectations

Tutorial notebooks should help a new student reproduce the main result without private local files. A good tutorial includes:

- A short goal statement.
- Package installation or import steps.
- OSF data download or mounting instructions.
- A minimal analysis workflow.
- One or more verification plots, tables, or summary metrics.
- Citations or notes for any datasets, papers, or external code used.
