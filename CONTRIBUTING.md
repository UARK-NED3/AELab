# Contributing to AELab

This repository is intended for collaborative research code and tutorials. Contributions should make it easier for another student or collaborator to reproduce an analysis.

## Where to Put Work

- Use `pd-ae/` for partial discharge acoustic emission analysis.
- Use `pd-immersion-ae/` for immersed partial discharge acoustic emission analysis.
- Use `pool-boiling-ae/` for pool boiling acoustic emission analysis.
- Use `flow-boiling-ae/` for flow boiling acoustic emission analysis.
- Keep older or reference work in `spier16/` unless it is being intentionally migrated into one of the project areas.

## Recommended Contribution Types

- Reusable analysis scripts or helper functions.
- Tutorial notebooks that can run in Colab, Jupyter, MATLAB Online, or another documented platform.
- Small metadata files or manifests that describe datasets.
- Documentation that explains how to reproduce an analysis.

## Data Policy

Do not commit large raw datasets. Use OSF or another approved data host and document:

- Dataset title.
- Stable URL or DOI.
- Expected file names.
- Download instructions.
- Required preprocessing steps.
- Contact person if access is restricted.

Small example files are acceptable when they are needed for tests, demonstrations, or parser verification.

## Notebook Guidelines

Notebooks should:

- Start with a short statement of the goal.
- Install or import required packages.
- Use relative paths or downloaded data, not personal absolute paths.
- Include enough outputs to verify that the notebook ran.
- Avoid committing unnecessary hidden checkpoints or large generated outputs.

## Pull Request Checklist

Before requesting review:

- Run the notebook or script from a clean environment when practical.
- Update the README in the relevant project folder.
- Confirm that no large raw data files are staged.
- Explain what data source was used.
- Mention any known limitations or next steps.
