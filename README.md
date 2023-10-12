# Artifacts - *EXPLORA*: AI/ML *EXPL*ainability for the Open *RA*N

This repository contains the artifacts for the following paper:

> *EXPLORA*: AI/ML *EXPL*ainability for the Open *RA*N<br>
> (authors anon.)<br>
> CoNEXT ’22, December 5–8, 2023, Paris, France <br>

## Structure

In this repository, we include all data and analysis scripts required to reproduce our results.
Please see the `README` files in each sub-directory for further details.

This repository is structured into the following sub-directories:

1. [`scripts/`](scripts): Contains the python code to reproduce our results.
2. [`data/`](data): Contains the data required by the python scripts.
3. [`results/`](results): Contains intermediate and final results.
4. [`paper-plots`](paper-plots): Contains the TiKZ code to generate the figures of the manuscript.

## Minimal Workflow

> Tested on `Linux 5.11.0-22-generic #23~20.04.1-Ubuntu`

- Make sure you have python Python 3.9.13 installed. Create a [virtual environment](https://docs.python.org/3/library/venv.html) and install the required dependencies (see [requirements.txt](scripts/requirements.txt) in the `scripts/` directory). Install [graphviz](https://graphviz.org/) too via `sudo apt-get install graphviz`.
- Clone this repository.
- Follow the instructions in the [`README`](scripts/README.md) of the [`scripts/`](scripts) sub-directory for the order of execution of the scripts. Check-back the [`data/`](data) to make sense of the workflow.
- Find the results of the processing in [`results/`](results) and the final plots of the paper in [`paper-plots`](paper-plots).
