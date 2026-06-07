# Black Hole Entropy Conjecture: Padé vs. Taylor Breakdown Analysis

This repository contains the computational framework to test the thermodynamic Entropy Conjecture in binary black hole (BBH) mergers. It compares Numerical Relativity (NR) ground truths with 2nd-order Post-Newtonian (2PN) analytic models to evaluate the breakdown of standard thermodynamic descriptions in the strong-field regime.

Specifically, the codebase evaluates the performance of rational **Padé approximants** versus polynomial **Taylor expansions** in predicting the physical merger radius ($r_{\text{merger}}$) through the maximization of the Bekenstein-Hawking entropy.

## Features

* **Numerical Relativity Ground Truth:** Utilizes surrogate models (via `surfinBH` and `lalsimulation`) to extract the dominant (2, 2) gravitational wave mode, reconstructing the orbital separation to pinpoint the exact moment of merger ($t=0$).
* **Analytic Thermodynamics:** Calculates the instantaneous Bekenstein-Hawking entropy of the binary system using 2PN expansions for total mass ($M$) and angular momentum ($J$).
* **Extrapolation & Peak Detection:** Fits analytic data dynamically and extrapolates deeply into the strong-field regime ($r \to 1.5 M$) to identify the thermodynamic breakdown (entropy peak).
* **Padé vs. Taylor Benchmarking:** Directly contrasts polynomial truncations against rational Padé approximants, demonstrating the superior stability of Padé in predicting physical limits.
* **Parameter Space Heatmaps:** Automatically generates high-resolution 2D heatmaps to map radial mismatches ($\Delta r$) and entropy errors ($\Delta S$) across various mass ratios ($q$) and spin asymmetries ($\Delta \chi$).

## Prerequisites & Installation

The analysis requires Python 3.12+ and relies heavily on scientific computing and gravitational wave libraries.

### Core Dependencies
* `numpy`
* `pandas`
* `scipy` (Specifically `scipy.optimize` for parameter fitting)
* `matplotlib` (For 1D breakdown curves and 2D heatmaps)
* `lalsimulation` / `surfinBH` (For NR surrogate waveform generation)

### Setup
It is recommended to run this project within a virtual environment:

```bash
git clone [https://github.com/YOUR-USERNAME/Entropy-conjecture.git](https://github.com/YOUR-USERNAME/Entropy-conjecture.git)
cd Entropy-conjecture
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

