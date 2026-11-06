# A Maximum Entropy Conjecture for Kerr Black Hole Mergers

## Overview

This repository provides a computational framework for investigating the **Maximum Entropy Conjecture** in binary Kerr black hole mergers. The project examines the thermodynamic description of binary black hole systems by comparing analytical post-Newtonian (PN) predictions with numerical relativity (NR) results.

The central objective is to investigate the breakdown of standard thermodynamic descriptions in the strong-field regime, focusing on the prediction of the physical merger radius through the maximization of the Bekenstein–Hawking entropy.

In particular, the analysis compares two approaches to the post-Newtonian expansion:

- **Taylor expansions:** conventional polynomial truncations of the PN series.
- **Padé approximants:** rational-function resummations designed to improve the behaviour of perturbative expansions beyond their immediate domain of validity.

By benchmarking both approaches against NR results, the project investigates their ability to capture the strong-field behaviour of the binary system and predict the location of the entropy maximum.

## Key Features

### 1. Numerical Relativity Benchmarking

Uses numerical relativity surrogate models, through `surfinBH` and `lalsimulation`, to obtain gravitational-wave information from binary black hole mergers. The dominant \((\ell,m)=(2,2)\) mode is used to reconstruct the orbital separation and identify the merger time, defined as \(t=0\).

### 2. Post-Newtonian Thermodynamic Modelling

Implements a second-order post-Newtonian (2PN) description of the binary system, calculating its instantaneous Bekenstein–Hawking entropy from the corresponding expansions of the total mass \(M\) and angular momentum \(J\).

### 3. Strong-Field Extrapolation and Peak Detection

Fits analytical predictions to the available data and extrapolates them towards the strong-field regime, approaching \(r \to 1.5M\). The resulting entropy profiles are analysed to identify their maxima and investigate the breakdown of the thermodynamic description.

### 4. Padé vs. Taylor Comparison

Directly benchmarks polynomial Taylor expansions against rational Padé approximants. The comparison focuses on their convergence properties, stability and ability to reproduce the physical behaviour inferred from numerical relativity.

### 5. Parameter-Space Exploration

Generates high-resolution two-dimensional heatmaps to investigate the dependence of the analytical predictions on binary parameters, including the mass ratio \(q\) and spin asymmetry \(\Delta\chi\).

The analysis quantifies discrepancies in the predicted merger radius (\(\Delta r\)) and entropy (\(\Delta S\)), providing a systematic view of the parameter space in which the different approximations remain reliable.

## Installation

### Requirements

The project requires **Python 3.12 or later** and relies on the following scientific computing and gravitational-wave libraries:

| Package | Purpose |
|---|---|
| `numpy` | Numerical computations |
| `pandas` | Data handling and analysis |
| `scipy` | Numerical methods and parameter fitting |
| `matplotlib` | Scientific visualisation and heatmaps |
| `lalsimulation` | Gravitational-wave modelling |
| `surfinBH` | Numerical relativity surrogate models |

### Setup

Clone the repository:

```bash
git clone https://github.com/giacomobrugali/Entropy-conjecture.git
cd Entropy-conjecture
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the Python dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** The `lalsimulation` and `surfinBH` dependencies may require additional installation steps depending on your operating system and Python environment.

## Repository Structure

The repository is organised around the numerical analysis, analytical modelling and visualisation of the maximum entropy conjecture.

*This section can be expanded to document the individual scripts, modules and generated results.*

## Scientific Context

The project builds on the thermodynamic description of black hole mergers and investigates whether resummation techniques can extend the predictive capabilities of perturbative post-Newtonian models into the strong-field regime.

By comparing analytical approximations with numerical relativity benchmarks, it aims to characterise the limitations of Taylor expansions and assess the extent to which Padé resummation provides a more reliable description of the merger dynamics.