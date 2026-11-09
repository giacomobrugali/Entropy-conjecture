# A Maximum Entropy Conjecture for Kerr Black Hole Mergers

---

## Abstract

The maximum-entropy conjecture proposes that thermodynamic principles may govern the selection of the final state of a black hole binary. The instantaneous binary configuration is mapped to a hypothetical Kerr black hole with the same mass and angular momentum, and the corresponding Bekenstein–Hawking entropy is studied as a function of the binary orbital separation. The entropy maximum then provides an estimate of the merger scale. While successfully tested for non-spinning binaries, the validity of this conjecture for spinning and precessing systems remains an open question.

This repository contains the code and analysis developed to test the conjecture in three stages: (i) reproduction of the non-spinning reference case, (ii) extension to aligned-spin binaries, with validation of Taylor and Padé extrapolations of the post-Newtonian (PN) entropy curves, and (iii) application of the same strategy to a restricted catalogue of precessing binaries, compared against an effective merger separation extracted from numerical-relativity (NR) surrogate waveforms.

Within the configurations explored, the conjecture can be extended beyond the non-spinning scenario. Padé extrapolation provides the most robust behaviour, although agreement with NR data in the precessing regime remains sensitive to the spin geometry, the PN order and the extrapolation method.

---

## Table of Contents

1. [Scientific Background](#scientific-background)
2. [Methodology](#methodology)
3. [Summary of Results](#summary-of-results)
4. [Installation](#installation)
5. [Dependencies](#dependencies)
6. [Limitations](#limitations)
7. [Future Work](#future-work)
8. [Citation](#citation)
9. [References](#references)
10. [Acknowledgements and Contact](#acknowledgements-and-contact)

---

## Theoretical Background

A quasi-circular binary black hole (BBH) evolves through inspiral, merger and ringdown. During the inspiral, post-Newtonian theory provides an analytical description of the dynamics. As the separation shrinks, the gravitational field becomes strongly nonlinear and the PN expansion loses accuracy, so that a faithful description of the late inspiral and merger requires numerical relativity.

Black holes admit a thermodynamic description in which the horizon area is identified with entropy. For a Kerr black hole of mass $M$ and angular momentum $J$ (units $G=c=\hbar=k_B=1$):

$$
S_{\mathrm{Kerr}}(M,J) = 2\pi M^2\left(1+\sqrt{1-\left(\frac{J}{M^2}\right)^2}\right),
\qquad \left|\frac{J}{M^2}\right|\le 1 .
$$

The inequality is the Kerr bound. When it is violated, the entropy becomes imaginary and the pair $(M,J)$ cannot be associated with a physical Kerr black hole.

**The maximum-entropy conjecture** maps the instantaneous mass $M(a)=m+E_{\mathrm{bind}}(a)$ and angular momentum $J(a)$ of the binary, at dimensionless separation $a=r/m$, onto those of a hypothetical Kerr black hole. The separation $a_\star$ at which the resulting entropy curve is maximal is interpreted as an estimate of the merger scale:

$$
a_\star^{\mathrm{PN}} = \arg\max_a S(a).
$$

In its original formulation (Rincon-Ramirez et al., 2026) the conjecture was applied to non-spinning binaries to infer the properties of the remnant. This work instead investigates the location of the entropy maximum as a marker of the **merger separation**, and extends the test to spinning and precessing systems.

---

## Methodology

### 1. Post-Newtonian entropy curves

- Quasi-circular PN binding energy and angular momentum are implemented following Blanchet (2024) and Kidder (1995), up to **2PN order**. At 2PN, the non-spinning orbital terms are complemented by the leading spin-orbit (1.5PN) and spin-spin (2PN) contributions. The non-spinning reference calculation is additionally reproduced up to 4PN.
- For **aligned-spin** binaries the total angular momentum is a scalar, $J = L + S_{1z} + S_{2z}$.
- For **precessing** binaries the total angular momentum is computed vectorially, $J=|\vec L+\vec S_1+\vec S_2|$, with the spin and orbital directions evolved using the [`precession`](https://github.com/dgerosa/precession) package.
- The PN description is used only in the controlled region $100 \ge a \ge 10$.

### 2. Taylor and Padé extrapolation

To continue the entropy curve toward the merger ($10 \ge a \ge 2$), the binding energy $E_{\mathrm{bind}}$ and the dimensionless spin $\chi = J/M^2$ are fitted separately, as functions of the centred and rescaled variable $\tilde\xi=(\xi-\xi_c)/\sigma_\xi$ with $\xi=a^{-1/2}$:

- **Taylor:** cubic polynomial, $y(\tilde\xi)=c_0+c_1\tilde\xi+c_2\tilde\xi^2+c_3\tilde\xi^3$.
- **Padé:** $[2/1]$ rational approximant, $y(\tilde\xi)=\dfrac{p_0+p_1\tilde\xi+p_2\tilde\xi^2}{1+q_1\tilde\xi}$.

Both models have four free parameters, enabling a like-for-like comparison. The entropy is then reconstructed from $M_{\mathrm{ext}}$ and $J_{\mathrm{ext}}=\chi_{\mathrm{ext}}M_{\mathrm{ext}}^2$, retaining only points that satisfy the Kerr bound.

**Validation criterion.** In the aligned-spin case the extrapolated maximum is compared with the direct PN maximum. An extrapolation is deemed *PN-consistent* if

$$
\Delta a_{\mathrm{val}} = \left|a_\star^{\mathrm{ext}}-a_\star^{\mathrm{PN}}\right| \le 0.05\, a_\star^{\mathrm{PN}} .
$$

### 3. Effective merger separation from NR surrogates

The reference merger scale is extracted from the dominant $(\ell,m)=(2,2)$ mode of the `NRSur7dq4` surrogate model, accessed through [`gwsurrogate`](https://github.com/sxs-collaboration/gwsurrogate):

1. The merger time is the peak of the mode amplitude, $t_m=\arg\max_t |h_{22}(t)|$.
2. The phase of $h_{22}$ is unwrapped and differentiated to obtain $\omega_{22}(t)$.
3. The orbital frequency $\Omega_{\mathrm{orb}}=\omega_{22}/2$ is converted into an effective separation via Kepler's law, $a_{\mathrm{inst}}=\Omega_{\mathrm{orb}}^{-2/3}$.
4. The effective merger separation is the median of $a_{\mathrm{inst}}$ over the eleven samples centred on the amplitude peak ($\Delta t=0.1\,m$):

$$
a^{\mathrm{eff}}_{\mathrm{NR}}=\mathrm{median}\{a_{\mathrm{inst}}(t_{i_m+k}) : k=-5,\dots,5\}.
$$

This quantity is frequency-based and is **not** a gauge-invariant geometrical distance. The surrogate is used exclusively to define the reference separation and does not enter the PN entropy construction.

### 4. Diagnostics

The comparison between theory and NR is quantified by the signed and absolute separation errors

$$
\Delta a = a_\star - a^{\mathrm{eff}}_{\mathrm{NR}}, \qquad |\Delta a| = \left|a_\star - a^{\mathrm{eff}}_{\mathrm{NR}}\right|,
$$

and, for precessing systems, by the method-dependence measure

$$
\Delta a_{\mathrm{method}}=\left|a^{\mathrm{ext}}_{\star,\mathrm{Padé}}-a^{\mathrm{ext}}_{\star,\mathrm{Taylor}}\right| .
$$

### 5. Parameter-space exploration

- **Aligned spins:** two-dimensional heatmaps of $|\Delta a|$ in the $(\chi_{1z},\chi_{2z})$ plane for $q\in\{0.3,0.6,0.9\}$ and PN orders 0–2, for both extrapolation methods; projections onto the effective spin $\chi_{\mathrm{eff}}=(\chi_{1z}+q\chi_{2z})/(1+q)$ and onto the mass ratio $q\in[0.25,1.0]$ (step $\Delta q=0.05$).
- **Precessing spins:** a catalogue of 26 spin geometries at fixed $(q,\chi_1,\chi_2)=(0.3,0.2,0.4)$, parametrised by the tilt angles $(\theta_1,\theta_2)$ and the relative azimuth $\Delta\phi$, defined in a frame with $\hat L\parallel\hat z$.

---

## Summary of Results

**Non-spinning reference.** The entropy maximum is reproduced for all PN orders, and its location approaches the NR-based effective merger separation as the PN order increases.

| PN order | $a_\star^{\mathrm{PN}}$ | $\Delta a$ | $\|\Delta a\|$ |
|:---:|:---:|:---:|:---:|
| 0PN | 4.000 | −2.183 | 2.183 |
| 1PN | 4.981 | −1.202 | 1.202 |
| 2PN | 5.809 | −0.374 | 0.374 |
| 3PN | 6.373 | 0.190 | 0.190 |
| 4PN | 6.791 | 0.608 | 0.608 |

**Aligned spins.** For the representative configuration $(q,\chi_{1z},\chi_{2z})=(0.35,0.1,0.4)$, Padé extrapolation is PN-consistent at all three orders, whereas Taylor extrapolation fails the 5% criterion at 0PN and 2PN. Padé is therefore adopted as the primary method and Taylor is retained as an independent check. The merger-scale error is largest for anti-aligned configurations and decreases toward larger aligned spins, with a dominant dependence on $\chi_{\mathrm{eff}}$. A tentative interpretation is provided by the orbital hang-up effect.

**Precessing binaries.** The entropy maximum remains finite and physical when the total angular momentum is treated vectorially. Several configurations attain $|\Delta a|<1$, with 1PN Padé results among the most accurate. The accuracy, however, depends on the spin geometry, the PN order and the extrapolation method, and $\Delta a_{\mathrm{method}}$ increases at 2PN, showing that higher PN order does not necessarily reduce the extrapolation uncertainty.

**Conclusion.** The results support the maximum-entropy conjecture as a promising thermodynamic diagnostic of the inspiral-to-merger transition within the configurations explored, but do not establish it as a general merger criterion for spinning binaries.

---

## Installation

### Requirements

Python 3.12 or later.

### Setup

```bash
git clone https://github.com/giacomobrugali/Entropy-conjecture.git
cd Entropy-conjecture

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** `gwsurrogate` and its surrogate data files, as well as other gravitational-wave dependencies, may require additional installation steps depending on the operating system and Python environment. Please consult the documentation of the respective packages.

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Numerical computations |
| `scipy` | Numerical methods and parameter fitting |
| `pandas` | Data handling and analysis |
| `matplotlib` | Scientific visualisation and heatmaps |
| `gwsurrogate` | Access to the `NRSur7dq4` numerical-relativity surrogate model |
| `precession` | PN evolution of spin and orbital angular momentum for precessing binaries |

---

## Limitations

1. The spinning analysis requires PN quantities to be extrapolated below $a=10$, where the PN approximation is no longer quantitatively controlled. The validation criterion tests the preservation of the entropy maximum, not the exactness of the continued curve.
2. The comparison is formulated in terms of an effective, frequency-based separation, which is not a gauge-invariant observable.
3. The precessing study is restricted to a finite catalogue with fixed mass ratio and spin magnitudes.

---

## Future Work

- Extending the precessing analysis to broader ranges of mass ratios, spin magnitudes and spin orientations.
- Reformulating the comparison directly in terms of orbital or gravitational-wave frequency, to reduce the dependence on the effective Keplerian separation and provide a more invariant test.

---

## Citation

If you use this work, please cite the thesis:

```bibtex
@thesis{Brugali2026MaxEntropy,
  author      = {Brugali, Giacomo},
  title       = {A Maximum Entropy Conjecture for Kerr Black Hole Mergers},
  type        = {Bachelor's thesis},
  institution = {University of Milano-Bicocca, Department of Physics},
  year        = {2026},
  note        = {Supervisor: Davide Gerosa; Co-supervisor: Matteo Boschini}
}
```

---

## References

1. M. Rincon-Ramirez, N. K. Johnson-McDaniel, E. Bianchi, I. Gupta, V. Prasad and B. S. Sathyaprakash, *A Maximum Entropy Conjecture for Black Hole Mergers*, arXiv:2601.22388 [gr-qc] (2026).
2. L. Blanchet, *Post-Newtonian theory for gravitational waves*, Living Reviews in Relativity **27**, 4 (2024).
3. L. E. Kidder, *Coalescing binary systems of compact objects to (post)$^{5/2}$-Newtonian order. V. Spin effects*, Physical Review D **52**, 821 (1995).
4. D. Gerosa and M. Kesden, *PRECESSION: Dynamics of spinning black-hole binaries with Python*, Physical Review D **93**, 124066 (2016).
5. V. Varma et al., *Surrogate models for precessing binary black hole simulations with unequal masses*, Physical Review Research **1**, 033015 (2019).
6. S. E. Field et al., *GWSurrogate: A Python package for gravitational wave surrogate models*, Journal of Open Source Software **10**(107), 7073 (2025).
7. J. D. Bekenstein, *Black Holes and Entropy*, Physical Review D **7**, 2333 (1973).
8. S. W. Hawking, *Particle Creation by Black Holes*, Communications in Mathematical Physics **43**, 199 (1975).
9. G. A. Baker Jr. and P. Graves-Morris, *Padé Approximants*, 2nd ed., Cambridge University Press (1996).
10. M. Campanelli, C. O. Lousto and Y. Zlochower, *Spinning-black-hole binaries: The orbital hang-up*, Physical Review D **74**, 041501(R) (2006).

The complete bibliography is available in the thesis.

---

## Acknowledgements and Contact

This work was carried out under the supervision of Prof. Davide Gerosa and Dr. Matteo Boschini at the Department of Physics, University of Milano-Bicocca.

For questions or comments, please open an issue on this repository.
