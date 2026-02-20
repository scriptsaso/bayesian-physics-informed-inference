# Research Documentation  
Bayesian Physics-Informed Modeling of BHJ Thin Films

---

## Scientific Background

This framework quantifies structural reorganization in bulk heterojunction (BHJ) thin films under external stimuli.

Structural descriptors are extracted from:

- Grazing-Incidence Wide-Angle X-ray Scattering (GIWAXS)
- Grazing-Incidence Small-Angle X-ray Scattering (GISAXS)

GIWAXS resolves molecular packing and π–π stacking, while GISAXS captures nanoscale morphology and interfacial evolution.

The objective is to quantify interfacial area modulation under:

- Electric field exposure
- Thermal treatment
- Controlled annealing
- Dopant incorporation

---

## Bayesian Formulation

The model maps structural descriptors to interfacial metrics using a Bayesian regression framework.

\[
y \sim \text{Laplace}(\mu, b)
\]

\[
\mu = \alpha + X_{morph}\beta + X_{stim}\gamma
\]

The Laplace likelihood provides robustness to experimental outliers and heavy-tailed residual behavior.

Posterior inference is performed using Hamiltonian Monte Carlo (NUTS).

---

## Interpretability Framework

### Global Attribution

Global SHAP analysis estimates the posterior expectation of structural contributions across the full experimental domain.

### High-Interfacial-Area Regime

SHAP is additionally evaluated for posterior samples yielding maximal interfacial area.

This reveals structural descriptors governing interface expansion under specific stimulus conditions.

---

## Bayesian Diagnostics

Model robustness is assessed via:

- Leave-One-Out cross-validation (LOO)
- Pareto-\( k \) diagnostics
- Posterior predictive checks (PPC)
- Residual analysis
- HMC energy diagnostics

These ensure posterior stability and reliable uncertainty quantification.

---

## Research Significance

This framework establishes a quantitative bridge between morphology, external stimuli, dopant incorporation, and functional interfacial behavior through uncertainty-aware Bayesian inference.