# Bayesian Physics-Informed Inference API

Uncertainty-aware Bayesian regression system for quantifying interfacial area modulation in bulk heterojunction thin films.

This project integrates experimental GIWAXS/GISAXS structural data with Hamiltonian Monte Carlo inference and structured interpretability, packaged as a production-ready FastAPI service.

---

## Why This Matters

- Quantifies morphology-driven interface evolution under external stimuli
- Provides posterior uncertainty estimates (not just point predictions)
- Enables stimulus-conditioned structural sensitivity analysis
- Bridges experimental physics and deployable ML systems

---
## LLM-Augmented Scientific Interpretation

After Bayesian inference and SHAP analysis, posterior results are 
automatically interpreted via Claude API, providing:

- Physical interpretation of dominant structural descriptors
- Assessment of interfacial area behavior under current conditions
- Suggested next experimental conditions
- Confidence levels for each conclusion

This closes the hypothesis → analysis → reasoning loop, enabling 
autonomous scientific discovery workflows.

### Usage
Set your API key before running:
```bash
export ANTHROPIC_API_KEY="your-key-here"
```
Then run in `ppc` mode — interpretation is generated automatically.

## Key Features

- Bayesian regression with Laplace likelihood
- Hamiltonian Monte Carlo (NUTS) posterior sampling (PyMC)
- Posterior predictive diagnostics (LOO, PPC, energy)
- Multi-level SHAP-based structural interpretability
- Deterministic inference from frozen posterior artifacts
- Dockerized and cloud-deployable
- Infrastructure-as-Code ready (Terraform, AWS ECS)

---

## Execution Modes

### Training

```
python -m app.main --data path/to/data.csv --mode train
```

- Builds Bayesian model
- Runs NUTS sampling
- Saves posterior trace
- Freezes inference artifacts

### Posterior Predictive & Interpretation (PPC)

```
python -m app.main --data path/to/data.csv --mode ppc
```

- Loads frozen posterior
- Generates posterior predictive samples
- Runs diagnostics
- Identifies high-interfacial-area regime
- Performs SHAP-based structural attribution

Training and inference are strictly separated to ensure reproducibility.

---

## Model Artifacts

- `posterior.pkl`
- `scaler.pkl`
- `target_stats.pkl`
- `trace_v2.nc`

Artifacts are frozen and version-controlled.

---

## Containerization

```
docker build -t bayesian-model .
docker run -p 8000:8000 bayesian-model
```

---

## Cloud Deployment

Designed for AWS deployment using:

- ECS (Fargate)
- Application Load Balancer
- IAM roles
- CloudWatch logging
- Terraform Infrastructure-as-Code

---
### Architecture Overview

![Cloud Architecture](docs/architecture.png)

*Containerized FastAPI inference service deployed on AWS ECS Fargate with infrastructure provisioned via Terraform.*

For full scientific methodology and Bayesian formulation, see:

👉 [README_research.md](README_research.md)