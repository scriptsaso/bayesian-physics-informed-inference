#app/model_bayes.py
import pymc as pm
import arviz as az
import os


# ==========================================================
# BUILD MODEL (Graph only)
# ==========================================================

def build_model(X_morph, X_stim, y):

    n_morph = X_morph.shape[1]
    n_stim  = X_stim.shape[1]

    model = pm.Model()

    with model:

        # Priors
        α = pm.Normal("α", mu=0, sigma=1)

        β = pm.Normal("β", mu=0, sigma=0.5, shape=n_morph)
        γ = pm.Normal("γ", mu=0, sigma=0.5, shape=n_stim)

        # Linear predictor
        μ = α + pm.math.dot(X_morph, β) + pm.math.dot(X_stim, γ)

        # Likelihood
        b = pm.HalfCauchy("b", beta=0.7)
        pm.Laplace("y_obs", mu=μ, b=b, observed=y)

    return model


# ==========================================================
# TRAIN MODEL
# ==========================================================

def train_bayesian(model,
                   draws=3000,
                   tune=2000,
                   chains=4,
                   target_accept=0.999,
                   trace_path="artifacts/trace.nc"):

    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=42,
            idata_kwargs={"log_likelihood": True}
        )

    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    az.to_netcdf(trace, trace_path)

    print(f"\nTrace saved to: {trace_path}")
    return trace


# ==========================================================
# LOAD TRACE
# ==========================================================

def load_trace(trace_path):

    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"Trace not found at {trace_path}")

    trace = az.from_netcdf(trace_path)
    print(f"\nTrace loaded from: {trace_path}")

    return trace


# ==========================================================
# POSTERIOR PREDICTIVE
# ==========================================================

def generate_ppc(model, trace):

    with model:
        ppc = pm.sample_posterior_predictive(
            trace,
            model=model,
            var_names=["y_obs"],
            random_seed=42
        )

    return ppc

