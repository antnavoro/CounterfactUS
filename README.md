# CounterfactUS: Optimal and Robust Counterfactual Explanations for GPLSIMs

**CounterfactUS** is a Python library designed to generate optimal and robust counterfactual explanations for Generalized Partially Linear Single-Index Models (GPLSIMs). It formulates the counterfactual generation as a Mixed-Integer Programming (MIP) problem, allowing for exact counterfactuals.

## Key Features

*   **Model Support:** Designed for Generalized Partially Linear Single-Index Models (GPLSIMs), including special subcases like Single-Index Models (SIMs) and Generalized Linear Models (GLMs). (Uses R's `mgcv` and `gplsim` internally).
*   **Optimal Explanations:** Uses Gurobi as an optimization backend to find counterfactuals that minimize distance exactly, rather than relying on heuristics.
*   **Statistical Robustness:** Provides ground-truth model guarantees by enforcing prediction thresholds within a user-defined *confidence level*.
*   **Highly Customizable Constraints:**
    *   **Distance Metrics:** Supports L1, L2, Gower and other distance metrics.
    *   **Asymmetric Feature Weights:** Assign different weights to increasing or decreasing specific features.
    *   **Sparsity (L0):** Limit the maximum number of features that can be modified.
    *   **Immutability and Bounds:** Prevent specific features (e.g., *Age*, *Race*) from changing or restrict their allowable ranges.
    *   **Linear Constraints:** Define linear relationships (e.g., $X_1 + X_2 \le b$).
*   **Automated Preprocessing:** Automatically infers feature types (binary, categorical, continuous), normalizes data, and handles one-hot encoding internally.

## Installation

This package requires Python 3.9+ and an active Gurobi license (or academic license, https://www.gurobi.com/academics) to solve the MIP problems. 

Additionally, it requires an R installation. You must manually install the required R packages by opening your R console and running:
```R
install.packages(c("mgcv", "gplsim"))
```

You can install the package directly via pip:

```bash
pip install CounterfactUS
```

*Note: Ensure that your system's R environment is accessible to Python via `rpy2`.*

## Quick Start

Generating counterfactuals is done by calling the `compute_counterfactuals` function. 

```python
import pandas as pd
from counterfactus import compute_counterfactuals

# 1. Load your dataset
X_train = pd.DataFrame(...) # Non-parametric features
Z_train = pd.DataFrame(...) # Linear features
Y_train = pd.Series(...)    # Target variable

# 2. Define the query instances you want to explain
x_queries = pd.DataFrame(...)
z_queries = pd.DataFrame(...)

# 3. Generate counterfactuals
results_df = compute_counterfactuals(
    X=X_train, 
    Z=Z_train, 
    Y=Y_train,
    family_name="binomial", 
    link_name="logit",
    prediction_threshold=0.5,       # Target prediction threshold
    x0s=x_queries,
    z0s=z_queries,
    distance="l1",                  # Distance metric to minimize
    max_l0=2,                       # Allow a maximum of 2 feature changes
    confidence_level=0.95,          # Ensure 95% statistical confidence in the recourse
    inmutable_z=["Race", "Age"],    # Prevent certain features from changing
    gurobi_verbose=False
)

# 4. View results
print(results_df)
# The results are also saved in a CSV file in the current working directory.
```

## Advanced Usage

`compute_counterfactuals` handles complex parameter grids natively. You can pass lists to parameters like `prediction_threshold`, `max_l0`, or `confidence_level` to explore the Pareto frontier of counterfactuals. 

By default, the library computes the Cartesian product of all list parameters. You can change this behavior to iterate through the lists concurrently by setting `use_zip_iteration=True`.

```python
# Explore trade-offs between sparsity and target threshold
results = compute_counterfactuals(
    # ... data ...
    prediction_threshold=[0.5, 0.6, 0.7, 0.8],
    max_l0=[1, 2, 3], 
    confidence_level=[0.5, 0.90, 0.95],
)
# or
# Automatically generate trade-off plots
results = compute_counterfactuals(
    # ... data ...
    plot_figures=True
)
```

## License

MIT License
