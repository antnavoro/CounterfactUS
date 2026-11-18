"""
fit_GPLSIM.py
-------------
Fits the appropriate model given X (single-index part) and Z (linear part)
and returns (beta, gamma, knots, coeffs) — the parametric description of
the fitted model.

The four cases handled, depending on whether X and Z are provided:

  - X=None, Z=None  -> constant model  (intercept-only GLM via R)
  - X=None, Z given -> pure GLM        (no smooth, via stats::glm)
  - X with 1 col    -> GAM             (single univariate smooth, GCV.Cp)
  - X with 2+ cols  -> GPLSIM          (via the gplsim R package)

X and Z are considered absent when they are None or have 0 columns.
"""

import numpy as np
from typing import Optional
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
import rpy2.rinterface_lib.callbacks

# Mute all R console output and warnings/messages. Comment these two lines to obtain the R messages.
rpy2.rinterface_lib.callbacks.consolewrite_print = lambda x: None
rpy2.rinterface_lib.callbacks.consolewrite_warnerror = lambda x: None

np.random.seed(42)

# Activate numpy <-> R automatic conversion (replaces deprecated activate()).
_numpy_converter = ro.conversion.get_conversion() + numpy2ri.converter
ro.conversion.set_conversion(_numpy_converter)

# Pre-import mgcv once at module load time so mgcv::gam() is available in R.
_mgcv = importr("mgcv")


# ---------------------------------------------------------------------------
# Global defaults — change here to affect all four model types at once.
# ---------------------------------------------------------------------------

# Default for the number of points to generate for the spline fitting
DEFAULT_POINTS = 500


# ---------------------------------------------------------------------------
# R -> Python conversion helper.
# ---------------------------------------------------------------------------

def _r_to_array(r_obj) -> np.ndarray:
    """Convert an R vector to a 1-D numpy array."""
    return np.array(r_obj).ravel()

# ---------------------------------------------------------------------------
# Helper to fit piecewise polynomial from grid
# ---------------------------------------------------------------------------

def _fit_piecewise_from_grid(t_grid, eta_grid, knots_full, degree=3):
    """Fit degree-d polynomials on each inter-knot interval from a dense grid."""
    unique_knots = np.sort(np.unique(knots_full))
    knots_out, coeffs_out = [], []
    for i in range(len(unique_knots) - 1):
        a, b = unique_knots[i], unique_knots[i + 1]
        mask = (t_grid >= a) & (t_grid <= b if i == len(unique_knots) - 2 else t_grid < b)
        x_seg, y_seg = t_grid[mask], eta_grid[mask]
        if len(x_seg) < degree + 1:
            continue
        knots_out.append(float(a))
        coeffs_out.append(np.polyfit(x_seg, y_seg, degree)[::-1].tolist())
    knots_out.append(float(unique_knots[-1]))
    return knots_out, coeffs_out

# ---------------------------------------------------------------------------
# Case 1: constant model  (intercept only)
# ---------------------------------------------------------------------------

def _fit_constant(X, Z, Y, family_name, link_name, robustness=False) -> tuple:
    """Fit g(mu) = beta_0 via glm(Y ~ 1). Returns (beta=[], gamma=[], knots, coeffs, y_pred_r[, cov_matrix, H_analytic])."""
    ro.globalenv["Y_r"] = ro.FloatVector(Y)
    ro.r(f'fam_r <- {family_name}(link="{link_name}")')
    ro.r("model_r <- glm(Y_r ~ 1, family=fam_r)")

    intercept = float(_r_to_array(ro.r("coef(model_r)"))[0])
    y_pred_r = _r_to_array(ro.r("predict(model_r, type='response')"))
    # Trivial constant function: one segment over (-inf, +inf)
    
    if robustness:
        from .covariance_GPLSIM import compute_sandwich_covariance
        cov_matrix = compute_sandwich_covariance('constant', X, Z, Y, np.array([]), np.array([]), family_name, link_name)
        return np.array([]), np.array([]), [-np.inf, np.inf], [[intercept]], y_pred_r, cov_matrix, None
        
    return np.array([]), np.array([]), [-np.inf, np.inf], [[intercept]], y_pred_r


# ---------------------------------------------------------------------------
# Case 2: pure GLM  (only linear covariates, no smooth terms)
# ---------------------------------------------------------------------------

def _fit_glm(X, Z, Y, family_name, link_name, robustness=False) -> tuple:
    """Fit a GLM with Z as linear predictors. Returns (beta=[], gamma, knots, coeffs, y_pred_r[, cov_matrix, H_analytic])."""
    k = Z.shape[0]

    ro.globalenv["Y_r"]   = ro.FloatVector(Y)
    ro.globalenv["X_lin"] = ro.r.matrix(
        ro.FloatVector(Z.ravel(order="F")), nrow=k, ncol=Z.shape[1]
    )
    ro.r(f'fam_r <- {family_name}(link="{link_name}")')
    ro.r("model_r <- glm(Y_r ~ X_lin, family=fam_r)")

    coefs = _r_to_array(ro.r("coef(model_r)"))
    y_pred_r = _r_to_array(ro.r("predict(model_r, type='response')"))
    
    gamma = coefs[1:]
    if robustness:
        from .covariance_GPLSIM import compute_sandwich_covariance
        cov_matrix = compute_sandwich_covariance('glm', X, Z, Y, np.array([]), gamma, family_name, link_name)
        return np.array([]), gamma, [-np.inf, np.inf], [[coefs[0]]], y_pred_r, cov_matrix, None
        
    return np.array([]), gamma, [-np.inf, np.inf], [[coefs[0]]], y_pred_r


# ---------------------------------------------------------------------------
# Case 3: GAM with a single univariate smooth
# ---------------------------------------------------------------------------

def _fit_gam_single(X, Z, Y, family_name, link_name, points=DEFAULT_POINTS, robustness=False, include_tau=False, spline_degree=None, num_knots=None, spline_basis="ps", smooth_method="GCV.Cp") -> tuple:
    """
    Fit g(mu) = eta(x) [+ Z gamma] via mgcv::gam with one smooth term.
    X must have exactly 1 column (the smooth variable).
    Returns (beta=[1.0], gamma, knots, coeffs, y_pred_r[, cov_matrix, H_analytic]).

    Skip condition: if the smooth variable has fewer than 10 unique values
    (binary/near-discrete) a smooth is not meaningful; raises ValueError.
    """
    k        = X.shape[0]
    x_smooth = X[:, 0]

    # Skip near-discrete variables — mgcv cannot support 10 knots on them.
    n_unique = len(np.unique(x_smooth))
    if n_unique < 10:
        raise ValueError(
            f"Single-index variable (col 0 of X) has only {n_unique} "
            f"unique values (need >= 10). Skipping."
        )

    ro.globalenv["Y_r"]    = ro.FloatVector(Y)
    ro.globalenv["x_sm_r"] = ro.FloatVector(x_smooth)
    ro.r(f'fam_r <- {family_name}(link="{link_name}")')

    s_args = f'bs="{spline_basis}"'
    if num_knots is not None:
        s_args += f', k={num_knots}'
    if spline_degree is not None:
        s_args += f', m={spline_degree - 1}'

    has_z = Z is not None
    if has_z:
        ro.globalenv["X_lin"] = ro.r.matrix(
            ro.FloatVector(Z.ravel(order="F")), nrow=k, ncol=Z.shape[1]
        )
        ro.r(
            f'model_r <- mgcv::gam(Y_r ~ s(x_sm_r, {s_args}) + X_lin, '
            f'family=fam_r, method="{smooth_method}")'
        )
    else:
        ro.r(
            f'model_r <- mgcv::gam(Y_r ~ s(x_sm_r, {s_args}), '
            f'family=fam_r, method="{smooth_method}")'
        )

    coefs = _r_to_array(ro.r("model_r$coefficients"))
    beta  = np.array([1.0])
    n_lin = Z.shape[1] if has_z else 0
    gamma = coefs[1 : 1 + n_lin] if n_lin > 0 else np.array([])

    # Extract GAM knots early so we can evaluate over their full span
    knots_r = np.array(list(ro.r("model_r$smooth[[1]]$knots")))

    # Build dense grid of eta(x) predictions (Z=0) for piecewise fitting over the FULL knot span.
    vals = np.linspace(knots_r.min(), knots_r.max(), points)
    ro.globalenv["._tg"] = ro.FloatVector(vals.tolist())
    if has_z:
        ro.r("""
            ._nd <- list(x_sm_r = ._tg,
                         X_lin  = matrix(0, length(._tg), ncol(X_lin)))
            ._gp <- predict(model_r, newdata=._nd, type="link")
        """)
    else:
        ro.r('._gp <- predict(model_r, newdata=list(x_sm_r=._tg), type="link")')
    etas = _r_to_array(ro.r("._gp"))

    fit_degree = 3 if spline_degree is None else spline_degree
    knots, coeffs = _fit_piecewise_from_grid(vals, etas, knots_r, degree=fit_degree)
    # Evaluate response values
    y_pred_r = _r_to_array(ro.r("predict(model_r, type='response')"))
    
    if robustness:
        from .covariance_GPLSIM import compute_sandwich_covariance, get_H_analytic
        cov_matrix = compute_sandwich_covariance('gam', X, Z, Y, beta, gamma, family_name, link_name, include_tau=include_tau)
        H_analytic = get_H_analytic('gam', knots_r, n_lin, points=points)
        return beta, gamma, knots, coeffs, y_pred_r, cov_matrix, H_analytic
        
    return beta, gamma, knots, coeffs, y_pred_r


# ---------------------------------------------------------------------------
# Case 4: GPLSIM  (two or more single-index components)
# ---------------------------------------------------------------------------

def _fit_gplsim(X, Z, Y, family_name, link_name, user_init=None, points=DEFAULT_POINTS, robustness=False, include_tau=False, spline_degree=None, num_knots=None, spline_basis="ps", smooth_method="GCV.Cp") -> tuple:
    """
    Fit g(mu) = eta(beta^T X) [+ Z gamma] via the gplsim R package.
    X is the single-index matrix (2+ columns); Z is the optional linear part.
    Returns (beta, gamma, knots, coeffs, y_pred_r[, cov_matrix, H_analytic]).

    Passing Z=NULL explicitly prevents R's lazy-evaluation 'promise' crash.

    Skip condition: if ALL columns of X have fewer than 10 unique values
    the smooth is degenerate; raises ValueError.
    """
    ro.r("library(gplsim)")
    k, n_si = X.shape

    # Skip if all single-index variables are near-discrete.
    n_unique_si = [len(np.unique(X[:, j])) for j in range(n_si)]
    if all(nu < 10 for nu in n_unique_si):
        raise ValueError(
            f"All single-index variables have "
            f"< 10 unique values {n_unique_si}. Skipping."
        )

    if user_init is not None:
        user_init = np.asarray(user_init, dtype=float)
        if len(user_init) != n_si:
            raise ValueError(f"user_init must have length {n_si}, got {len(user_init)}")
        ro.globalenv["user_init_r"] = ro.FloatVector(user_init)
        user_init_arg = ", user.init=user_init_r"
    else:
        user_init_arg = ""

    ro.globalenv["Y_r"]    = ro.FloatVector(Y)
    ro.globalenv["X_si_r"] = ro.r.matrix(
        ro.FloatVector(X.ravel(order="F")), nrow=k, ncol=n_si
    )
    if link_name:
        ro.r(f'fam_r <- {family_name}(link="{link_name}")')
    else:
        ro.r(f'fam_r <- {family_name}()')

    extra_args = ""
    if num_knots is not None:
        extra_args += f', k={num_knots}'
    if spline_degree is not None:
        extra_args += f', m={spline_degree - 1}'
    else:
        extra_args += f', m=2'
        
    has_z = Z is not None
    if has_z:
        ro.globalenv["X_lin_r"] = ro.r.matrix(
            ro.FloatVector(Z.ravel(order="F")), nrow=k, ncol=Z.shape[1]
        )
        ro.r(
            f'model_r <- gplsim(Y_r, X_si_r, Z=X_lin_r, family=fam_r, '
            f'bs="{spline_basis}", method="{smooth_method}"{extra_args}{user_init_arg})'
        )
    else:
        # Z=NULL prevents R's lazy-evaluation 'promise already under evaluation' crash.
        ro.r(
            f'model_r <- gplsim(Y_r, X_si_r, Z=NULL, family=fam_r, '
            f'bs="{spline_basis}", method="{smooth_method}"{extra_args}{user_init_arg})'
        )

    # --- Extract beta (single-index direction) and gamma (linear coefficients) ---
    beta = _r_to_array(ro.r("model_r$theta"))

    if has_z:
        n_z_cols = Z.shape[1]
        # model_r$coefficients (GAM parametric coefs) lists the intercept first,
        # then the parametric coefficients (z), then the smooth basis coefs (s(a).x)
        all_coefs = _r_to_array(ro.r("model_r$coefficients"))
        gamma = all_coefs[1:1+n_z_cols]
        if len(gamma) != n_z_cols:
            raise ValueError(f"Expected {n_z_cols} gamma coefs, got {len(gamma)}")
    else:
        gamma = np.array([])

    index_vals = X @ beta
    
    # Extract GPLSIM knots early so we can evaluate over their full span
    knots_r = np.array(list(ro.r("model_r$smooth[[1]]$knots")))
        
    vals = np.linspace(knots_r.min(), knots_r.max(), points)
    n_z  = Z.shape[1] if has_z else 1   # gplsim always expects a z matrix
    ro.globalenv["._t"]  = ro.FloatVector(vals.tolist())
    ro.globalenv["._nz"] = ro.r("as.integer")(n_z)
    ro.globalenv["._np"] = ro.r("as.integer")(points)

    ro.r("._gp <- predict(model_r, newdata=list(a=._t, z=matrix(0, ._np, ._nz)), type='link')")
    etas = _r_to_array(ro.r("._gp"))

    # Also get the fitted values (response scale) for the training data
    y_pred_r = _r_to_array(ro.r("predict(model_r, type='response')"))

    fit_degree = 3 if spline_degree is None else spline_degree
    knots, coeffs = _fit_piecewise_from_grid(vals, etas, knots_r, degree=fit_degree)
    
    if robustness:
        from .covariance_GPLSIM import compute_sandwich_covariance, get_H_analytic
        cov_matrix = compute_sandwich_covariance('gplsim', X, Z, Y, beta, gamma, family_name, link_name, include_tau=include_tau)
        n_lin = Z.shape[1] if has_z else 0
        H_analytic = get_H_analytic('gplsim', knots_r, n_lin, points=points)
        all_coefs = _r_to_array(ro.r("model_r$coefficients"))
        H_analytic['intercept'] = float(all_coefs[0])
        H_analytic['alpha'] = all_coefs[1+n_z:]
        return beta, gamma, knots, coeffs, y_pred_r, cov_matrix, H_analytic
        
    return beta, gamma, knots, coeffs, y_pred_r


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fit_gplsim_loglik(
    X          : Optional[np.ndarray],
    Z          : Optional[np.ndarray],
    Y          : np.ndarray,
    family_name: str,
    link_name  : str,
    user_init   : Optional[np.ndarray | list] = None,
    points      : int = DEFAULT_POINTS,
    robustness  : bool = False,
    include_tau : bool = False,
    spline_degree: Optional[int] = None,
    num_knots: Optional[int] = None,
    spline_basis: str = "ps",
    smooth_method: str = "GCV.Cp",
) -> tuple:
    """
    Fit the appropriate model and return its parametric representation, predictions, and optionally covariance components.

    Dispatches based on whether X and Z are provided:
      - X=None, Z=None  -> constant model
      - X=None, Z given -> pure GLM
      - X with 1 col    -> GAM (single smooth)
      - X with 2+ cols  -> GPLSIM

    X and Z are treated as absent when they are None or have 0 columns.

    Parameters
    ----------
    X           : (k, n_si) array for the single-index (nonparametric) part,
                  or None / shape (k, 0) if absent.
    Z           : (k, n_lin) array for the linear (parametric) part,
                  or None / shape (k, 0) if absent.
    Y           : (k,) array of responses
    family_name : R family string, e.g. "binomial", "gaussian", "poisson"
    link_name   : R link string,   e.g. "logit", "identity", "log"
    user_init   : optional initial direction vector for GPLSIM (length n_si)
    points      : number of grid points for the eta prediction curve
    robustness  : bool (Whether to compute and return sandwich covariance and analytic H)
    include_tau : bool (Whether to include scale parameter tau in covariance, e.g. for gaussian)
    spline_degree : int (Degree of the polynomial piece fit, defaults to 3)
    num_knots   : int (Number of spline basis knots, k in mgcv/gplsim)

    Returns
    -------
    beta   : np.ndarray  — single-index direction (empty for GLM/constant)
    gamma  : np.ndarray  — linear coefficients for Z (empty if no Z)
    knots  : list[float] — knot boundaries of the piecewise polynomial
                           [-inf, +inf] for constant/GLM (trivial function)
    coeffs : list[list]  — polynomial coefficients per segment (ascending degree)
                           eta(u) = sum(c_k * u**k) on each interval
    y_pred_r : np.ndarray — fitted predictions on the training data
    cov_matrix : np.ndarray (only if robustness=True) — Sandwich covariance matrix
    H_analytic : dict or None (only if robustness=True) — Analytic derivatives of the spline
    """
    Y = np.asarray(Y, dtype=float)
    k = Y.shape[0]

    # Normalise X and Z: None or 0-column arrays both mean "absent".
    def _prep(arr):
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(k, -1)
        return arr if arr.shape[1] > 0 else None

    X = _prep(X)
    Z = _prep(Z)

    has_x = X is not None
    has_z = Z is not None
    n_si  = X.shape[1] if has_x else 0

    if not has_x and not has_z:
        res = _fit_constant(X, Z, Y, family_name, link_name, robustness)
    elif not has_x:
        res = _fit_glm(X, Z, Y, family_name, link_name, robustness)
    elif n_si == 1:
        res = _fit_gam_single(X, Z, Y, family_name, link_name, points, robustness, include_tau, spline_degree, num_knots, spline_basis, smooth_method)
    else:
        res = _fit_gplsim(X, Z, Y, family_name, link_name, user_init, points, robustness, include_tau, spline_degree, num_knots, spline_basis, smooth_method)

    return res