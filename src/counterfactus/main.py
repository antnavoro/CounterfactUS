"""
CounterfactUS.py
----------------
Main entry point for generating optimal and robust counterfactual explanations.
The process consists of two main steps:
1. Fitting the underlying predictive model (GPLSIM, GAM, GLM, etc.) to the data.
2. Formulating and solving the optimization tasks to find the counterfactuals (CEs)
   based on the desired constraints, such as sparsity (L0), and confidence level.
"""

import itertools
import time
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from typing import Optional, Union, List, Dict, Tuple

from .fit_GPLSIM import fit_gplsim_loglik
from .solver import GPLSIM_CE

DEFAULT_WEIGHTS = True
DEFAULT_BOUNDS  = True
SAVE_TIME = False
USE_PREVIOUS_CF_AS_INIT = False

def _normalize_df(df: Optional[Union[pd.DataFrame, np.ndarray]], mins=None, maxs=None) -> Tuple[Optional[Union[pd.DataFrame, np.ndarray]], Optional[Union[pd.Series, np.ndarray]], Optional[Union[pd.Series, np.ndarray]]]:
    """Min-max normalize a dataframe or array to [0, 1] and return the normalized data, mins, and maxs."""
    if df is None or (isinstance(df, (pd.DataFrame, pd.Series)) and df.empty) or (isinstance(df, np.ndarray) and df.size == 0):
        return df, None, None
    if mins is None:
        mins = df.min(axis=0) if hasattr(df, 'min') else np.min(df, axis=0)
    if maxs is None:
        maxs = df.max(axis=0) if hasattr(df, 'max') else np.max(df, axis=0)
    diff_orig = maxs - mins
    
    if isinstance(diff_orig, (pd.Series, pd.DataFrame)):
        mask = (diff_orig == 0)
        diff = diff_orig.copy()
        diff[mask] = 1.0
        normalized = (df - mins) / diff
        if isinstance(mask, pd.Series):
            for col in mask[mask].index:
                normalized[col] = df[col]
        else:
            normalized.loc[:, mask] = df.loc[:, mask]
    else:
        mask = (diff_orig == 0)
        diff = np.where(mask, 1.0, diff_orig)
        normalized = (df - mins) / diff
        
        # If it's a 1D array, mask is a scalar boolean
        if df.ndim == 1:
            if mask:
                normalized = df.copy()
        else:
            normalized[:, mask] = df[:, mask]
        
    return np.array(normalized), mins, maxs


def _preprocess_and_infer_types(data: Optional[Union[pd.DataFrame, np.ndarray]], col_names: List[str]) -> Tuple[Optional[pd.DataFrame], List[str], List[str], List[str], Dict]:
    if data is None:
        return None, [], [], [], {}
        
    if not isinstance(data, pd.DataFrame):
        df = pd.DataFrame(data, columns=col_names)
    else:
        df = data.copy()
        if len(col_names) == len(df.columns):
            df.columns = col_names
            
    processed_cols = {}
    new_col_names = []
    var_types = []
    dropped_cats = []
    
    mappings = {}
    
    for col in df.columns:
        series = df[col]
        is_cat = False
        
        if pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(series) or pd.api.types.is_string_dtype(series):
            is_cat = True
            
        unique_vals = series.dropna().unique()
        n_unique = len(unique_vals)
        
        if is_cat or (not pd.api.types.is_numeric_dtype(series)):
            if n_unique <= 2:
                # Binary mapping
                mapping = {val: float(i) for i, val in enumerate(np.sort(unique_vals))}
                mappings[str(col)] = {'type': 'binary', 'map': mapping}
                processed_cols[str(col)] = series.map(mapping)
                new_col_names.append(str(col))
                var_types.append("binary")
            else:
                # One-hot encoding
                all_dummy_cols = pd.get_dummies(series, prefix=f"{col}__cat", drop_first=False).columns
                if len(all_dummy_cols) > 0:
                    dropped_cats.append(str(all_dummy_cols[0]))
                dummies = pd.get_dummies(series, prefix=f"{col}__cat", drop_first=True, dtype=float)
                mappings[str(col)] = {'type': 'onehot', 'columns': list(dummies.columns), 'all_columns': list(all_dummy_cols)}
                for d_col in dummies.columns:
                    processed_cols[str(d_col)] = dummies[d_col]
                    new_col_names.append(str(d_col))
                    var_types.append("binary")
        else:
            # Numeric
            mappings[str(col)] = {'type': 'numeric'}
            processed_cols[str(col)] = series.astype(float)
            new_col_names.append(str(col))
            
            if n_unique <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                var_types.append("binary")
            elif pd.api.types.is_integer_dtype(series) or np.all(np.mod(unique_vals, 1) == 0):
                var_types.append("integer")
            else:
                var_types.append("continuous")
                
    new_df = pd.DataFrame(processed_cols, index=df.index)
    return new_df, new_col_names, var_types, dropped_cats, mappings

def _align_query(query_data, base_col_names, base_df, mappings):
    if query_data is None:
        return None
    
    if not hasattr(query_data, 'columns'):
        if isinstance(query_data, pd.Series):
            query_data = pd.DataFrame([query_data], columns=base_col_names)
        else:
            arr = np.array(query_data)
            if arr.ndim == 1 and not (isinstance(query_data, list) and len(query_data) > 0 and isinstance(query_data[0], dict)):
                query_data = pd.DataFrame([query_data], columns=base_col_names)
            else:
                query_data = pd.DataFrame(query_data, columns=base_col_names)
        
    q_df = pd.DataFrame(index=query_data.index)
    
    for col in query_data.columns:
        col_str = str(col)
        if col_str not in mappings:
            continue
        mapping = mappings[col_str]
        series = query_data[col]
        
        if mapping['type'] == 'numeric':
            q_df[col_str] = series.astype(float)
        elif mapping['type'] == 'binary':
            # Use the saved mapping; fallback to 0.0 if not found
            q_df[col_str] = series.map(mapping['map']).fillna(0.0)
        elif mapping['type'] == 'onehot':
            for dummy_col in mapping['columns']:
                cat_val = dummy_col[len(col_str) + 6:] # strip {col}__cat_
                q_df[dummy_col] = (series.astype(str) == cat_val).astype(float)
                
    for c in base_df.columns:
        if c not in q_df.columns:
            q_df[c] = 0.0
    return q_df[base_df.columns]

def _parse_weights(weights, col_names, original_col_names):
    """
    Parses user-provided weights and transforms them into a list of tuples: [(w_down, w_up), ...].
    If only one weight is given for a variable, it is duplicated (w, w).
    Supports weights as a list (matched by original features) or a dict.
    If no weights are provided, defaults to [(1.0, 1.0)] for all expanded features.
    """
    n = len(col_names)
    w_out = [(1.0, 1.0) for _ in range(n)]
    
    if weights is None:
        return w_out
        
    def unpack(v):
        if isinstance(v, (int, float)):
            return float(v), float(v)
        if isinstance(v, (tuple, list)) and len(v) == 2:
            return float(v[0]), float(v[1])
        raise ValueError(f"Weight must be a numeric value or (w_down, w_up) pair, got {v!r}")

    # If it's a list, try to map it to original features first, then expanded features
    if isinstance(weights, list):
        if len(weights) == len(original_col_names):
            weights = dict(zip(original_col_names, weights))
        elif len(weights) == len(col_names):
            weights = dict(zip(col_names, weights))
        else:
            raise ValueError(f"List of weights must have length {len(original_col_names)} (original features) or {len(col_names)} (expanded features), got {len(weights)}")

    if isinstance(weights, dict):
        for key, val in weights.items():
            # If the key is an integer index, assume it maps to original features
            if isinstance(key, int):
                if 0 <= key < len(original_col_names):
                    key = original_col_names[key]
                else:
                    raise IndexError(f"Weight index {key} out of bounds for original features.")
            
            if isinstance(key, str):
                if key in col_names:
                    idx = col_names.index(key)
                    w_out[idx] = unpack(val)
                else:
                    found = False
                    for i, c in enumerate(col_names):
                        if c.startswith(f"{key}__cat_"):
                            w_out[i] = unpack(val)
                            found = True
                    if not found:
                        raise ValueError(f"Weight key '{key}' not found in columns or categorical groups.")
    else:
        raise TypeError("Weights must be None, a dict, or a list.")
        
    return w_out


def _parse_inmutables(inmutable, col_names, original_col_names):
    """
    Parses user-provided immutability constraints and transforms them into a list of booleans: [True, False, ...].
    True means the feature is immutable (cannot be changed). False means it can be changed.
    Supports inmutable as:
    - a list of strings (names of immutable features)
    - a list of booleans (matched by original features or expanded features)
    - a dict (matched by column name, categorical prefix, or index) to boolean.
    If no inmutable constraints are provided, defaults to [False, ...] for all expanded features.
    """
    n = len(col_names)
    inm_out = [False for _ in range(n)]
    
    if inmutable is None:
        return inm_out

    # If it's a list of strings, convert to dict {name: True}
    if isinstance(inmutable, list) and all(isinstance(x, str) for x in inmutable):
        inmutable = {k: True for k in inmutable}
        
    # If it's a list of booleans
    elif isinstance(inmutable, list):
        if len(inmutable) == len(original_col_names):
            inmutable = dict(zip(original_col_names, inmutable))
        elif len(inmutable) == len(col_names):
            inmutable = dict(zip(col_names, inmutable))
        else:
            raise ValueError(f"List of inmutable booleans must have length {len(original_col_names)} (original) or {len(col_names)} (expanded), got {len(inmutable)}")

    if isinstance(inmutable, dict):
        for key, val in inmutable.items():
            val = bool(val)
                
            if isinstance(key, int):
                if 0 <= key < len(original_col_names):
                    key = original_col_names[key]
                else:
                    raise IndexError(f"Inmutable index {key} out of bounds for original features.")
            
            if isinstance(key, str):
                if key in col_names:
                    idx = col_names.index(key)
                    inm_out[idx] = val
                else:
                    found = False
                    for i, c in enumerate(col_names):
                        if c.startswith(f"{key}__cat_"):
                            inm_out[i] = val
                            found = True
                    if not found:
                        raise ValueError(f"Inmutable key '{key}' not found in columns or categorical groups.")
    else:
        raise TypeError("inmutable must be None, a list of strings, a list of booleans, or a dict.")
        
    return inm_out


def _parse_bounds(bounds, col_names, original_col_names, mins=None, maxs=None):
    """
    Parses user-provided bounds and transforms them into a list of tuples: [(lb, ub), ...].
    Supports bounds as a list (matched by original features) or a dict.
    If DEFAULT_BOUNDS is True, defaults to the natural bounds (mins, maxs).
    Otherwise, defaults to [(None, None)] for all expanded features.
    """
    n = len(col_names)
    
    if DEFAULT_BOUNDS and mins is not None and maxs is not None:
        b_out = [(float(mins[i]), float(maxs[i])) for i in range(n)]
    else:
        b_out = [(None, None) for _ in range(n)]
    
    if bounds is None:
        return b_out
        
    def unpack(v):
        if v is None:
            return (None, None)
        if isinstance(v, (tuple, list)) and len(v) == 2:
            lb = float(v[0]) if v[0] is not None else None
            ub = float(v[1]) if v[1] is not None else None
            return (lb, ub)
        raise ValueError(f"Bound must be None or a (lb, ub) pair, got {v!r}")

    # If it's a list, try to map it to original features first, then expanded features
    if isinstance(bounds, list):
        if len(bounds) == len(original_col_names):
            bounds = dict(zip(original_col_names, bounds))
        elif len(bounds) == len(col_names):
            bounds = dict(zip(col_names, bounds))
        else:
            raise ValueError(f"List of bounds must have length {len(original_col_names)} (original) or {len(col_names)} (expanded), got {len(bounds)}")

    if isinstance(bounds, dict):
        for key, val in bounds.items():
            if isinstance(key, int):
                if 0 <= key < len(original_col_names):
                    key = original_col_names[key]
                else:
                    raise IndexError(f"Bound index {key} out of bounds for original features.")
            
            if isinstance(key, str):
                if key in col_names:
                    idx = col_names.index(key)
                    b_out[idx] = unpack(val)
                else:
                    found = False
                    for i, c in enumerate(col_names):
                        if c.startswith(f"{key}__cat_"):
                            b_out[i] = unpack(val)
                            found = True
                    if not found:
                        raise ValueError(f"Bound key '{key}' not found in columns or categorical groups.")
    else:
        raise TypeError("bounds must be None, a list, or a dict.")
        
    return b_out


def _predict_initial(x0, z0, beta, gamma, knots, coeffs, family, link, X_mins, X_maxs, Z_mins, Z_maxs):
    # Normalize x0 and z0 first
    x0_norm = np.array(x0, dtype=float)
    if X_mins is not None and X_maxs is not None:
        for j in range(len(x0_norm)):
            if X_maxs[j] > X_mins[j]:
                x0_norm[j] = (x0_norm[j] - X_mins[j]) / (X_maxs[j] - X_mins[j])
                
    z0_norm = np.array(z0, dtype=float)
    if Z_mins is not None and Z_maxs is not None:
        for j in range(len(z0_norm)):
            if Z_maxs[j] > Z_mins[j]:
                z0_norm[j] = (z0_norm[j] - Z_mins[j]) / (Z_maxs[j] - Z_mins[j])

    if beta is None or len(beta) == 0:
        eta = coeffs[0][0] if coeffs else 0.0
    else:
        u = np.dot(beta, x0_norm)
        idx = 0
        for i in range(len(knots) - 1):
            if knots[i] <= u <= knots[i+1]:
                idx = i
                break
            if i == len(knots) - 2 and u > knots[-1]:
                idx = i
            if i == 0 and u < knots[0]:
                idx = 0
        
        eta = 0.0
        if coeffs and idx < len(coeffs):
            for k, c in enumerate(coeffs[idx]):
                eta += c * (u ** k)
    
    theta = eta + (np.dot(gamma, z0_norm) if gamma is not None and len(gamma) > 0 else 0.0)
    
    if link:
        robjects.r(f"._fam_init <- {family}(link='{link}')")
    else:
        robjects.r(f"._fam_init <- {family}()")
    return float(np.array(robjects.r(f"._fam_init$linkinv({theta})"))[0])


def compute_counterfactuals(
    X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    Y: Union[np.ndarray, pd.Series] = None,
    Z: Optional[Union[pd.DataFrame, np.ndarray]] = None,

    # ── Family / link
    family: str = "gaussian",
    link:   Optional[str] = None,

    # ── Fitting options
    var_types_X:     Optional[List[str]] = None,
    var_types_Z:     Optional[List[str]] = None,
    col_names_X:     Optional[List[str]] = None,
    col_names_Z:     Optional[List[str]] = None,
    spline_degree: int = None,
    num_knots:     int = None,
    spline_basis:  str = "ps",
    smooth_method: str = "GCV.Cp",
    r_verbose:     bool = False,
    user_init:     Optional[Union[np.ndarray, pd.Series, pd.DataFrame, List]] = None,

    # ── Counterfactual options
    prediction_threshold: Optional[Union[float, List[float]]] = None,
    distance_threshold:   Optional[Union[float, List[float]]] = None,
    distance:       str = "default",
    max_l0         : Union[int, List[int], None] = None,
    objective      : Union[str, List[str]] = "distance",
    x0s:            Optional[Union[np.ndarray, pd.Series, pd.DataFrame, List]] = None,
    z0s:            Optional[Union[np.ndarray, pd.Series, pd.DataFrame, List]] = None,
    init_x_list:    Optional[List[np.ndarray]] = None,
    init_z_list:    Optional[List[np.ndarray]] = None,
    weights_x:      Optional[Union[Dict, List]] = None,
    weights_z:      Optional[Union[Dict, List]] = None,
    inmutable_x:    Optional[Union[List[str], List[bool], Dict]] = None,
    inmutable_z:    Optional[Union[List[str], List[bool], Dict]] = None,
    bounds_x:       Optional[Union[List[Tuple], Dict]] = None,
    bounds_z:       Optional[Union[List[Tuple], Dict]] = None,
    A_x:            Optional[np.ndarray] = None,
    A_z:            Optional[np.ndarray] = None,
    b:              Optional[np.ndarray] = None,
    confidence_level:   Optional[Union[float, List[float]]] = 0.5,

    # ── Output options
    output_csv:           Optional[str] = None,
    gurobi_verbose:       bool = False,
    gurobi_log:           bool = False,
    gurobi_params:        Optional[Dict] = None,
    plot_figures:         bool = True,
    use_zip_iteration:    bool = False,
) -> pd.DataFrame:
    """
    Given some data X, Y, Z, fit a GPLSIM model and compute counterfactual explanations.
    

    Parameters
    ----------
    prediction_threshold : threshold value(s) for prediction probability/outcome; scalar or list.
    distance_threshold   : maximum distance budget(s); scalar or list.
    distance       : distance metric ('default', 'gower', 'l1', 'l2', 'l0+l1').
    max_l0         : sparsity limit(s); int, list of ints, or None.
    objective      : objective function(s); string ('prediction', 'robust_prediction', 'distance', 'l0', 'confidence_level') or list of strings.
    x0s            : query points for the X (nonparametric) part.
    z0s            : query points for the Z (linear) part.
    init_x_list    : optional list of initial X guesses.
    init_z_list    : optional list of initial Z guesses.
    weights_x      : distance weights for X; dict or list.
    weights_z      : distance weights for Z; dict or list.
    inmutable_x    : immutability constraints for X; list of str, list of bool, or dict.
    inmutable_z    : immutability constraints for Z; list of str, list of bool, or dict.
    bounds_x       : list of (lb, ub) bounds or dict for X; None entries = no bound.
    bounds_z       : list of (lb, ub) bounds or dict for Z; None entries = no bound.
    A_x, A_z, b    : optional linear constraints A_x·x + A_z·z <= b.
    col_names_X    : feature names for X.
    col_names_Z    : feature names for Z.
    confidence_level : confidence level(s) for robust constraints or objectives; scalar or list.
    output_csv     : path for results CSV (default: 'cf_results.csv').
    gurobi_verbose : print Gurobi log (default False).
    gurobi_log     : save per-solve Gurobi log files (default False).
    gurobi_params  : additional Gurobi parameters dict.
    plot_figures  : generate and save figures (default True).
    use_zip_iteration : if True, iterates lists concurrently; if False, takes their Cartesian product (default False).
    

    Returns
    -------
    pd.DataFrame with all results (also written to output_csv).
    """
    import rpy2.rinterface_lib.callbacks
    import sys
    if r_verbose:
        rpy2.rinterface_lib.callbacks.consolewrite_print = lambda x: sys.stdout.write(x)
        rpy2.rinterface_lib.callbacks.consolewrite_warnerror = lambda x: sys.stderr.write(x)
    else:
        rpy2.rinterface_lib.callbacks.consolewrite_print = lambda x: None
        rpy2.rinterface_lib.callbacks.consolewrite_warnerror = lambda x: None


    if X is not None:
        if col_names_X is None:
            if hasattr(X, 'columns'):
                col_names_X = list(X.columns)
            else:
                n_x = X.shape[1]
                col_names_X = [f"x{i}" for i in range(n_x)]
    if Z is not None:
        if col_names_Z is None:
            if hasattr(Z, 'columns'):
                col_names_Z = list(Z.columns)
            else:
                n_z = Z.shape[1]
                col_names_Z = [f"z{i}" for i in range(n_z)]

    original_col_names_X = list(col_names_X) if col_names_X is not None else []
    original_col_names_Z = list(col_names_Z) if col_names_Z is not None else []
    
    if plot_figures:
        print("Plotting figures. Ignoring x0s, z0s, thresholds, objective, confidence_level and max_l0.")
        x0s = None
        z0s = None
        prediction_threshold = None
        confidence_level = [0.5, 0.95]
        objective = "distance"
        use_zip_iteration = False
        
        n_features = len(original_col_names_X) + len(original_col_names_Z)
        max_l0 = list(range(1, min(4, n_features))) + [None]

    # Categorical preprocessing and variable type inference
    X, col_names_X, inferred_types_X, dropped_cats_X, mappings_X = _preprocess_and_infer_types(X, col_names_X)
    Z, col_names_Z, inferred_types_Z, dropped_cats_Z, mappings_Z = _preprocess_and_infer_types(Z, col_names_Z)

    if var_types_X is None:
        var_types_X = inferred_types_X
    if var_types_Z is None:
        var_types_Z = inferred_types_Z

    # Align provided queries
    if x0s is not None:
        x0s = _align_query(x0s, original_col_names_X, X, mappings_X)
    if z0s is not None:
        z0s = _align_query(z0s, original_col_names_Z, Z, mappings_Z)
        
    # Keep track of original X index and query indices
    X_orig_index = X.index if hasattr(X, 'index') else None
    # Keep track of query indices for the output CSV
    x0s_index = x0s.index if hasattr(x0s, 'index') else None

    # Normalize X and Z to [0, 1] and keep track of mins and maxs
    X, X_mins, X_maxs = _normalize_df(X)
    Z, Z_mins, Z_maxs = _normalize_df(Z)

    # Convert everything to numpy arrays for the solver
    if X is not None: X = np.asarray(X, dtype=float)
    if Z is not None: Z = np.asarray(Z, dtype=float)
    if X_mins is not None: X_mins = np.asarray(X_mins, dtype=float)
    if X_maxs is not None: X_maxs = np.asarray(X_maxs, dtype=float)
    if Z_mins is not None: Z_mins = np.asarray(Z_mins, dtype=float)
    if Z_maxs is not None: Z_maxs = np.asarray(Z_maxs, dtype=float)
    if x0s is not None: x0s = np.asarray(x0s, dtype=float)
    if z0s is not None: z0s = np.asarray(z0s, dtype=float)

    confidence_level_list = [confidence_level] if not isinstance(confidence_level, (list, tuple, np.ndarray)) else list(confidence_level)

    if any(a < 0.5 or a >= 1.0 for a in confidence_level_list):
        raise ValueError("All confidence_level values must be in the range [0.5, 1)")
    
    objective_list = [objective] if isinstance(objective, str) else objective
    any_robust = any(a > 0.5 for a in confidence_level_list) or any(obj in ['robust_prediction', 'confidence_level'] for obj in objective_list)
    
    if any_robust:
        beta, gamma, knots, coeffs, y_pred_r, cov_matrix, H_analytic = fit_gplsim_loglik(
                X=X, Z=Z, Y=Y,
                family_name=family,
                link_name=link,
                user_init=user_init,
                robustness=True,
                spline_degree=spline_degree,
                num_knots=num_knots,
                spline_basis=spline_basis,
                smooth_method=smooth_method
            )
    else:
        beta, gamma, knots, coeffs, y_pred_r = fit_gplsim_loglik(
                X=X, Z=Z, Y=Y,
                family_name=family,
                link_name=link,
                user_init=user_init,
                robustness=False,
                spline_degree=spline_degree,
                num_knots=num_knots,
                spline_basis=spline_basis,
                smooth_method=smooth_method
            )
        cov_matrix, H_analytic = None, None

    if x0s is None:
        if y_pred_r is not None and len(y_pred_r) > 0:
            s10_indices = []
            for p in [10, 20, 30, 40, 50]:
                val = np.percentile(y_pred_r, p)
                idx = int(np.argmin(np.abs(y_pred_r - val)))
                s10_indices.append(idx)
            s10_unique = list(dict.fromkeys(s10_indices))
            
            x0s = X[s10_unique] if X is not None else None
            z0s = Z[s10_unique] if Z is not None else None
            
            if X_orig_index is not None:
                x0s_index = X_orig_index[s10_unique]
            else:
                x0s_index = s10_unique
        else:
            x0s = X.copy() if X is not None else None
            z0s = Z.copy() if Z is not None else None

    # ---------------------------------------------------------
    # Sanity Checks
    # ---------------------------------------------------------
    if x0s is not None and x0s.ndim == 1:
        x0s = x0s.reshape(1, -1)
    if z0s is not None and z0s.ndim == 1:
        z0s = z0s.reshape(1, -1)

    if x0s is not None and x0s.size == 0: x0s = None
    if z0s is not None and z0s.size == 0: z0s = None

    if x0s is not None and z0s is not None:
        if x0s.shape[0] != z0s.shape[0]:
            raise ValueError(f"x0s and z0s must have the same number of queries (rows). Got {x0s.shape[0]} and {z0s.shape[0]}.")
            
    if x0s is not None and beta is not None and len(beta) > 0:
        if x0s.shape[1] != len(beta):
            raise ValueError(f"Dimension of x0s ({x0s.shape[1]}) does not match dimension of beta ({len(beta)}).")
            
    if z0s is not None and gamma is not None and len(gamma) > 0:
        if z0s.shape[1] != len(gamma):
            raise ValueError(f"Dimension of z0s ({z0s.shape[1]}) does not match dimension of gamma ({len(gamma)}).")

    if distance_threshold is not None:
        if isinstance(distance_threshold, (list, tuple, np.ndarray)):
            if any(d is not None and d < 0 for d in distance_threshold): raise ValueError("distance_threshold must be >= 0.")
        else:
            if distance_threshold < 0: raise ValueError("distance_threshold must be >= 0.")
            
    if b is not None:
        if A_x is None:
            A_x = np.zeros((len(b), len(col_names_X))).tolist()
        if A_z is None:
            A_z = np.zeros((len(b), len(col_names_Z))).tolist()
    elif A_x is not None or A_z is not None:
        raise ValueError("If you provide A_x or A_z, you must provide the rhs vector b.")

    # Now compute the counterfactuals
    # First change the weights according to the one-hot encoded categories:
    use_default_weights = DEFAULT_WEIGHTS and (weights_x is None) and (weights_z is None)
    if use_default_weights:
        def get_default_weight(dist, vt, mn, mx):
            rng = float(mx - mn)
            if dist == 'default':
                if vt in ['continuous', 'integer']:
                    return 1.0 / (rng**2) if rng > 0 else 1.0
                else:
                    return 1.0
            elif dist == 'gower':
                return 1.0
            elif dist == 'l2':
                return 1.0 / (rng**2) if rng > 0 else 1.0
            else: # 'l1', 'l0+l1'
                return 1.0 / rng if rng > 0 else 1.0

        weights_x = []
        for i in range(len(col_names_X)):
            mn = X_mins.iloc[i] if hasattr(X_mins, 'iloc') else X_mins[i]
            mx = X_maxs.iloc[i] if hasattr(X_maxs, 'iloc') else X_maxs[i]
            w = get_default_weight(distance, var_types_X[i], mn, mx)
            weights_x.append((w, w))
            
        weights_z = []
        for i in range(len(col_names_Z)):
            mn = Z_mins.iloc[i] if hasattr(Z_mins, 'iloc') else Z_mins[i]
            mx = Z_maxs.iloc[i] if hasattr(Z_maxs, 'iloc') else Z_maxs[i]
            w = get_default_weight(distance, var_types_Z[i], mn, mx)
            weights_z.append((w, w))
    else:
        weights_x = _parse_weights(weights_x, col_names_X, original_col_names_X)
        weights_z = _parse_weights(weights_z, col_names_Z, original_col_names_Z)
    inmutable_x = _parse_inmutables(inmutable_x, col_names_X, original_col_names_X)
    inmutable_z = _parse_inmutables(inmutable_z, col_names_Z, original_col_names_Z)
    bounds_x = _parse_bounds(bounds_x, col_names_X, original_col_names_X, X_mins, X_maxs)
    bounds_z = _parse_bounds(bounds_z, col_names_Z, original_col_names_Z, Z_mins, Z_maxs)

    # ---------------------------------------------------------
    # Solver Execution Loop
    # ---------------------------------------------------------
    # Prepare query lists
    x0_list = [x0s[i] for i in range(x0s.shape[0])] if x0s is not None else [None] * (z0s.shape[0] if z0s is not None else 0)
    z0_list = [z0s[i] for i in range(z0s.shape[0])] if z0s is not None else [None] * len(x0_list)
    if init_x_list is None: init_x_list = [None] * len(x0_list)
    if init_z_list is None: init_z_list = [None] * len(z0_list)
    
    if len(x0_list) == 0:
        print("No queries to process.")
        return pd.DataFrame()

    counterfactuals_list = []
    
    needs_target_overall = any(obj in ['distance', 'l0', 'confidence_level'] for obj in objective_list) or any(c > 0.5 for c in confidence_level_list)
    
    if prediction_threshold is None and needs_target_overall and y_pred_r is not None and len(y_pred_r) > 0:
        prediction_threshold_list = [float(np.percentile(y_pred_r, p)) for p in [50, 60, 70, 80, 90]]
    else:
        prediction_threshold_list = [prediction_threshold] if not isinstance(prediction_threshold, (list, tuple, np.ndarray)) else list(prediction_threshold)
        
    distance_threshold_list = [distance_threshold] if not isinstance(distance_threshold, (list, tuple, np.ndarray)) else list(distance_threshold)
    l0_list = [max_l0] if not isinstance(max_l0, (list, tuple, np.ndarray)) else list(max_l0)
    

            
    # Generate tasks
    tasks = []
    if use_zip_iteration:
        lists_to_zip = [
            ("x0s", x0_list),
            ("z0s", z0_list),
            ("init_x", init_x_list),
            ("init_z", init_z_list),
            ("prediction_threshold", prediction_threshold_list),
            ("distance_threshold", distance_threshold_list),
            ("max_l0", l0_list),
            ("objective", objective_list),
            ("confidence_level", confidence_level_list)
        ]
        max_len = max(len(lst) for name, lst in lists_to_zip)
        
        for name, lst in lists_to_zip:
            if len(lst) > 1 and len(lst) != max_len:
                raise ValueError(f"List length mismatch for zip iteration: '{name}' has length {len(lst)}, but max length is {max_len}. All parameter lists must have either length 1 or the same max length.")

        def get_val(lst, idx):
            return lst[idx] if len(lst) > 1 else lst[0]
            
        for i in range(max_len):
            tasks.append({
                "loop_i": i if len(x0_list) > 1 else 0,
                "x0": get_val(x0_list, i),
                "z0": get_val(z0_list, i),
                "init_x": get_val(init_x_list, i),
                "init_z": get_val(init_z_list, i),
                "p_t": get_val(prediction_threshold_list, i),
                "d_t": get_val(distance_threshold_list, i),
                "l0": get_val(l0_list, i),
                "obj": get_val(objective_list, i),
                "conf": get_val(confidence_level_list, i)
            })
    else:
        configs = list(itertools.product(
            prediction_threshold_list,
            distance_threshold_list,
            l0_list,
            objective_list,
            confidence_level_list
        ))
        
        for loop_i, (x0, z0) in enumerate(zip(x0_list, z0_list)):
            explicit_init_x = init_x_list[loop_i]
            explicit_init_z = init_z_list[loop_i]
            for (p_t, d_t, l0, obj, conf) in configs:
                tasks.append({
                    "loop_i": loop_i,
                    "x0": x0,
                    "z0": z0,
                    "init_x": explicit_init_x,
                    "init_z": explicit_init_z,
                    "p_t": p_t,
                    "d_t": d_t,
                    "l0": l0,
                    "obj": obj,
                    "conf": conf
                })
                
    last_x_cf = None
    last_z_cf = None
    prev_outer_config = None
    
    for task_idx, task in enumerate(tasks):
        loop_i = task["loop_i"]
        x0 = task["x0"]
        z0 = task["z0"]
        explicit_init_x = task["init_x"]
        explicit_init_z = task["init_z"]
        p_t = task["p_t"]
        d_t = task["d_t"]
        l0 = task["l0"]
        obj = task["obj"]
        current_confidence_level = task["conf"]
        
        current_outer_config = (loop_i, p_t, d_t, l0, obj)
        if current_outer_config != prev_outer_config:
            last_x_cf = None
            last_z_cf = None
            prev_outer_config = current_outer_config
            
        rob = current_confidence_level > 0.5
        current_p_t = p_t
        needs_target = obj in ['distance', 'l0', 'confidence_level'] or rob
        if current_p_t is None and needs_target:
            raise ValueError(f"A prediction_threshold must be provided when objective is '{obj}' or confidence_level > 0.5.")
                
        print(f"Solving query {task_idx+1}/{len(tasks)} with objective={obj}, pred_t={current_p_t}, dist_t={d_t}, max_l0={l0}, confidence_level={current_confidence_level}...")
        
        prob_summary = ["\n  --- Problem Formulation ---"]
        if obj == 'prediction':
            prob_summary.append("  Objective: Maximize prediction")
        elif obj == 'robust_prediction':
            prob_summary.append(f"  Objective: Maximize robust prediction (confidence_level={current_confidence_level})")
        elif obj == 'distance':
            prob_summary.append(f"  Objective: Minimize {distance} distance")
        elif obj == 'l0':
            prob_summary.append("  Objective: Minimize L0 norm (number of changed features)")
        elif obj == 'confidence_level':
            prob_summary.append(f"  Objective: Maximize confidence (confidence_level) of surpassing threshold {current_p_t}")
        
        prob_summary.append("  Subject to:")
        if current_p_t is not None:
            if obj != 'confidence_level' or rob:
                prefix = f"Robust (confidence_level = {current_confidence_level}) prediction" if rob else "Prediction"
                prob_summary.append(f"    - {prefix} surpassing threshold {current_p_t}")
        if d_t is not None:
            prob_summary.append(f"    - {distance.capitalize()} distance <= {d_t}")
        if l0 is not None:
            prob_summary.append(f"    - Changed features (L0) <= {l0}")
        if A_x is not None and A_z is not None and b is not None:
            prob_summary.append("    - Custom linear constraints")
        has_custom_bounds = any(b != [None, None] for b in bounds_x) or any(b != [None, None] for b in bounds_z)
        if has_custom_bounds:
            prob_summary.append("    - Custom feature bounds")
        if (inmutable_x is not None and any(inmutable_x)) or (inmutable_z is not None and any(inmutable_z)):
            prob_summary.append("    - Immutable features preserved")
            
        print("\n".join(prob_summary))
        
        if explicit_init_x is not None and explicit_init_z is not None:
            current_init_x = explicit_init_x
            current_init_z = explicit_init_z
        elif USE_PREVIOUS_CF_AS_INIT and last_x_cf is not None and last_z_cf is not None:
            current_init_x = last_x_cf
            current_init_z = last_z_cf
        else:
            current_init_x = x0
            current_init_z = z0
        
        if SAVE_TIME:
            start_time = time.time()
            
        # We use x0 and z0 as initial values for the solver
        cost, prediction, l0_val, robust_pred, confidence_val, x_cf, z_cf, status, mipgap, theta_val, robust_theta, y_sqrt_val = GPLSIM_CE(
            beta=beta, gamma=gamma, knots=knots, coeffs=coeffs, 
            init_x=current_init_x, init_z=current_init_z, x0=x0, z0=z0, 
            family=family, link=link, prediction_threshold=current_p_t, distance_threshold=d_t, distance_metric=distance, max_l0=l0, 
            inmutable_x=inmutable_x, inmutable_z=inmutable_z, 
            X_mins=X_mins, X_maxs=X_maxs, Z_mins=Z_mins, Z_maxs=Z_maxs, 
            col_names_X=col_names_X, col_names_Z=col_names_Z, 
            var_types_X=var_types_X, var_types_Z=var_types_Z, 
            dropped_cats_X=dropped_cats_X, dropped_cats_Z=dropped_cats_Z, 
            objective=obj, 
            weights_x=weights_x, bounds_x=bounds_x, 
            weights_z=weights_z, bounds_z=bounds_z, 
            A_x=A_x, A_z=A_z, b=b, 
            confidence_level=current_confidence_level, cov_matrix=cov_matrix, H_analytic=H_analytic,
            gurobi_verbose=gurobi_verbose, gurobi_params=gurobi_params, gurobi_log=gurobi_log
        )
        
        if SAVE_TIME:
            compute_time = time.time() - start_time
        
        if status in [2, 9, 13] and x_cf is not None and z_cf is not None:
            last_x_cf = x_cf
            last_z_cf = z_cf
    
        prediction_0 = _predict_initial(x0, z0, beta, gamma, knots, coeffs, family, link, X_mins, X_maxs, Z_mins, Z_maxs)

        # Format the row to be saved
        row = {
            "id_x0": x0s_index[loop_i] if x0s_index is not None else loop_i,
            "y0": prediction_0,
            "y": prediction,
            "robust_prediction": robust_pred,
            "confidence": confidence_val,
            "prediction_threshold": current_p_t,
            "distance_threshold": d_t,
            "max_l0": l0,
            "l0": l0_val,
            "confidence_level": current_confidence_level,
            "objective": obj,
            "status": status,
            "cost": cost,
            "mipgap": mipgap,
            "theta": theta_val if x_cf is not None else np.nan,
            "robust_theta": robust_theta if x_cf is not None else np.nan,
            "y_sqrt": y_sqrt_val if x_cf is not None else np.nan,
            "beta": list(beta) if beta is not None else [],
            "gamma": list(gamma) if gamma is not None else [],
            "knots": list(knots) if knots is not None else [],
            "coefficients": [list(c) for c in coeffs] if coeffs is not None else [],
        }
        
        if plot_figures:
            row["percentile"] = [10, 20, 30, 40, 50][loop_i] if loop_i < 5 else np.nan
        
        if SAVE_TIME:
            row["compute_time"] = compute_time
        
        # Add original and counterfactual features for X
        if x0 is not None:
            for j, col in enumerate(col_names_X):
                row[f"x0_{col}"] = x0[j]
                row[f"x_cf_{col}"] = x_cf[j] if x_cf is not None else None
        
        # Add original and counterfactual features for Z
        if z0 is not None:
            for j, col in enumerate(col_names_Z):
                row[f"z0_{col}"] = z0[j]
                row[f"z_cf_{col}"] = z_cf[j] if z_cf is not None else None
                
        counterfactuals_list.append(row)
        
        if status == 2: # GRB.OPTIMAL
            print(f"  -> Found optimal counterfactual. Cost: {cost:.4f}, Prediction: {prediction:.4f}")
        elif status in [9, 13] and cost is not None:
            print(f"  -> Found sub-optimal counterfactual (Status: {status}). Cost: {cost:.4f}, Prediction: {prediction:.4f}")
        else:
            print(f"  -> No solution found (Status: {status})")

    results_df = pd.DataFrame(counterfactuals_list)
    
    # Save to CSV
    if output_csv is None:
        output_csv = "counterfactuals_results.csv"
        
    if output_csv != '':
        results_df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")
    
    if plot_figures:
        _generate_plots(results_df, output_csv if output_csv != '' else "counterfactuals_results.csv")
    
    return results_df

def _generate_plots(df, output_csv):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import re
    
    plt.rcParams.update({
        'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18,
        'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 12, 'figure.titlesize': 20
    })
    
    df = df.copy()
    if 'percentile' not in df.columns:
        print("Cannot plot figures: 'percentile' column not found.")
        return
        
    df = df.dropna(subset=['percentile']).copy()
    df['max_l0'] = df['max_l0'].fillna(-1)
    
    x0_cols = [c for c in df.columns if c.startswith('x0_') or c.startswith('z0_')]
    features = [c[3:] for c in x0_cols]
    display_features = [re.sub(r'^(X\d+|Z)_', '', f) for f in features]
    num_features = len(features)
    
    ranges = {}
    for f, c0 in zip(features, x0_cols):
        rng = df[c0].max() - df[c0].min()
        ranges[f] = rng if rng != 0 else 1.0
        
    def make_heatmap_matrix(sub_df):
        mat = np.zeros((len(sub_df), len(features)))
        for i, (_, row) in enumerate(sub_df.iterrows()):
            for j, f in enumerate(features):
                c0 = f'x0_{f}' if f'x0_{f}' in row else f'z0_{f}'
                ccf = f'x_cf_{f}' if f'x_cf_{f}' in row else f'z_cf_{f}'
                mat[i, j] = float((row[ccf] - row[c0]) / ranges[f])
        return mat

    def draw_crosses(ax, mat):
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isnan(mat[i, j]):
                    ax.plot([j, j+1], [i, i+1], color='black', lw=1.5)
                    ax.plot([j, j+1], [i+1, i], color='black', lw=1.5)

    base_name = output_csv.replace('.csv', '')
    
    # 1c. Grid Heatmaps (5 percentiles)
    df_grid = df[np.isclose(df['confidence_level'], 0.5) | np.isclose(df['confidence_level'], 0.95)].copy()
    df_grid['pred_thresh_round'] = df_grid['prediction_threshold'].round(4)
    df_grid = df_grid.drop_duplicates(subset=['id_x0', 'pred_thresh_round', 'max_l0', 'confidence_level']).drop(columns=['pred_thresh_round'])
    
    conf_vals = sorted(df_grid['confidence_level'].unique())
    l0_vals = sorted(df_grid['max_l0'].unique(), key=lambda x: 999 if x == -1 else x)
    s10_percentiles = sorted(df_grid['percentile'].unique())
    
    if len(conf_vals) > 0 and len(l0_vals) > 0:
        for pct in s10_percentiles:
            df_pct = df_grid[df_grid['percentile'] == pct]
            if df_pct.empty: continue
            
            import matplotlib.gridspec as gridspec
            fig = plt.figure(figsize=(4 * len(l0_vals) + 2, 4 * len(conf_vals)))
            gs = gridspec.GridSpec(len(conf_vals), len(l0_vals) + 1, width_ratios=[1]*len(l0_vals) + [0.1])
            axes = np.empty((len(conf_vals), len(l0_vals)), dtype=object)
            for i in range(len(conf_vals)):
                for j in range(len(l0_vals)):
                    axes[i, j] = fig.add_subplot(gs[i, j])
            cbar_ax = fig.add_subplot(gs[:, -1])
            cbar_drawn = False
            
            for i, conf in enumerate(conf_vals):
                for j, l0 in enumerate(l0_vals):
                    ax = axes[i, j]
                    sub = df_pct[(df_pct['max_l0'] == l0) & (df_pct['confidence_level'] == conf)].sort_values('prediction_threshold')
                    l0_lbl = str(num_features) if l0 == -1 else str(int(l0))
                    
                    if sub.empty:
                        ax.axis('off')
                        continue
                        
                    show_x = (i == len(conf_vals) - 1)
                    show_y = (j == 0)
                    
                    mat = make_heatmap_matrix(sub)
                    sns.heatmap(mat, ax=ax, cmap='coolwarm', vmin=-1, vmax=1,
                                xticklabels=display_features if show_x else False, 
                                yticklabels=[f"{r:.3f}" for r in sub['prediction_threshold']] if show_y else False,
                                cbar=not cbar_drawn, cbar_ax=cbar_ax if not cbar_drawn else None)
                    cbar_drawn = True
                    draw_crosses(ax, mat)
                    
                    if show_y: ax.set_ylabel(f"($1-\\alpha = {conf}$)\nThreshold ($\\rho$)")
                    if i == 0: ax.set_title(f"$\\sigma = {l0_lbl}$")
                    
            plt.tight_layout()
            out_f = f"{base_name}_1c_pct{int(pct)}.pdf"
            plt.savefig(out_f)
            plt.close()
            print(f"Saved figure {out_f}")
            
        # 2. Pareto Curves
        fig, axes = plt.subplots(len(conf_vals), len(l0_vals), figsize=(5 * len(l0_vals), 4 * len(conf_vals)), squeeze=False, sharex=True, sharey=True)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_dict = {pct: colors[idx % len(colors)] for idx, pct in enumerate(s10_percentiles)}
        handle_dict = {}
        
        for i, conf in enumerate(conf_vals):
            for j, l0 in enumerate(l0_vals):
                ax = axes[i, j]
                sub = df_grid[(df_grid['max_l0'] == l0) & (df_grid['confidence_level'] == conf)]
                l0_lbl = str(num_features) if l0 == -1 else str(int(l0))
                
                if sub.empty:
                    ax.axis('off')
                    continue
                    
                for pct in s10_percentiles:
                    sub_pct = sub[sub['percentile'] == pct].sort_values('prediction_threshold').dropna(subset=['cost'])
                    if not sub_pct.empty:
                        line, = ax.plot(sub_pct['prediction_threshold'], sub_pct['cost'], marker='o', linewidth=2, color=color_dict[pct])
                        if pct not in handle_dict: handle_dict[pct] = line
                            
                ax.grid(True, alpha=0.3)
                if j == 0: ax.set_ylabel(f"($1-\\alpha = {conf}$)\nDistance")
                if i == len(conf_vals) - 1: ax.set_xlabel("Prediction Threshold ($\\rho$)")
                if i == 0: ax.set_title(f"$\\sigma = {l0_lbl}$")
        
        sorted_pcts = sorted(handle_dict.keys())
        handles = [handle_dict[p] for p in sorted_pcts]
        labels = [f"Per {int(p)}" for p in sorted_pcts]
        fig.legend(handles, labels, loc='lower center', ncol=len(s10_percentiles), bbox_to_anchor=(0.5, 0.0))
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        out_f = f"{base_name}_2_pareto.pdf"
        plt.savefig(out_f)
        plt.close()
        print(f"Saved figure {out_f}")
