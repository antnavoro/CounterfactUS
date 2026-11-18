"""
solver.py
---------
Mixed-Integer Programming (MIP) solver for finding optimal counterfactual explanations.
It uses Gurobi to solve the optimization problem based on the trained predictive model
parameters (beta, gamma, knots, coeffs) and the selected distance metric.
It also enforces user-defined constraints like immutable features and boundaries,
and supports statistical guarantees via delta-method approximations.
"""

from __future__ import annotations

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from dataclasses import dataclass
from typing import Dict, List, Optional
import rpy2.robjects as ro
from scipy.stats import norm



def robust_cholesky(matrix, eps=1e-8, name=""):
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(matrix)
        vals = np.maximum(vals, eps)
        M_psd = vecs @ np.diag(vals) @ vecs.T
        diff_norm = np.linalg.norm(matrix - M_psd, 'fro')
        name_str = f" for {name}" if name else ""
        return np.linalg.cholesky(M_psd)



def GPLSIM_CE(beta, gamma, knots, coeffs, init_x, init_z, x0, z0, family, link, prediction_threshold, distance_threshold, distance_metric, max_l0, immutable_x, immutable_z, X_mins, X_maxs, Z_mins, Z_maxs, col_names_X, col_names_Z, var_types_X, var_types_Z, dropped_cats_X, dropped_cats_Z, objective, weights_x, bounds_x, weights_z, bounds_z, A_x, A_z, b, confidence_level, cov_matrix, H_analytic, gurobi_verbose, gurobi_params, gurobi_log, **kwargs):
    """
    Core function to solve the counterfactual optimization problem using Gurobi.
    
    Arguments:
    beta           : list[float] or np.ndarray (Coefficients for X variables)
    gamma          : list[float] or np.ndarray (Coefficients for Z variables)
    knots          : list[float] (Spline knot boundaries)
    coeffs         : list[list[float]] (Polynomial coefficients for each piecewise spline segment)
    init_x         : list[float] (Warm start values for X)
    init_z         : list[float] (Warm start values for Z)
    x0             : list[float] (Original instance unnormalized X values)
    z0             : list[float] (Original instance unnormalized Z values)
    family         : str (R family name, e.g., "binomial")
    link           : str or None (R link name, e.g., "logit")
    prediction_threshold : float (Target prediction threshold)
    distance_threshold : float (Max allowable distance, or None)
    distance_metric: str ('default', 'gower', 'l1', 'l2', 'l0+l1')
    max_l0         : int or None (Maximum number of feature changes allowed)
    immutable_x    : list[bool] (Flags for which X features cannot be changed)
    immutable_z    : list[bool] (Flags for which Z features cannot be changed)
    X_mins         : list[float] (Minimum values of X for normalization)
    X_maxs         : list[float] (Maximum values of X for normalization)
    Z_mins         : list[float] (Minimum values of Z for normalization)
    Z_maxs         : list[float] (Maximum values of Z for normalization)
    col_names_X    : list[str] (Names of the X columns)
    col_names_Z    : list[str] (Names of the Z columns)
    var_types_X    : list[str] (Types: "binary", "integer", or "continuous")
    var_types_Z    : list[str] (Types: "binary", "integer", or "continuous")
    dropped_cats_X : list[str] (Names of the dropped one-hot categories for X)
    dropped_cats_Z : list[str] (Names of the dropped one-hot categories for Z)
    objective      : str ('prediction', 'robust_prediction', 'distance', 'l0', 'confidence_level')
    weights_x      : list[tuple[float, float]] (Negative and positive directional weights for X features)
    bounds_x       : list[tuple[float, float]] or list[None] (Custom lower/upper bounds for X)
    weights_z      : list[tuple[float, float]] (Negative and positive directional weights for Z features)
    bounds_z       : list[tuple[float, float]] or list[None] (Custom lower/upper bounds for Z)
    A_x            : list[list[float]] or None (Coefficients matrix for linear constraints on X)
    A_z            : list[list[float]] or None (Coefficients matrix for linear constraints on Z)
    b              : list[float] or None (RHS boundaries for linear constraints)
    confidence_level : float (Confidence level 1-alpha for robust constraints or objectives)
    cov_matrix     : np.ndarray (Sandwich covariance matrix for the model parameters)
    H_analytic     : dict (Analytical evaluations of the H function and its derivatives)
    gurobi_verbose : bool (Toggle for solver output)
    gurobi_params  : dict[str, any] or None (Additional parameters to pass to model.setParam)
    gurobi_log     : bool (Whether to write Gurobi logs to a file)
    
    Returns distance, prediction, l0_val, robust_pred_val, confidence_val, x_vals, z_vals, status, mipgap, theta_val, robust_theta, y_sqrt_val
    """
    if "inmutable_x" in kwargs:
        immutable_x = kwargs.pop("inmutable_x")
    if "inmutable_z" in kwargs:
        immutable_z = kwargs.pop("inmutable_z")

    n_x = len(beta)
    n_z = len(gamma)
    n=n_x+n_z

    if link is None:
        ro.r(f"._fam <- {family}()")
    else:
        ro.r(f"._fam <- {family}(link='{link}')")
    ref       = float(np.array(ro.r("._fam$linkinv(1)"))[0])
    ref2      = float(np.array(ro.r("._fam$linkinv(2)"))[0])
    increasing = ref2 > ref
    
    if prediction_threshold is not None:
        tau = float(np.array(ro.r(f"._fam$linkfun({prediction_threshold})"))[0])
    else:
        tau = None
        

    model = gp.Model('CE_GPLSIM')
    model.setParam("OutputFlag", 1 if gurobi_verbose else 0)
    model.setParam("NonConvex", 2)
    if gurobi_log:
        model.setParam("LogFile", "gurobi.log")
    for k, v in (gurobi_params or {}).items():
        model.setParam(k, v)

    gower_indices_x = []
    l2_indices_x = []

    x = []
    x_unnorm = []
    for j, col in enumerate(col_names_X):
        vt = var_types_X[j]
        if vt == "binary":
            xi=model.addVar(vtype=GRB.BINARY, name=f"x_{j}_{col}")
            yi=xi
            x.append(xi)
            x_unnorm.append(yi)
            if "__cat_" not in col:
                gower_indices_x.append(j)
        elif vt == "integer":
            yi = model.addVar(vtype=GRB.INTEGER, lb=-GRB.INFINITY, name=f"yx_{j}_{col}")
            xi = model.addVar(lb=-GRB.INFINITY, name=f"x_{j}_{col}")
            x.append(xi)
            x_unnorm.append(yi)
            model.addConstr(xi == (yi - X_mins[j]) / (X_maxs[j] - X_mins[j]))
            l2_indices_x.append(j)
        else:
            yi = model.addVar(lb=-GRB.INFINITY, name=f"yx_{j}_{col}")
            xi = model.addVar(lb=-GRB.INFINITY, name=f"x_{j}_{col}")
            x.append(xi)
            x_unnorm.append(yi)
            model.addConstr(xi == (yi - X_mins[j]) / (X_maxs[j] - X_mins[j]))
            l2_indices_x.append(j)
        if immutable_x[j]:
            model.addConstr(yi == x0[j])
        if bounds_x[j][0] is not None:
            model.addConstr(yi >= bounds_x[j][0])
        if bounds_x[j][1] is not None:
            model.addConstr(yi <= bounds_x[j][1])

    categorical_groups_x = {}
    categorical_groups_x0 = {}
    for j, cat in enumerate(dropped_cats_X):
        parts = cat.split("__cat_")
        if len(parts) == 2:
            prefix = parts[0]
            suffix = parts[1]
            if prefix not in categorical_groups_x:
                categorical_groups_x[prefix] = []
            if prefix not in categorical_groups_x0:
                categorical_groups_x0[prefix] = []
            for i, col in enumerate(col_names_X):
                if prefix + "__cat_" in col:
                    categorical_groups_x[prefix].append(x[i])
                    categorical_groups_x0[prefix].append(x0[i])
            x_drop = model.addVar(vtype=GRB.BINARY, name=f"xcat_{n_x+j}_{cat}")
            categorical_groups_x[prefix].append(x_drop)
            categorical_groups_x0[prefix].append(1 - np.sum(categorical_groups_x0[prefix]))
            model.addConstr(gp.quicksum(categorical_groups_x[prefix]) == 1)

    gower_indices_z = []
    l2_indices_z = []

    z = []
    z_unnorm = []
    for j, col in enumerate(col_names_Z):
        vt = var_types_Z[j]
        if vt == "binary":
            zi=model.addVar(vtype=GRB.BINARY, name=f"z_{j}_{col}")
            yi=zi
            z.append(zi)
            z_unnorm.append(yi)
            if "__cat_" not in col:
                gower_indices_z.append(j)
        elif vt == "integer":
            yi = model.addVar(vtype=GRB.INTEGER, lb=-GRB.INFINITY, name=f"yz_{j}_{col}")
            zi = model.addVar(lb=-GRB.INFINITY, name=f"z_{j}_{col}")
            z.append(zi)
            z_unnorm.append(yi)
            model.addConstr(zi == (yi - Z_mins[j]) / (Z_maxs[j] - Z_mins[j]))
            l2_indices_z.append(j)
        else:
            yi = model.addVar(lb=-GRB.INFINITY, name=f"yz_{j}_{col}")
            zi = model.addVar(lb=-GRB.INFINITY, name=f"z_{j}_{col}")
            z.append(zi)
            z_unnorm.append(yi)
            model.addConstr(zi == (yi - Z_mins[j]) / (Z_maxs[j] - Z_mins[j]))
            l2_indices_z.append(j)
        if immutable_z[j]:
            model.addConstr(yi == z0[j])
        if bounds_z[j][0] is not None:
            model.addConstr(yi >= bounds_z[j][0])
        if bounds_z[j][1] is not None:
            model.addConstr(yi <= bounds_z[j][1])

    categorical_groups_z = {}
    categorical_groups_z0 = {}
    for j, cat in enumerate(dropped_cats_Z):
        parts = cat.split("__cat_")
        if len(parts) == 2:
            prefix = parts[0]
            suffix = parts[1]
            if prefix not in categorical_groups_z:
                categorical_groups_z[prefix] = []
            if prefix not in categorical_groups_z0:
                categorical_groups_z0[prefix] = []
            for i, col in enumerate(col_names_Z):
                if prefix + "__cat_" in col:
                    categorical_groups_z[prefix].append(z[i])
                    categorical_groups_z0[prefix].append(z0[i])
            z_drop = model.addVar(vtype=GRB.BINARY, name=f"zcat_{n_z+j}_{cat}")
            categorical_groups_z[prefix].append(z_drop)
            categorical_groups_z0[prefix].append(1 - np.sum(categorical_groups_z0[prefix]))
            model.addConstr(gp.quicksum(categorical_groups_z[prefix]) == 1)

    model.update()

    for i in range(n_x):
        x_unnorm[i].start = float(init_x[i])

    for i in range(n_z):
        z_unnorm[i].start = float(init_z[i])

    if A_x is not None and A_z is not None and b is not None:
        for row_x, row_z, rhs in zip(A_x, A_z, b):
            model.addConstr(gp.quicksum(cx * vx for cx, vx in zip(row_x, x)) + gp.quicksum(cz * vz for cz, vz in zip(row_z, z)) <= rhs)

    distance = model.addVar(lb=0, name="distance")

    if n_x>0:
        u = model.addVar(lb=knots[0], ub=knots[-1], name="u")
        model.addConstr(u == gp.quicksum(beta[i] * x[i] for i in range(len(x))), name="def_u")
        # Binary piece-selector
        P  = len(knots)-1
        w  = model.addVars(P, vtype=GRB.BINARY, name="w")
        model.addConstr(w.sum() == 1, name="one_piece")
        
        has_cubic = any(len(c) == 4 for c in coeffs)
        if has_cubic:
            u2 = model.addVar(lb=-GRB.INFINITY, name="u2")
            model.addConstr(u2 == u * u, name="def_u2")

        eta = model.addVar(lb=-GRB.INFINITY, name="eta")

        for i in range(P):
            model.addGenConstrIndicator(w[i], True, u >= knots[i],   name=f"u_lb_{i}")
            model.addGenConstrIndicator(w[i], True, u <= knots[i+1], name=f"u_ub_{i}")

            ui = model.addVar(lb=-GRB.INFINITY, name=f"u_nl_{i}")

            if len(coeffs[i]) == 4:
                c0, c1, c2, c3 = coeffs[i]
                poly_expr = c0 + c1 * u + c2 * u2 + c3 * u * u2
            else:
                poly_expr = sum(c * u**k for k, c in enumerate(coeffs[i]))
            
            model.addConstr(ui == poly_expr, name=f"poly_{i}")

            model.addGenConstrIndicator(w[i], True, eta - ui == 0, name=f"eta_on_{i}")
    else:
        eta = coeffs[0][0]
    
    theta = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="theta")
    model.addConstr(theta == eta + gp.quicksum(gamma[i]*z[i] for i in range(n_z)), name="def_theta")
    
    # Delta method auxiliary variables
    if confidence_level > 0.5 or objective == 'confidence_level':
        Z_alpha = norm.ppf(confidence_level) if confidence_level > 0.5 else 0.0
        l_alpha = len(H_analytic['alpha']) if H_analytic is not None else 0
        l_beta = len(beta)
        l_gamma = len(gamma) + 1 # intercept + z
        n_params = l_alpha + l_beta + l_gamma
        
        # 1. v_alpha: H(u) evaluated
        v_alpha_vars = []
        if n_x > 0:
            for j in range(l_alpha):
                v_alpha_j = model.addVar(lb=-GRB.INFINITY, name=f"v_alpha_{j}")
                v_alpha_vars.append(v_alpha_j)
                
                for i in range(P):
                    hij = model.addVar(lb=-GRB.INFINITY, name=f"h_{j}_{i}")
                    coeffs_hj = H_analytic['H'][j][i]
                    
                    if len(coeffs_hj) == 4:
                        c0, c1, c2, c3 = coeffs_hj
                        poly_expr_hj = c0 + c1 * u + c2 * u2 + c3 * u * u2
                    else:
                        poly_expr_hj = sum(c * (u**k) for k, c in enumerate(coeffs_hj)) if len(coeffs_hj) > 0 else 0.0
                        
                    model.addConstr(hij == poly_expr_hj, name=f"poly_h_{j}_{i}")
                    model.addGenConstrIndicator(w[i], True, v_alpha_j - hij == 0, name=f"sel_h_{j}_{i}")
                    
            # 2. v_beta: eta'(u) * x
            eta_prime = model.addVar(lb=-GRB.INFINITY, name="eta_prime")
            for i in range(P):
                epi = model.addVar(lb=-GRB.INFINITY, name=f"eta_prime_{i}")
                coeffs_i = coeffs[i]
                deriv_coeffs = [k * c for k, c in enumerate(coeffs_i) if k > 0]
                
                if len(deriv_coeffs) == 3:
                    c0, c1, c2 = deriv_coeffs
                    poly_expr_epi = c0 + c1 * u + c2 * u2
                else:
                    poly_expr_epi = sum(c * (u**k) for k, c in enumerate(deriv_coeffs)) if deriv_coeffs else 0.0
                
                model.addConstr(epi == poly_expr_epi, name=f"poly_ep_{i}")
                model.addGenConstrIndicator(w[i], True, eta_prime - epi == 0, name=f"sel_ep_{i}")
                
            v_beta_vars = []
            for k in range(n_x):
                v_beta_k = model.addVar(lb=-GRB.INFINITY, name=f"v_beta_{k}")
                model.addConstr(v_beta_k == eta_prime * x[k], name=f"bilin_v_beta_{k}")
                v_beta_vars.append(v_beta_k)
        else:
            # Fallback if no x features (e.g. glm/constant)
            v_alpha_vars = [0.0] * l_alpha
            v_beta_vars = []
            
        # 3. v_gamma: (1, z)
        v_gamma_0 = model.addVar(lb=1.0, ub=1.0, name="v_gamma_0")
        v_gamma_vars = [v_gamma_0] + z
        
        # Assemble V vector
        V_vars = v_alpha_vars + v_beta_vars + v_gamma_vars
        
        # 4. S = cov_matrix. V^T S V = || L^T V ||^2
        L = robust_cholesky(cov_matrix, name="delta_method")
        L_T = L.T
        
        W_vars = []
        for i in range(n_params):
            w_i = model.addVar(lb=-GRB.INFINITY, name=f"W_{i}")
            expr = gp.quicksum(L_T[i, j] * V_vars[j] for j in range(n_params))
            model.addConstr(w_i == expr, name=f"def_W_{i}")
            W_vars.append(w_i)
            
        y_sqrt = model.addVar(lb=0, name="y_sqrt")
        model.addQConstr(y_sqrt * y_sqrt >= gp.quicksum(w_i * w_i for w_i in W_vars), name="socp_y_sqrt")
    
    # L0 modeling (1 unit changes for categorical variables)
    s_x_vars = []
    cat_indices_x = set()
    for prefix in categorical_groups_x:
        for i, col in enumerate(col_names_X):
            if prefix + "__cat_" in col:
                cat_indices_x.add(i)

    for i in range(n_x):
        if i not in cat_indices_x:
            s_i = model.addVar(vtype=GRB.BINARY, name=f"s_x_{i}")
            model.addGenConstrIndicator(s_i, 0, x_unnorm[i] == x0[i])
            s_x_vars.append(s_i)

    for prefix in categorical_groups_x:
        s_cat = model.addVar(vtype=GRB.BINARY, name=f"s_xcat_{prefix}")
        s_x_vars.append(s_cat)
        group_vars = categorical_groups_x[prefix]
        group_x0 = categorical_groups_x0[prefix]
        aux_match = model.addVar(lb=0, ub=1, name=f"aux_match_x_{prefix}")
        model.addConstr(aux_match == gp.quicksum(v0 * v for v0, v in zip(group_x0, group_vars)))
        model.addGenConstrIndicator(s_cat, 0, aux_match == 1.0)

    s_z_vars = []
    cat_indices_z = set()
    for prefix in categorical_groups_z:
        for i, col in enumerate(col_names_Z):
            if prefix + "__cat_" in col:
                cat_indices_z.add(i)

    for i in range(n_z):
        if i not in cat_indices_z:
            s_i = model.addVar(vtype=GRB.BINARY, name=f"s_z_{i}")
            model.addGenConstrIndicator(s_i, 0, z_unnorm[i] == z0[i])
            s_z_vars.append(s_i)

    for prefix in categorical_groups_z:
        s_cat = model.addVar(vtype=GRB.BINARY, name=f"s_zcat_{prefix}")
        s_z_vars.append(s_cat)
        group_vars = categorical_groups_z[prefix]
        group_z0 = categorical_groups_z0[prefix]
        aux_match = model.addVar(lb=0, ub=1, name=f"aux_match_z_{prefix}")
        model.addConstr(aux_match == gp.quicksum(v0 * v for v0, v in zip(group_z0, group_vars)))
        model.addGenConstrIndicator(s_cat, 0, aux_match == 1.0)

    l0 = model.addVar(lb=0, name="l0")
    model.addConstr(l0 == gp.quicksum(s_x_vars) + gp.quicksum(s_z_vars))
    if max_l0 is not None:
        model.addConstr(l0 <= max_l0)
    
    if distance_metric in ['default', 'gower']:
        # Continuous variable modeling
        cont_dist = model.addVar(lb=0, name="cont_dist")
        p_x = model.addVars(len(l2_indices_x), lb=0, name="px")
        p_z = model.addVars(len(l2_indices_z), lb=0, name="pz")
        q_x = model.addVars(len(l2_indices_x), lb=0, name="qx")
        q_z = model.addVars(len(l2_indices_z), lb=0, name="qz")
        
        for i,j in enumerate(l2_indices_x):
            model.addConstr(x_unnorm[j]-x0[j] == p_x[i]-q_x[i])
        for i,j in enumerate(l2_indices_z):
            model.addConstr(z_unnorm[j]-z0[j] == p_z[i]-q_z[i])
            
        if distance_metric == 'default':
            model.addConstr(cont_dist*cont_dist >= gp.quicksum(weights_x[j][1]*p_x[i]*p_x[i] for i,j in enumerate(l2_indices_x))+gp.quicksum(weights_x[j][0]*q_x[i]*q_x[i] for i,j in enumerate(l2_indices_x)) +
            gp.quicksum(weights_z[j][1]*p_z[i]*p_z[i] for i,j in enumerate(l2_indices_z))+gp.quicksum(weights_z[j][0]*q_z[i]*q_z[i] for i,j in enumerate(l2_indices_z)))
        else: # 'gower' uses normalized L1 for continuous variables
            model.addConstr(cont_dist == gp.quicksum(weights_x[j][1]*p_x[i]/(X_maxs[j]-X_mins[j] if X_maxs[j]!=X_mins[j] else 1.0) + weights_x[j][0]*q_x[i]/(X_maxs[j]-X_mins[j] if X_maxs[j]!=X_mins[j] else 1.0) for i,j in enumerate(l2_indices_x)) +
            gp.quicksum(weights_z[j][1]*p_z[i]/(Z_maxs[j]-Z_mins[j] if Z_maxs[j]!=Z_mins[j] else 1.0) + weights_z[j][0]*q_z[i]/(Z_maxs[j]-Z_mins[j] if Z_maxs[j]!=Z_mins[j] else 1.0) for i,j in enumerate(l2_indices_z)))

        # Gower modelling for binary/categorical
        gower = model.addVar(lb=0, name="gower")
        gower_terms = []

        for j in gower_indices_x:
            gower_terms.append((1-x0[j]) * weights_x[j][1] * x[j] + x0[j]*weights_x[j][0]*(1-x[j]))

        for j in gower_indices_z:
            gower_terms.append((1-z0[j]) * weights_z[j][1] * z[j] + z0[j]*weights_z[j][0]*(1-z[j]))

        for j, cat in enumerate(dropped_cats_X):
            prefix = cat.split("__cat_")[0]
            group_vars = categorical_groups_x[prefix]
            group_x0 = categorical_groups_x0[prefix]
            
            idxs = [i for i, col in enumerate(col_names_X) if prefix + "__cat_" in col]
            w_eff = np.mean([0.5 * (weights_x[i][1] + weights_x[i][0]) for i in idxs])
            
            gower_terms.append(w_eff * (1 - gp.quicksum(v0 * v for v0, v in zip(group_x0, group_vars))))

        for j, cat in enumerate(dropped_cats_Z):
            prefix = cat.split("__cat_")[0]
            group_vars = categorical_groups_z[prefix]
            group_z0 = categorical_groups_z0[prefix]
            
            idxs = [i for i, col in enumerate(col_names_Z) if prefix + "__cat_" in col]
            w_eff = np.mean([0.5 * (weights_z[i][1] + weights_z[i][0]) for i in idxs])
            
            gower_terms.append(w_eff * (1 - gp.quicksum(v0 * v for v0, v in zip(group_z0, group_vars))))

        model.addConstr(gower == gp.quicksum(gower_terms))

        # Definition of distance
        model.addConstr(distance == cont_dist + gower, name="def_distance")
        
    elif distance_metric in ['l1', 'l2', 'l0+l1']:
        p_x = model.addVars(n_x, lb=0, name="px")
        q_x = model.addVars(n_x, lb=0, name="qx")
        p_z = model.addVars(n_z, lb=0, name="pz")
        q_z = model.addVars(n_z, lb=0, name="qz")
        
        for i in range(n_x):
            model.addConstr(x_unnorm[i]-x0[i] == p_x[i]-q_x[i])
        for i in range(n_z):
            model.addConstr(z_unnorm[i]-z0[i] == p_z[i]-q_z[i])
            
        if distance_metric == 'l1':
            model.addConstr(distance == gp.quicksum(weights_x[i][1]*p_x[i] + weights_x[i][0]*q_x[i] for i in range(n_x)) +
                                        gp.quicksum(weights_z[i][1]*p_z[i] + weights_z[i][0]*q_z[i] for i in range(n_z)), name="def_distance_l1")
        elif distance_metric == 'l2':
            model.addConstr(distance * distance >= gp.quicksum(weights_x[i][1]*p_x[i]*p_x[i] + weights_x[i][0]*q_x[i]*q_x[i] for i in range(n_x)) +
                                                   gp.quicksum(weights_z[i][1]*p_z[i]*p_z[i] + weights_z[i][0]*q_z[i]*q_z[i] for i in range(n_z)), name="def_distance_l2")
        elif distance_metric == 'l0+l1':
            l1 = model.addVar(lb=0, name="l1")
            model.addConstr(l1 == gp.quicksum(weights_x[i][1]*p_x[i] + weights_x[i][0]*q_x[i] for i in range(n_x)) +
                                  gp.quicksum(weights_z[i][1]*p_z[i] + weights_z[i][0]*q_z[i] for i in range(n_z)), name="def_distance_l1")
            
            l0_indep = model.addVar(lb=0, name="l0_indep")
            s_x_indep = model.addVars(n_x, vtype=GRB.BINARY, name="s_indep_x")
            s_z_indep = model.addVars(n_z, vtype=GRB.BINARY, name="s_indep_z")
            for i in range(n_x):
                model.addGenConstrIndicator(s_x_indep[i], 0, x_unnorm[i] == x0[i])
            for i in range(n_z):
                model.addGenConstrIndicator(s_z_indep[i], 0, z_unnorm[i] == z0[i])
            model.addConstr(l0_indep == s_x_indep.sum() + s_z_indep.sum(), name="def_l0_indep")
            
            model.addConstr(distance == l0_indep + l1, name="def_distance_l0_l1")
    else:
        raise ValueError(f"Unknown distance_metric: {distance_metric}")
    
    # Constraints
    if distance_threshold is not None:
        model.addConstr(distance <= distance_threshold, name="max_distance")
        
    if tau is not None:
        if increasing:
            if confidence_level > 0.5:
                model.addConstr(theta - Z_alpha * y_sqrt >= tau, name="robust_pred_req")
            elif objective != 'confidence_level':
                model.addConstr(theta >= tau, name="pred_req")
        else:
            if confidence_level > 0.5:
                model.addConstr(theta + Z_alpha * y_sqrt <= tau, name="robust_pred_req")
            elif objective != 'confidence_level':
                model.addConstr(theta <= tau, name="pred_req")
                
    # Objectives
    if objective == 'prediction':
        if increasing:
            model.setObjective(theta, GRB.MAXIMIZE)
        else:
            model.setObjective(theta, GRB.MINIMIZE)
    elif objective == 'robust_prediction':
        if confidence_level < 0.5:
            raise ValueError("robust_prediction objective requires confidence_level >= 0.5")
        if increasing:
            model.setObjective(theta - Z_alpha * y_sqrt, GRB.MAXIMIZE)
        else:
            model.setObjective(theta + Z_alpha * y_sqrt, GRB.MINIMIZE)
    elif objective == 'distance':
        model.setObjective(distance, GRB.MINIMIZE)
    elif objective == 'l0':
        model.setObjective(l0, GRB.MINIMIZE)
    elif objective == 'confidence_level':
        if tau is None:
            raise ValueError("confidence_level objective requires a prediction_threshold")
        obj_var = model.addVar(lb=-GRB.INFINITY, name="obj_var")
        if increasing:
            model.addConstr(obj_var * y_sqrt <= theta - tau, name="obj_constr")
        else:
            model.addConstr(obj_var * y_sqrt <= tau - theta, name="obj_constr")
        model.setObjective(obj_var, GRB.MAXIMIZE)
    else:
        raise ValueError(f"Unknown objective: {objective}")

    # Solve
    model.optimize()

    if model.Status == GRB.OPTIMAL:
        pass # just to keep the structure clear
    elif model.SolCount > 0:
        print(f"  -> No optimal solution found (Status: {model.Status}), but a sub-optimal solution is available.")
    else:
        print(f"Optimization terminated with status {model.Status}. No solution found.")

        return None, None, None, None, None, None, None, model.Status, float('inf'), None, None, None

    # If we get here, we either have an optimal solution or a sub-optimal one
    dist_val = distance.X
    theta_val = theta.X
    x_vals = [var.X for var in x_unnorm]
    z_vals = [var.X for var in z_unnorm]
    pred_val = float(np.array(ro.r(f"._fam$linkinv({theta_val})"))[0])
    l0_val = l0.X
    
    if confidence_level > 0.5 or objective == 'confidence_level':
        y_sqrt_val = y_sqrt.X
        if increasing:
            robust_theta = theta_val - Z_alpha * y_sqrt_val
        else:
            robust_theta = theta_val + Z_alpha * y_sqrt_val
        robust_pred_val = float(np.array(ro.r(f"._fam$linkinv({robust_theta})"))[0])
        
        if tau is not None:
            if y_sqrt_val > 1e-10:
                if increasing:
                    z_val = (theta_val - tau) / y_sqrt_val
                else:
                    z_val = (tau - theta_val) / y_sqrt_val
                # Using Phi(...) which represents the confidence.
                confidence_val = norm.cdf(z_val)
            else:
                confidence_val = 1.0 if (theta_val >= tau if increasing else theta_val <= tau) else 0.0
        else:
            confidence_val = 0.5
    else:
        robust_pred_val = pred_val
        robust_theta = theta_val
        y_sqrt_val = 0.0
        if tau is not None:
            confidence_val = 1.0 if (theta_val >= tau if increasing else theta_val <= tau) else 0.0
        else:
            confidence_val = 0.5
    
    mipgap = None
    try:
        if model.IsMIP:
            mipgap = model.MIPGap
    except AttributeError:
        pass
        
    return dist_val, pred_val, l0_val, robust_pred_val, confidence_val, x_vals, z_vals, model.Status, mipgap, theta_val, robust_theta, y_sqrt_val

