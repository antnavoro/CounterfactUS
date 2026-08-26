"""
covariance_GPLSIM.py
--------------------
Computes the sandwich covariance matrix for generalized partially linear single-index 
models (GPLSIM), as well as standard GAMs, GLMs, and constant models. 
It extracts analytical derivatives from the piecewise polynomial spline segments 
and computes the necessary gradients and Hessians to estimate the variance 
of the model parameters, which is later used to compute robust CEs.
"""

import numpy as np
import rpy2.robjects as ro
from scipy.stats import norm

def get_link_derivatives(family_name, link_name):
    if family_name == 'gaussian' and link_name == 'identity':
        def phi(eta): return eta
        def phi_prime(eta): return np.ones_like(eta)
        def phi_prime_prime(eta): return np.zeros_like(eta)
        def psi_prime(theta): return theta
        def psi_prime_prime(theta): return np.ones_like(theta)

    elif family_name == 'poisson' and link_name == 'log':
        def phi(eta): return eta
        def phi_prime(eta): return np.ones_like(eta)
        def phi_prime_prime(eta): return np.zeros_like(eta)
        def psi_prime(theta): return np.exp(theta)
        def psi_prime_prime(theta): return np.exp(theta)

    elif family_name == 'binomial' and link_name == 'logit':
        def phi(eta): return eta
        def phi_prime(eta): return np.ones_like(eta)
        def phi_prime_prime(eta): return np.zeros_like(eta)
        def psi_prime(theta): 
            return 1.0 / (1.0 + np.exp(-theta))
        def psi_prime_prime(theta): 
            p = 1.0 / (1.0 + np.exp(-theta))
            return p * (1.0 - p)

    elif family_name == 'binomial' and link_name == 'probit':
        def phi(eta): 
            mu = norm.cdf(eta)
            mu = np.clip(mu, 1e-15, 1 - 1e-15)
            return np.log(mu / (1 - mu))
        def phi_prime(eta): 
            mu = norm.cdf(eta)
            mu = np.clip(mu, 1e-15, 1 - 1e-15)
            pdf = norm.pdf(eta)
            return pdf / (mu * (1 - mu))
        def phi_prime_prime(eta): 
            mu = norm.cdf(eta)
            mu = np.clip(mu, 1e-15, 1 - 1e-15)
            pdf = norm.pdf(eta)
            term1 = -eta * pdf * mu * (1 - mu)
            term2 = (pdf**2) * (1 - 2 * mu)
            return (term1 - term2) / ((mu * (1 - mu))**2)
        def psi_prime(theta): 
            return 1.0 / (1.0 + np.exp(-theta))
        def psi_prime_prime(theta): 
            p = 1.0 / (1.0 + np.exp(-theta))
            return p * (1.0 - p)
    else:
        raise ValueError(f"Currently unsupported family/link combination for the computation of the sandwich formula: {family_name}/{link_name}")
        
    return phi, phi_prime, phi_prime_prime, psi_prime, psi_prime_prime

def extract_H_and_derivatives(model_type, X, beta, n_z):
    """ Extract H, H', H'' evaluations on the single-index values for gplsim or gam """
    k = X.shape[0]
    si = X @ beta if model_type == 'gplsim' else X[:, 0]
    eps = 1e-5
    
    ro.globalenv['si_r'] = ro.FloatVector(si)
    ro.globalenv['si_plus_r'] = ro.FloatVector(si + eps)
    ro.globalenv['si_minus_r'] = ro.FloatVector(si - eps)
    
    if model_type == 'gplsim':
        z_cols = max(1, n_z)
        ro.globalenv['z_cols_r'] = ro.r('as.integer')(z_cols)
        ro.globalenv['n_p_r'] = ro.r('as.integer')(k)
        
        ro.r("lpmat <- predict(model_r, newdata=list(a=si_r, z=matrix(0, n_p_r, z_cols_r)), type='lpmatrix')")
        ro.r("lpmat_plus <- predict(model_r, newdata=list(a=si_plus_r, z=matrix(0, n_p_r, z_cols_r)), type='lpmatrix')")
        ro.r("lpmat_minus <- predict(model_r, newdata=list(a=si_minus_r, z=matrix(0, n_p_r, z_cols_r)), type='lpmatrix')")
        start_col = 1 + n_z
        
    elif model_type == 'gam':
        if n_z > 0:
            ro.globalenv['z_cols_r'] = ro.r('as.integer')(n_z)
            ro.globalenv['n_p_r'] = ro.r('as.integer')(k)
            ro.r("lpmat <- predict(model_r, newdata=list(x_sm_r=si_r, X_lin=matrix(0, n_p_r, z_cols_r)), type='lpmatrix')")
            ro.r("lpmat_plus <- predict(model_r, newdata=list(x_sm_r=si_plus_r, X_lin=matrix(0, n_p_r, z_cols_r)), type='lpmatrix')")
            ro.r("lpmat_minus <- predict(model_r, newdata=list(x_sm_r=si_minus_r, X_lin=matrix(0, n_p_r, z_cols_r)), type='lpmatrix')")
            start_col = 1 + n_z
        else:
            ro.r("lpmat <- predict(model_r, newdata=list(x_sm_r=si_r), type='lpmatrix')")
            ro.r("lpmat_plus <- predict(model_r, newdata=list(x_sm_r=si_plus_r), type='lpmatrix')")
            ro.r("lpmat_minus <- predict(model_r, newdata=list(x_sm_r=si_minus_r), type='lpmatrix')")
            start_col = 1
            
    H = np.array(ro.r("lpmat"), dtype=np.float64, copy=True)[:, start_col:]
    H_plus = np.array(ro.r("lpmat_plus"), dtype=np.float64, copy=True)[:, start_col:]
    H_minus = np.array(ro.r("lpmat_minus"), dtype=np.float64, copy=True)[:, start_col:]
    
    H_prime = (H_plus - H_minus) / (2 * eps)
    H_prime_prime = (H_plus - 2 * H + H_minus) / (eps ** 2)
    
    return H, H_prime, H_prime_prime

def get_H_analytic(model_type, knots_r, n_z, degree=3, points=500):
    from .fit_GPLSIM import _fit_piecewise_from_grid
    
    vals = np.linspace(knots_r.min(), knots_r.max(), points)
    
    if model_type == 'gplsim':
        z_cols = max(1, n_z)
        ro.globalenv['z_cols_r'] = ro.r('as.integer')(z_cols)
        ro.globalenv['n_p_r'] = ro.r('as.integer')(points)
        ro.globalenv['si_r'] = ro.FloatVector(vals)
        
        ro.r("lpmat <- predict(model_r, newdata=list(a=si_r, z=matrix(0, n_p_r, z_cols_r)), type='lpmatrix')")
        start_col = 1 + n_z
        
    elif model_type == 'gam':
        ro.globalenv['si_r'] = ro.FloatVector(vals)
        if n_z > 0:
            ro.globalenv['z_cols_r'] = ro.r('as.integer')(n_z)
            ro.globalenv['n_p_r'] = ro.r('as.integer')(points)
            ro.r("lpmat <- predict(model_r, newdata=list(x_sm_r=si_r, X_lin=matrix(0, n_p_r, z_cols_r)), type='lpmatrix')")
            start_col = 1 + n_z
        else:
            ro.r("lpmat <- predict(model_r, newdata=list(x_sm_r=si_r), type='lpmatrix')")
            start_col = 1

    H = np.array(ro.r("lpmat"), dtype=np.float64, copy=True)[:, start_col:]
    l_alpha = H.shape[1]
    
    H_analytic = {'knots': knots_r.tolist(), 'H': [], 'H_prime': [], 'H_prime_prime': []}
    
    for j in range(l_alpha):
        _, coeffs_H = _fit_piecewise_from_grid(vals, H[:, j], knots_r, degree)
        H_analytic['H'].append(coeffs_H)
        
        coeffs_prime = [np.polyder(c[::-1])[::-1].tolist() for c in coeffs_H]
        coeffs_prime_prime = [np.polyder(c[::-1])[::-1].tolist() for c in coeffs_prime]
        
        H_analytic['H_prime'].append(coeffs_prime)
        H_analytic['H_prime_prime'].append(coeffs_prime_prime)
        
    return H_analytic

def compute_sandwich_covariance(model_type, X, Z, Y, beta, gamma, family_name, link_name, include_tau=False):
    """
    Computes the sandwich covariance matrix universally for gplsim, gam, glm, and constant models.
    """
    k = len(Y)
    
    has_alpha = model_type in ['gplsim', 'gam']
    has_beta = model_type == 'gplsim'
    
    if has_alpha:
        lambda_ = float(ro.r("model_r$sp")[0])
        D = np.array(ro.r("model_r$smooth[[1]]$S[[1]]"), dtype=np.float64, copy=True)
    else:
        lambda_ = 0.0
        D = None
    
    coefs = np.array(ro.r("model_r$coefficients"), dtype=np.float64, copy=True)
    intercept = coefs[0]
    n_z = Z.shape[1] if Z is not None else 0
    
    if has_alpha:
        alpha = coefs[1+n_z:]
    else:
        alpha = np.array([], dtype=np.float64)
        
    if Z is not None:
        Z_ext = np.column_stack([np.ones(k), np.array(Z, dtype=np.float64, copy=True)])
        gamma_ext = np.concatenate([[intercept], np.array(gamma, dtype=np.float64, copy=True)])
    else:
        Z_ext = np.ones((k, 1), dtype=np.float64)
        gamma_ext = np.array([intercept], dtype=np.float64)
        
    if has_alpha:
        H, H_prime, H_prime_prime = extract_H_and_derivatives(model_type, X, beta, n_z)
    else:
        H, H_prime, H_prime_prime = None, None, None
    
    phi, phi_prime, phi_prime_prime, psi_prime, psi_prime_prime = get_link_derivatives(family_name, link_name)
    
    l_alpha = len(alpha) if has_alpha else 0
    l_beta = len(beta) if has_beta else 0
    l_gamma = len(gamma_ext)
    
    n_params = l_alpha + l_beta + l_gamma
    omega_meat = np.zeros((n_params, n_params))
    
    hess_alpha_alpha = np.zeros((l_alpha, l_alpha))
    hess_alpha_beta = np.zeros((l_alpha, l_beta))
    hess_alpha_gamma = np.zeros((l_alpha, l_gamma))
    hess_beta_beta = np.zeros((l_beta, l_beta))
    hess_beta_gamma = np.zeros((l_beta, l_gamma))
    hess_gamma_gamma = np.zeros((l_gamma, l_gamma))
    
    if has_alpha:
        theta_all = np.dot(H, alpha) + np.dot(Z_ext, gamma_ext)
    else:
        theta_all = np.dot(Z_ext, gamma_ext)
        
    tau = 1.0
    if include_tau:
        if family_name == 'gaussian':
            y_pred = np.array(ro.r("predict(model_r, type='response')"))
            sigma2 = np.sum((Y - y_pred)**2) / max(1, k - n_params)
            tau = 1.0 / sigma2
        else:
            tau = 1.0

    phi_all = phi(theta_all)
    phi_prime_all = phi_prime(theta_all)
    phi_prime_prime_all = phi_prime_prime(theta_all)
    psi_prime_all = psi_prime(phi_all)
    psi_prime_prime_all = psi_prime_prime(phi_all)
    
    for i in range(k):
        x_i = X[i, :] if X is not None else None
        z_i = Z_ext[i, :]
        y_i = Y[i]
        
        grad_alpha_i = H[i, :] if has_alpha else np.array([])
        
        if has_alpha and has_beta:
            hess_alpha_beta_i = np.outer(H_prime[i, :], x_i) 
            grad_beta_i = np.dot(alpha, H_prime[i, :]) * x_i  
            hess_beta_beta_i = np.dot(alpha, H_prime_prime[i, :]) * np.outer(x_i, x_i)
        else:
            grad_beta_i = np.array([])
        
        grad_gamma_i = z_i
        
        phi_i = phi_all[i]
        phi_prime_i = phi_prime_all[i]
        phi_prime_prime_i = phi_prime_prime_all[i]
        psi_prime_i = psi_prime_all[i]
        psi_prime_prime_i = psi_prime_prime_all[i]
        
        grad_L_alpha_i = (y_i - psi_prime_i) * phi_prime_i * grad_alpha_i
        if has_alpha:
            grad_L_alpha_i -= (lambda_ / k) * D @ alpha
            
        grad_L_beta_i = (y_i - psi_prime_i) * phi_prime_i * grad_beta_i
        grad_L_gamma_i = (y_i - psi_prime_i) * phi_prime_i * grad_gamma_i
        
        grad_i_parts = []
        if has_alpha: grad_i_parts.append(grad_L_alpha_i)
        if has_beta: grad_i_parts.append(grad_L_beta_i)
        grad_i_parts.append(grad_L_gamma_i)
        grad_i = np.concatenate(grad_i_parts)
        
        omega_meat += (1 / k) * np.outer(grad_i, grad_i)
        
        term1 = -psi_prime_prime_i * (phi_prime_i ** 2)
        term2 = (y_i - psi_prime_i) * phi_prime_prime_i
        term3 = (y_i - psi_prime_i) * (phi_prime_i ** 2)
        
        if has_alpha:
            hess_alpha_alpha += (1 / k) * (term1 * np.outer(grad_alpha_i, grad_alpha_i) + term2 * np.outer(grad_alpha_i, grad_alpha_i))
            hess_alpha_gamma += (1 / k) * (term1 * np.outer(grad_alpha_i, grad_gamma_i) + term2 * np.outer(grad_alpha_i, grad_gamma_i))
            if has_beta:
                hess_alpha_beta += (1 / k) * (term1 * np.outer(grad_alpha_i, grad_beta_i) + term2 * np.outer(grad_alpha_i, grad_beta_i) + term3 * hess_alpha_beta_i)
        
        if has_beta:
            hess_beta_beta += (1 / k) * (term1 * np.outer(grad_beta_i, grad_beta_i) + term2 * np.outer(grad_beta_i, grad_beta_i) + term3 * hess_beta_beta_i)
            hess_beta_gamma += (1 / k) * (term1 * np.outer(grad_beta_i, grad_gamma_i) + term2 * np.outer(grad_beta_i, grad_gamma_i))
            
        hess_gamma_gamma += (1 / k) * (term1 * np.outer(grad_gamma_i, grad_gamma_i) + term2 * np.outer(grad_gamma_i, grad_gamma_i))

    if has_alpha:
        hess_alpha_alpha -= (lambda_ / k) * D
    
    # Build block hessian dynamically
    blocks = []
    if has_alpha:
        row = [hess_alpha_alpha]
        if has_beta: row.append(hess_alpha_beta)
        row.append(hess_alpha_gamma)
        blocks.append(row)
        
    if has_beta:
        row = [hess_alpha_beta.T] if has_alpha else []
        row.append(hess_beta_beta)
        row.append(hess_beta_gamma)
        blocks.append(row)
        
    # Gamma row
    row = [hess_alpha_gamma.T] if has_alpha else []
    if has_beta: row.append(hess_beta_gamma.T)
    row.append(hess_gamma_gamma)
    blocks.append(row)
    
    hess_total = np.block(blocks)
    
    if include_tau:
        hess_total = tau * hess_total
        omega_meat = tau * omega_meat
    
    fisher_inv = np.linalg.inv(hess_total)


    sandwich_estimator = (1 / k) * fisher_inv @ omega_meat @ fisher_inv.T
    
    return sandwich_estimator
