"""Measured quantities, not fitted closures. All calculations use float64."""

import numpy as np


def probabilities(values):
    p = np.asarray(values, dtype=np.float64)
    if p.ndim != 1 or not p.size or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("population weights must be a finite nonnegative vector")
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("population weights must have positive finite mass")
    return p / total


def moments(scores, population_weights=None):
    z = np.asarray(scores, dtype=np.float64)
    if z.ndim != 1 or not z.size or not np.isfinite(z).all():
        raise ValueError("scores must be a nonempty finite vector")
    p = probabilities(np.ones(z.size) if population_weights is None else population_weights)
    if p.shape != z.shape:
        raise ValueError("scores and population weights must align")
    # Center before summing to reduce loss of precision under trait translations.
    mu = float(z[0] + np.dot(p, z - z[0]))
    x = z - mu
    with np.errstate(over="raise", invalid="raise"):
        v, m3, m4 = (float(np.dot(p, x ** k)) for k in (2, 3, 4))
    return {"mu": mu, "V": v, "M3": m3, "M4": m4,
            "skewness": m3 / v ** 1.5 if v > 0 else None,
            "zero_fraction": float(p[z == 0].sum())}


def selection_fields(mu, variance, covariance, binary=False):
    """Numerically undefined coefficients remain missing; no statistical mask is implied."""
    if not all(np.isfinite(x) for x in (mu, variance, covariance)) or variance < 0:
        raise ValueError("invalid state or covariance")
    if binary and not 0 <= mu <= 1:
        raise ValueError("binary prevalence must be in [0, 1]")
    return {"beta": covariance / variance if variance > 0 else None,
            "S": covariance / (mu * (1 - mu)) if binary and 0 < mu < 1 else None,
            "coefficient_status": "defined_precision_not_assessed" if variance > 0 else "zero_variance"}


def flux_statistics(scores, omega, population_weights=None):
    """Plug-in covariances; likelihood ratios are NEVER normalized to mean one."""
    z, w = np.asarray(scores, dtype=float), np.asarray(omega, dtype=float)
    if w.shape != z.shape or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("likelihood ratios must be finite, nonnegative, and aligned")
    p = probabilities(np.ones(z.size) if population_weights is None else population_weights)
    out = moments(z, p)
    x = z - out["mu"]
    mean_w = float(np.dot(p, w))
    if mean_w <= 0:
        raise ValueError("likelihood ratios have no positive mass")
    centered_w = w - mean_w
    c = float(np.dot(p, centered_w * x))
    q = float(np.dot(p, centered_w * x ** 2))
    out.update(selection_fields(out["mu"], out["V"], c))
    out.update(C=c, Q=q, mean_omega=mean_w,
               C_known_normalizer=float(np.dot(p, (w - 1) * z)),
               C_self_normalized=c / mean_w,
               omega_max=float(w.max()),
               omega_second_moment=float(np.dot(p, w * w)),
               Q_minus_beta_M3=q - out["beta"] * out["M3"] if out["beta"] is not None else None,
               variance_change_from_flux=q - c * c)
    return out


def enumerated_transition(scores, old_weights, new_weights):
    """Exact finite-distribution fluxes; detect support loss in stored weights."""
    p, q = probabilities(old_weights), probabilities(new_weights)
    if p.shape != q.shape:
        raise ValueError("old/new population weights must align")
    old, new = moments(scores, p), moments(scores, q)
    x = np.asarray(scores, dtype=float) - old["mu"]
    c, second_flux = float(np.dot(q - p, x)), float(np.dot(q - p, x * x))
    old.update(selection_fields(old["mu"], old["V"], c))
    old.update(C=c, Q=second_flux, mu_next=new["mu"], V_next=new["V"],
               direct_delta=new["mu"] - old["mu"],
               accounting_residual=new["mu"] - old["mu"] - c,
               variance_change_from_flux=second_flux - c * c,
               variance_identity_residual=new["V"] - old["V"] - second_flux + c * c,
               Q_minus_beta_M3=second_flux - old["beta"] * old["M3"] if old["beta"] is not None else None,
               support_failure=bool(np.any((p == 0) & (q > 0))))
    return old


def conditional_bins(scores, omega, population_weights=None, count=10):
    """Weighted-quantile bin means; descriptive, with no IID error-bar assumption."""
    z, w = np.asarray(scores, dtype=float), np.asarray(omega, dtype=float)
    stats = flux_statistics(z, w, population_weights)
    p = probabilities(np.ones(z.size) if population_weights is None else population_weights)
    active = p > 0
    order = np.argsort(z[active], kind="stable")
    zs, ps = z[active][order], p[active][order]
    edges = np.unique(np.interp(np.linspace(0, 1, count + 1)[1:-1], np.cumsum(ps), zs))
    assignment = np.searchsorted(edges, z, side="right")
    result = []
    for index in range(len(edges) + 1):
        mask = (assignment == index) & active
        mass = float(p[mask].sum())
        if not mass:
            continue
        zm = float(np.dot(p[mask], z[mask]) / mass)
        wm = float(np.dot(p[mask], w[mask]) / mass)
        predicted = 1 + stats["beta"] * (zm - stats["mu"]) if stats["beta"] is not None else None
        result.append({"bin": index, "score_mean": zm, "omega_mean": wm,
                       "population_mass": mass, "sample_count": int(mask.sum()),
                       "linear_prediction": predicted,
                       "conditional_residual": wm - predicted if predicted is not None else None})
    return result
