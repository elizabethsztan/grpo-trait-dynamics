"""Read-only adapters for the explicitly selected historical experiment archives.

The legacy forecast adapters establish the archive layouts; this module keeps
missing states, interval lengths, and independent observations explicit instead
of reconstructing levels by accumulating covariance or interpolating checkpoints.
"""

from pathlib import Path
import json

import numpy as np

from .io import read_jsonl, sha256, steps
from .statistics import (conditional_bins, enumerated_transition, flux_statistics,
                         moments, probabilities, selection_fields)


def base(entry, run_id, trait_id, start, end, **extra):
    if end <= start:
        raise ValueError("transition end must follow start")
    return {"schema_version": 1, "run_id": run_id, "family": entry["family"],
            "setting_id": entry["setting_id"], "trait_id": trait_id,
            "inspection_status": entry["inspection_status"], "step": int(start),
            "step_end": int(end), "interval": int(end - start),
            "randomness_status": entry.get("randomness_status", "independence_unverified"),
            "fixed_trait_status": entry.get("fixed_trait_status", "fixed_by_definition"),
            **extra}


def binary_moments(mu):
    if not np.isfinite(mu) or not 0 <= mu <= 1:
        raise ValueError("invalid binary prevalence")
    v = mu * (1 - mu)
    return {"T": mu, "mu": mu, "V": v, "M3": v * (1 - 2 * mu),
            "M4": v * (1 - 3 * v)}


def _finite(array, name):
    if not np.isfinite(array).all():
        raise ValueError(f"non-finite {name}")


def simple(entry):
    rows, bins, accounting = [], [], []
    with np.load(entry["trajectories_path"], allow_pickle=False) as data:
        version = int(data["format_version"].item())
        if version not in (1, 2) or not bool(data["price_check"].item()):
            raise ValueError("unsupported or unmeasured simple-theory archive")
        mode = str(data["mode"].item())
        binary = entry["family"] == "tabular_binary"
        if mode != ("tabular" if binary else "neural"):
            raise ValueError("archive mode differs from registry family")
        seeds = data["seeds"]
        observed = np.asarray(data["observed_trait"], dtype=float)
        covs = np.asarray(data["exact_price_increments"], dtype=float)
        if observed.ndim != 2 or covs.shape != (len(seeds), observed.shape[1] - 1):
            raise ValueError("simple-theory state and transition shapes do not align")
        if len(set(seeds.tolist())) != len(seeds) or observed.shape[0] != len(seeds):
            raise ValueError("simple-theory seed identities do not align")
        _finite(observed, "trait means")
        _finite(covs, "Price increments")
        scores = data["particle_scores"] if "particle_scores" in data else None
        weights = data["particle_weights"] if "particle_weights" in data else None
        if (scores is None) != (weights is None):
            raise ValueError("particle scores and weights must be supplied together")
        if weights is not None and (weights.ndim != 3 or scores.shape != (len(seeds), weights.shape[2])
                                    or weights.shape[:2] != covs.shape):
            raise ValueError("particle archive shapes do not align")
        if scores is not None:
            _finite(scores, "particle scores")
            _finite(weights, "particle weights")
        for ri, seed in enumerate(seeds):
            rid = f"{entry['run_id']}:seed-{int(seed)}"
            n = covs.shape[1]
            chosen = {0, max(0, (n - 2) // 2), n - 2}
            for t in range(n):
                c = float(covs[ri, t])
                mu, next_mu = (float(observed[ri, i]) for i in (t, t + 1))
                row = base(entry, rid, "trait", t, t + 1, C=c, mu=mu,
                           direct_delta=next_mu - mu, mu_next=next_mu,
                           accounting_residual=next_mu - mu - c,
                           estimator="enumerated_archive", uncertainty_status="finite_population_enumeration",
                           state_source="enumerated_policy", covariance_panel="enumerated_policy",
                           accounting_basis="exact_finite_distribution")
                if binary:
                    row.update(binary_moments(mu))
                    row.update(selection_fields(mu, row["V"], c, binary=True))
                    q = c * (1 - 2 * mu)
                    row.update(Q=q, Q_origin="binary_algebraic_identity",
                               V_next=next_mu * (1 - next_mu),
                               variance_identity_residual=next_mu * (1 - next_mu) - row["V"] - q + c * c,
                               coefficient_status="exact_finite_population" if row["V"] > 0 else "zero_variance")
                elif weights is not None:
                    z, p = scores[ri], probabilities(weights[ri, t])
                    state = moments(z, p)
                    row.update(state)
                    row.update(selection_fields(state["mu"], state["V"], c))
                    row["state_rounding_residual"] = state["mu"] - mu
                    row["stored_price_C"] = c
                    row["Q_origin"] = "enumerated_population_weights"
                    if t + 1 < n:
                        q = probabilities(weights[ri, t + 1])
                        row.update(enumerated_transition(z, p, q))
                        row["stored_C_residual"] = row["C"] - c
                        if not row["support_failure"] and t in chosen:
                            w = np.divide(q, p, out=np.zeros_like(p), where=p > 0)
                            for b in conditional_bins(z, w, p):
                                bins.append({**base(entry, rid, "trait", t, t + 1), **b,
                                             "panel_kind": "enumerated_policy"})
                    else:
                        row["missing_reason"] = "next_policy_weights_not_saved; Q and next variance unavailable"
                    row["coefficient_status"] = "finite_population_with_float32_storage" if row["V"] > 0 else "zero_variance"
                else:
                    row["missing_reason"] = "particle_weights_not_saved; variance and beta unavailable"
                rows.append(row)
                accounting.append({**base(entry, rid, "trait", t, t + 1),
                                   "observed_delta": next_mu - mu, "C_sum": c,
                                   "residual": next_mu - mu - c,
                                   "comparison": "enumerated_state_vs_archived_C",
                                   "uncertainty_status": "finite_population_enumeration"})
    return rows, bins, accounting


def output_llm(entry):
    """Sparse direct means are matched only at their actual checkpoint indices."""
    raw = read_jsonl(entry["metrics_path"])
    indices = steps([r["step"] for r in raw])
    if not np.array_equal(indices, np.arange(len(raw))):
        raise ValueError("legacy output logs must be contiguous, with step j meaning j-1 -> j")
    distribution = entry.get("distribution", "eval_wrong_hint")
    direct = {int(r["step"]): float(r["observed_eval"][distribution]["agreement_rate"])
              for r in raw if distribution in r.get("observed_eval", {})}
    for mu in direct.values():
        binary_moments(mu)
    rows, accounting, cs = [], [], {}
    for record in raw[1:]:
        end = int(record["step"])
        item = record.get("price", {}).get(distribution, {}).get("output_agreement")
        if item is None:
            continue
        c = float(item["cov_step"])
        _finite(np.array([c]), "output covariance")
        cs[end] = c
        row = base(entry, entry["run_id"], "output_agreement", end - 1, end,
                   C=c, estimator="historical_plugin_covariance", distribution=distribution,
                   covariance_panel="independent_old_policy_price_rollouts",
                   uncertainty_status="unavailable_raw_responses_and_cluster_ids_not_saved",
                   accounting_basis="independent_means_at_logged_checkpoints",
                   measurement_status="historical_sampling_scoring_audit_required",
                   n_price=item.get("n"), mean_omega=item.get("mean_omega"),
                   omega_max=item.get("max_omega"), ess=item.get("ess"))
        if end - 1 in direct:
            mu = direct[end - 1]
            row.update(binary_moments(mu))
            row.update(selection_fields(mu, row["V"], c, binary=True))
            row.update(state_source="independent_direct_old_checkpoint",
                       Q=c * (1 - 2 * mu), Q_origin="binary_algebraic_from_separate_state_estimate")
        else:
            row["missing_reason"] = "no_direct_old_checkpoint_state; no_interpolation"
        if end in direct:
            row["mu_next"] = direct[end]
        rows.append(row)
    ordered = sorted(direct)
    for start, end in zip(ordered, ordered[1:]):
        available = all(i in cs for i in range(start + 1, end + 1))
        total = sum(cs[i] for i in range(start + 1, end + 1)) if available else None
        delta = direct[end] - direct[start]
        accounting.append({**base(entry, entry["run_id"], "output_agreement", start, end),
                           "observed_delta": delta, "C_sum": total,
                           "residual": delta - total if total is not None else None,
                           "comparison": "independent_direct_change_vs_sum_of_unit_C",
                           "uncertainty_status": "unavailable_joint_sampling_covariance"})
    return rows, [], accounting


def activation(entry):
    rows, bins, accounting = [], [], []
    with np.load(entry["pool_path"], allow_pickle=False) as pool:
        z = np.asarray(pool["s"], dtype=float)
        w = np.asarray(pool["omega"], dtype=float)
        ids, controls = pool["feat_ids"], pool["is_control"]
        old_steps = steps(pool["steps"])
        # Explicit endpoints are required; never assume the next step was +1.
        endpoint_basis = "archive_checkpoint_steps"
        if "ckpt_steps" in pool:
            cp = steps(pool["ckpt_steps"])
        elif entry.get("endpoints_path"):
            evidence = json.loads(Path(entry["endpoints_path"]).read_text())
            if evidence["pool_sha256"] != sha256(entry["pool_path"]):
                raise ValueError("endpoint supplement belongs to a different pool")
            cp = steps(evidence["checkpoint_steps"])
            bank = Path(entry["pool_path"]).parent / "checkpoints"
            actual = sorted(int(p.name.split("_")[1]) for p in bank.glob("step_*"))
            if actual != cp.tolist():
                raise ValueError("endpoint supplement does not match the saved checkpoint bank")
            endpoint_basis = "explicit_pool_hash_bound_checkpoint_bank_supplement"
        else:
            raise KeyError("ckpt_steps missing; no verified endpoint supplement supplied")
        if z.ndim != 3 or w.shape != z.shape[:2] or old_steps.shape != (len(w),):
            raise ValueError("SAE score/ratio/step shapes do not align")
        if ids.shape != (z.shape[2],) or controls.shape != ids.shape or len(set(ids.tolist())) != len(ids):
            raise ValueError("SAE feature identities do not align")
        if cp.size != len(w) + 1 or not np.array_equal(old_steps, cp[:-1]):
            raise ValueError("SAE checkpoint endpoints do not align")
        _finite(z, "SAE scores")
        _finite(w, "SAE ratios")
        direct = {}
        if entry.get("direct_path") and Path(entry["direct_path"]).is_file():
            with np.load(entry["direct_path"], allow_pickle=False) as d:
                if not np.array_equal(d["checkpoint_steps"], cp) or not np.array_equal(d["feat_ids"], ids):
                    raise ValueError("direct mean checkpoints/features differ from pool")
                dm = np.asarray(d["direct_means"], dtype=float)
                if dm.shape != (len(cp), len(ids)):
                    raise ValueError("direct mean matrix has incompatible shape")
                _finite(dm, "direct means")
                direct = {(int(cp[t]), int(fid)): float(dm[t, f])
                          for t in range(len(cp)) for f, fid in enumerate(ids)}
        logged = {}
        if entry.get("price_eval_path") and Path(entry["price_eval_path"]).is_file():
            for item in read_jsonl(entry["price_eval_path"]):
                key = (int(item["step"]), int(item["feature_id"]))
                if key not in logged or item["N"] > logged[key]["N"]:
                    logged[key] = item
        chosen = {0, len(w) // 2, len(w) - 1}
        for t, (start, end) in enumerate(zip(cp, cp[1:])):
            for f, fid in enumerate(ids):
                row = base(entry, entry["run_id"], str(int(fid)), start, end,
                           endpoint_basis=endpoint_basis,
                           is_control=bool(controls[f]), n_price=z.shape[1],
                           estimator="raw_pool_plugin_covariance", state_source="same_old_policy_price_panel",
                           covariance_panel="old_policy_pool", distribution=entry.get("distribution", "eval"),
                           uncertainty_status="unavailable_prompt_clusters_and_direct_responses_not_saved",
                           measurement_status="historical_sampling_scoring_audit_required")
                row.update(flux_statistics(z[t, :, f], w[t]))
                row["ess"] = float(w[t].sum() ** 2 / np.dot(w[t], w[t]))
                row["Q_origin"] = "measured_raw_pool"
                # This is an empirical projection residual, not a fitted dynamics model.
                row["linear_negative_mass"] = (float(np.mean(1 + row["beta"] * (z[t, :, f] - row["mu"]) < 0))
                                                if row["beta"] is not None else None)
                delta, comparison = None, "direct_observations_missing"
                if (int(start), int(fid)) in direct and (int(end), int(fid)) in direct:
                    row["direct_mu_old"] = direct[(int(start), int(fid))]
                    row["direct_mu_new"] = direct[(int(end), int(fid))]
                    delta = row["direct_mu_new"] - row["direct_mu_old"]
                    comparison = "separate_direct_panel_aggregate_only"
                elif (int(start), int(fid)) in logged:
                    item = logged[(int(start), int(fid))]
                    logged_end = item.get("step_end")
                    # Old unit-step exports lack an end index; accept only verified unit pairs.
                    if logged_end == int(end) or (logged_end is None and end == start + 1):
                        delta = float(item["direct_drift"])
                        comparison = "historical_rounded_direct_drift_panel_provenance_unverified"
                row["direct_delta"] = delta
                row["accounting_residual"] = delta - row["C"] if delta is not None else None
                row["accounting_basis"] = comparison
                rows.append(row)
                accounting.append({**base(entry, entry["run_id"], str(int(fid)), start, end),
                                   "observed_delta": delta, "C_sum": row["C"],
                                   "residual": row["accounting_residual"], "comparison": comparison,
                                   "uncertainty_status": row["uncertainty_status"]})
                if t in chosen:
                    for b in conditional_bins(z[t, :, f], w[t]):
                        bins.append({**base(entry, entry["run_id"], str(int(fid)), start, end),
                                     **b, "panel_kind": "sampled_old_policy_pool_no_cluster_intervals"})
    return rows, bins, accounting


LOADERS = {"tabular_binary": simple, "neural_continuous": simple,
           "llm_binary": output_llm, "llm_continuous": activation}
