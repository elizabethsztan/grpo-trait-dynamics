import json
import subprocess
import sys

import numpy as np
import pytest
import yaml

from closure_study.adapters import activation, output_llm, simple
from closure_study.io import inventory_entry, load_registry, sha256
from closure_study.pipeline import run


def entry(family, **kwargs):
    return {"run_id": family, "family": family, "setting_id": "test-setting",
            "inspection_status": "development", **kwargs}


def output_fixture(path):
    rows = []
    for step in range(7):
        direct = {"eval_wrong_hint": {"agreement_rate": .2 + .01 * step}} if step % 3 == 0 else {}
        price = {"eval_wrong_hint": {"output_agreement": {"cov_step": .01, "n": 20}}} if step else {}
        rows.append({"step": step, "observed_eval": direct, "price": price})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return entry("llm_binary", metrics_path=str(path))


def test_sparse_binary_states_are_not_interpolated_or_time_shifted(tmp_path):
    e = output_fixture(tmp_path / "metrics.jsonl")
    rows, bins, checks = output_llm(e)
    assert [r["step"] for r in rows if "S" in r] == [0, 3]
    np.testing.assert_allclose(rows[0]["S"], .01 / (.2 * .8))
    assert all(r["interval"] == 1 for r in rows)
    assert [r["interval"] for r in checks] == [3, 3]
    np.testing.assert_allclose([r["C_sum"] for r in checks], [.03, .03])
    assert max(abs(r["residual"]) for r in checks) < 1e-15
    assert not bins


def neural_fixture(path):
    z = np.array([-1., 0., 2.])
    p = np.array([[.2, .5, .3], [.18, .48, .34], [.16, .46, .38], [.14, .44, .42]])
    mu = p @ z
    np.savez(path, format_version=2, mode="neural", price_check=True, seeds=[123],
             observed_trait=mu[None, :], exact_price_increments=np.diff(mu)[None, :],
             particle_scores=z[None, :], particle_weights=p[None, :-1, :])
    return entry("neural_continuous", trajectories_path=str(path))


def test_neural_moments_and_missing_terminal_weights(tmp_path):
    e = neural_fixture(tmp_path / "neural.npz")
    rows, bins, checks = simple(e)
    assert len(rows) == 3 and len(checks) == 3
    assert rows[0]["V"] > 0
    assert abs(rows[0]["variance_identity_residual"]) < 1e-14
    assert "Q" not in rows[-1] and "V_next" not in rows[-1]
    assert rows[-1]["beta"] is not None
    assert rows[0]["run_id"].endswith("seed-123")
    assert bins


def sae_fixture(path):
    cp = np.array([0, 2, 7])
    z = np.array([[[0.], [1.], [2.]], [[0.], [2.], [4.]]])
    w = np.array([[.5, 1., 1.5], [.8, 1., 1.2]])
    np.savez(path, s=z, omega=w, feat_ids=[17], is_control=[False], steps=cp[:-1], ckpt_steps=cp)
    return entry("llm_continuous", pool_path=str(path))


def test_nonunit_intervals_preserved_without_coefficient_rescaling(tmp_path):
    e = sae_fixture(tmp_path / "pool.npz")
    rows, _, _ = activation(e)
    assert [r["interval"] for r in rows] == [2, 5]
    np.testing.assert_allclose(rows[0]["beta"], .5)
    assert rows[0]["direct_delta"] is None
    assert "mu_next" not in rows[0]


def test_direct_feature_misalignment_rejected(tmp_path):
    e = sae_fixture(tmp_path / "pool.npz")
    direct = tmp_path / "direct.npz"
    np.savez(direct, checkpoint_steps=[0, 2, 7], feat_ids=[18], direct_means=np.ones((3, 1)))
    with pytest.raises(ValueError, match="checkpoints/features"):
        activation({**e, "direct_path": str(direct)})


def test_missing_endpoint_metadata_cannot_be_assumed_unit_steps(tmp_path):
    e = sae_fixture(tmp_path / "pool.npz")
    with np.load(e["pool_path"]) as d:
        values = {k: d[k] for k in d.files if k != "ckpt_steps"}
    np.savez(e["pool_path"], **values)
    with pytest.raises(KeyError):
        activation(e)


def test_legacy_endpoints_require_pool_hash_and_checkpoint_bank(tmp_path):
    e = sae_fixture(tmp_path / "pool.npz")
    with np.load(e["pool_path"]) as d:
        values = {k: d[k] for k in d.files if k != "ckpt_steps"}
    np.savez(e["pool_path"], **values)
    for step in [0, 2, 7]:
        (tmp_path / "checkpoints" / f"step_{step:04d}").mkdir(parents=True)
    evidence = tmp_path / "endpoints.json"
    evidence.write_text(json.dumps({"checkpoint_steps": [0, 2, 7], "pool_sha256": sha256(e["pool_path"])}))
    e["endpoints_path"] = str(evidence)
    rows, _, _ = activation(e)
    assert [r["interval"] for r in rows] == [2, 5]
    evidence.write_text(json.dumps({"checkpoint_steps": [0, 2, 7], "pool_sha256": "wrong"}))
    with pytest.raises(ValueError, match="different pool"):
        activation(e)


def test_untouched_registry_entries_rejected_before_data_access(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"schema_version": 1, "runs": [{"run_id": "sealed", "inspection_status": "holdout",
                                                             "metrics_path": "/does/not/exist"}]}))
    with pytest.raises(ValueError, match="non-development"):
        load_registry(path)


@pytest.mark.parametrize("family,config,learning_rate,eta", [
    ("tabular_binary", {"TrainConfig": {"eta": .3}}, None, .3),
    ("neural_continuous", {"TrainConfig": {"eta": .3}, "NeuralTrainConfig": {"lr": .02}}, .02, None),
    ("llm_binary", {"TrainConfig": {"learning_rate": 1e-5}}, 1e-5, None),
    ("llm_continuous", {"GRPOConfig": {"learning_rate": 1e-4}}, 1e-4, None),
])
def test_inventory_reads_family_update_parameters(tmp_path, family, config, learning_rate, eta):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    record = inventory_entry(entry(family, config_path=str(path)))
    assert record["learning_rate"] == learning_rate
    assert record["eta"] == eta


def test_end_to_end_all_four_families_preserves_sources_and_stops_at_diagnostics(tmp_path):
    binary = tmp_path / "binary.npz"
    np.savez(binary, format_version=1, mode="tabular", price_check=True, seeds=[9],
             observed_trait=[[.2, .3, .4]], exact_price_increments=[[.1, .1]])
    entries = [entry("tabular_binary", trajectories_path=str(binary)),
               neural_fixture(tmp_path / "neural.npz"), output_fixture(tmp_path / "metrics.jsonl"),
               sae_fixture(tmp_path / "pool.npz")]
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"schema_version": 1, "runs": entries}))
    sources = {str(p): sha256(p) for p in tmp_path.iterdir() if p.is_file()}
    output = tmp_path / "diagnostics"
    subprocess.run([sys.executable, "-m", "closure_study", "--registry", str(registry),
                    "--output", str(output)], check=True, capture_output=True, text=True)
    assert (output / "index.html").is_file()
    assert (output / "transitions.csv").is_file()
    assert (output / "source_issues.jsonl").read_text() == ""
    text = (output / "report.md").read_text()
    assert "No closure has been fitted or selected" in text
    assert all(sha256(p) == digest for p, digest in sources.items())
    with pytest.raises(ValueError, match="never overwritten"):
        run(registry, output)
