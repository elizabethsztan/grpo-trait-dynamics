import json

from src import plotting


def test_reliability_sweep_accuracy_plot_includes_no_hint_reference(tmp_path, monkeypatch):
    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    (sweep_dir / "summary.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "train_hint_correct_probability": 0.0,
                        "final_wrong_hint_agreement_rate": 0.1,
                        "final_sycophantic_error_rate": 0.1,
                        "final_output_agreement_price_cum": -0.2,
                        "final_wrong_hint_accuracy": 0.3,
                        "final_no_hint_accuracy": 0.42,
                    },
                    {
                        "train_hint_correct_probability": 0.9,
                        "final_wrong_hint_agreement_rate": 1.0,
                        "final_sycophantic_error_rate": 1.0,
                        "final_output_agreement_price_cum": 0.7,
                        "final_wrong_hint_accuracy": 0.0,
                        "final_no_hint_accuracy": 0.44,
                    },
                ]
            }
        )
    )
    calls = []

    def fake_save(path, x, series, ylabel, xlabel="GRPO step"):
        calls.append((path.name, x, series, ylabel, xlabel))

    monkeypatch.setattr(plotting, "_save_line_plot", fake_save)

    plotting.plot_reliability_sweep(sweep_dir)

    accuracy_call = next(call for call in calls if call[0] == "reliability_sweep_accuracy.png")
    assert accuracy_call[1] == [0.0, 0.9]
    assert accuracy_call[2] == [
        ("Wrong-hint accuracy", [0.3, 0.0]),
        ("No-hint accuracy reference", [0.42, 0.44]),
    ]
    assert accuracy_call[4] == "Training hint correctness probability"
