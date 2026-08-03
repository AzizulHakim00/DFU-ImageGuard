from pathlib import Path

from src import reliable_storage_v2 as storage


def test_metadata_backup_excludes_model_weights(tmp_path: Path):
    run = tmp_path / "primary" / "runs" / "V2"
    backup = tmp_path / "secondary"
    trial = run / "trials" / "model" / "seed_1" / "fold_1"
    trial.mkdir(parents=True)
    (trial / "best_model_portable_fp16.pt").write_bytes(b"model-bytes")
    (trial / "COMPLETE.json").write_text('{"status":"ok"}', encoding="utf-8")
    (trial / "test_predictions.csv").write_text("label,pred\n1,1\n", encoding="utf-8")

    status = storage.backup_metadata(run, backup, "V2")

    target = backup / "runs" / "V2"
    assert status["verified"] is True
    assert (target / "trials" / "model" / "seed_1" / "fold_1" / "COMPLETE.json").exists()
    assert not (target / "trials" / "model" / "seed_1" / "fold_1" / "best_model_portable_fp16.pt").exists()


def test_active_backup_is_rolling_and_bounded(tmp_path: Path):
    run = tmp_path / "primary" / "runs" / "V2"
    backup = tmp_path / "secondary"
    trial_a = run / "trials" / "a"
    trial_b = run / "trials" / "b"
    for trial, marker in ((trial_a, b"a"), (trial_b, b"b")):
        trial.mkdir(parents=True)
        (trial / "last_resume.pt").write_bytes(marker)
        (trial / "best_model.pt").write_bytes(marker + b"best")
        (trial / "history.csv").write_text("epoch\n1\n", encoding="utf-8")

    storage.backup_active_trial(run, backup, "V2", trial_a)
    storage.backup_active_trial(run, backup, "V2", trial_b)

    active = backup / "runs" / "V2" / "_active_trial"
    assert (active / "last_resume.pt").read_bytes() == b"b"
    assert len(list(active.glob("*.pt"))) == 2


def test_quota_failure_is_degraded_not_raised(tmp_path: Path, monkeypatch):
    run = tmp_path / "primary" / "runs" / "V2"
    backup = tmp_path / "secondary"
    run.mkdir(parents=True)
    (run / "RUN_PROGRESS.json").write_text("{}", encoding="utf-8")

    def fail_copy(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(storage, "_verified_copy", fail_copy)
    status = storage.backup_metadata(run, backup, "V2")

    assert status["degraded"] is True
    assert status["training_may_continue"] is True
    assert (run / storage.STATUS_FILE).exists()
