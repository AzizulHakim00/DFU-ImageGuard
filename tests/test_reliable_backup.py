import errno
from pathlib import Path

from src import reliable_backup


def test_completed_backup_uses_portable_checkpoint_only(tmp_path):
    run = tmp_path / "primary" / "runs" / "R1"
    trial = run / "trials" / "convnextv2_tiny" / "seed_2026" / "fold_1"
    trial.mkdir(parents=True)
    (trial / "COMPLETE.json").write_text("{}", encoding="utf-8")
    (trial / "history.csv").write_text("epoch,loss\n1,0.1\n", encoding="utf-8")
    (trial / "test_predictions.csv").write_text("label,pred\n1,1\n", encoding="utf-8")
    (trial / "best_model.pt").write_bytes(b"full-best")
    (trial / "last_resume.pt").write_bytes(b"full-resume")
    (trial / "best_model_portable_fp16.pt").write_bytes(b"portable")

    status = reliable_backup.backup_completed_artifacts(
        run=run,
        backup_root=tmp_path / "secondary",
        run_id="R1",
    )
    target = tmp_path / "secondary" / "runs" / "R1"
    copied_trial = target / "trials" / "convnextv2_tiny" / "seed_2026" / "fold_1"

    assert status["verified"] is True
    assert (copied_trial / "best_model_portable_fp16.pt").read_bytes() == b"portable"
    assert not (copied_trial / "best_model.pt").exists()
    assert not (copied_trial / "last_resume.pt").exists()


def test_epoch_backup_quota_failure_is_nonfatal(tmp_path, monkeypatch):
    run = tmp_path / "primary" / "runs" / "R2"
    trial = run / "trials" / "densenet121" / "seed_2028" / "fold_1"
    trial.mkdir(parents=True)
    (trial / "last_resume.pt").write_bytes(b"resume")
    (trial / "best_model.pt").write_bytes(b"best")
    (trial / "history.csv").write_text("epoch\n7\n", encoding="utf-8")

    def fail_copy(src: Path, dst: Path):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(reliable_backup, "_copy_verified", fail_copy)
    status = reliable_backup.backup_epoch_state(
        run=run,
        backup_root=tmp_path / "secondary",
        run_id="R2",
        trial=trial,
    )

    assert status["degraded"] is True
    assert status["training_may_continue"] is True
    assert (run / "SECONDARY_BACKUP_STATUS.json").exists()
