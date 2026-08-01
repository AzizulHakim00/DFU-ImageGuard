from pathlib import Path

path = Path("polarmorphnet/DFU_PolarMorphNet_AllInOne.py")
text = path.read_text(encoding="utf-8")
text = text.replace('bbox_inches_inches="tight"', 'bbox_inches="tight"')
text = text.replace('f"HEAD:{branch}", "--force-with-lease"', 'f"HEAD:{branch}", "--force"')
old = '''    verification = finalize_run(manifest, cfg, dirs)\n    (project / "LAST_COMPLETED_POLARMORPHNET_RUN.txt").write_text(cfg.RUN_ID + "\\n")\n    (project / "ACTIVE_POLARMORPHNET_RUN.txt").unlink(missing_ok=True)\n    return verification\n'''
new = '''    final_lock = dirs["locks"] / "finalize.lock"\n    try:\n        descriptor = os.open(final_lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)\n        os.close(descriptor)\n    except FileExistsError:\n        print("Another Colab account is performing final aggregation; no duplicate finalization started.")\n        return progress\n    try:\n        verification = finalize_run(manifest, cfg, dirs)\n        (project / "LAST_COMPLETED_POLARMORPHNET_RUN.txt").write_text(cfg.RUN_ID + "\\n")\n        (project / "ACTIVE_POLARMORPHNET_RUN.txt").unlink(missing_ok=True)\n        return verification\n    finally:\n        final_lock.unlink(missing_ok=True)\n'''
if old not in text:
    raise SystemExit("Expected train_mode finalization block not found")
text = text.replace(old, new)
path.write_text(text, encoding="utf-8")
print(path)
