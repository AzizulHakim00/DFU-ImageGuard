from __future__ import annotations
import base64, dataclasses, datetime as dt, gc, hashlib, json, math, os, pickle, platform, random, shutil, subprocess, sys, time, traceback, warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
warnings.filterwarnings("ignore", category=UserWarning)

from .config_data import *
from .models_training import *
from .evaluation import *
from .statistics_figures import savefig_multi

def corruption_fn(name: str, level: float, image_id: str) -> Callable[[Image.Image], Image.Image]:
    rng = np.random.default_rng(hash_text(f"{image_id}:{name}:{level}"))
    def apply(im: Image.Image) -> Image.Image:
        im = im.convert("RGB")
        if name == "gaussian_noise":
            arr = np.asarray(im).astype(np.float32) / 255.0
            arr = np.clip(arr + rng.normal(0, float(level), arr.shape), 0, 1)
            return Image.fromarray((arr * 255).astype(np.uint8))
        if name == "gaussian_blur": return im.filter(ImageFilter.GaussianBlur(radius=float(level)))
        if name == "brightness": return ImageEnhance.Brightness(im).enhance(float(level))
        if name == "contrast": return ImageEnhance.Contrast(im).enhance(float(level))
        if name == "jpeg":
            import io
            buf = io.BytesIO(); im.save(buf, format="JPEG", quality=int(level)); buf.seek(0)
            return Image.open(buf).convert("RGB")
        if name == "rotation": return im.rotate(float(level), resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
        if name == "occlusion":
            arr = np.asarray(im).copy(); h, w = arr.shape[:2]
            side = int(math.sqrt(float(level)) * min(h, w))
            y = int(rng.integers(0, max(1, h - side + 1))); x = int(rng.integers(0, max(1, w - side + 1)))
            arr[y:y + side, x:x + side] = np.median(arr.reshape(-1, 3), axis=0)
            return Image.fromarray(arr.astype(np.uint8))
        return im
    return apply


def run_robustness(manifest: pd.DataFrame, predictions: pd.DataFrame, cfg: Config, dirs: dict[str, Path]) -> pd.DataFrame:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_tf = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    class CDataset(Dataset):
        def __init__(self, frame, corruption, level):
            self.frame = frame.reset_index(drop=True); self.corruption = corruption; self.level = level
        def __len__(self): return len(self.frame)
        def __getitem__(self, i):
            row = self.frame.iloc[i]
            with Image.open(row.image_path) as im:
                im = corruption_fn(self.corruption, self.level, row.image_id)(im); x = base_tf(im)
            return x, torch.tensor(float(row.label)), i
    model_factories: dict[str, Callable[[], Any]] = {cfg.PRIMARY_MODEL_NAME: lambda: build_proposed_model(cfg)}
    for name, timm_name in cfg.BASELINE_MODELS.items():
        model_factories[name] = lambda n=timm_name: build_baseline_model(n)
    rows = []
    for model_name, factory in model_factories.items():
        safe = model_name.lower().replace("-", "_").replace(" ", "_")
        model_pred = predictions[predictions.model == model_name]
        for corruption, levels in cfg.ROBUSTNESS_LEVELS.items():
            for level in levels:
                pieces = []
                for fold in range(cfg.N_FOLDS):
                    fold_frame = manifest[manifest.outer_fold == fold].copy().reset_index(drop=True)
                    ckpt = dirs["models"] / (f"dfu_imageguard_fold_{fold + 1}.pt" if model_name == cfg.PRIMARY_MODEL_NAME else f"{safe}_fold_{fold + 1}.pt")
                    if not ckpt.exists(): continue
                    model = load_checkpoint_model(factory, ckpt, device)
                    loader = DataLoader(CDataset(fold_frame, corruption, level), batch_size=cfg.BATCH_SIZE,
                                        shuffle=False, num_workers=cfg.NUM_WORKERS)
                    logits, y, _ = predict_logits(model, loader, device)
                    fold_meta = model_pred[model_pred.outer_fold == fold].iloc[0]
                    t, thr = float(fold_meta.temperature), float(fold_meta.threshold)
                    p = 1 / (1 + np.exp(-np.clip(logits / t, -40, 40)))
                    pieces.append(pd.DataFrame({"label": y, "prob": p, "pred": (p >= thr).astype(int)}))
                    del model; gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                if not pieces: continue
                z = pd.concat(pieces, ignore_index=True)
                m = metric_dict(z.label, z.prob, 0.5)
                from sklearn.metrics import confusion_matrix, balanced_accuracy_score, fbeta_score
                tn, fp, fn, tp = confusion_matrix(z.label, z.pred, labels=[0, 1]).ravel()
                m.update({
                    "accuracy": float((z.pred == z.label).mean()),
                    "balanced_accuracy": balanced_accuracy_score(z.label, z.pred),
                    "recall_sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
                    "f1": fbeta_score(z.label, z.pred, beta=1, zero_division=0),
                    "fpr": fp / max(fp + tn, 1), "fnr": fn / max(fn + tp, 1),
                })
                rows.append({"model": model_name, "corruption": corruption, "level": level, **m})
    out = pd.DataFrame(rows); out.to_csv(dirs["tables"] / "robustness_results.csv", index=False)
    if not out.empty:
        import matplotlib.pyplot as plt
        for corruption, frame in out.groupby("corruption"):
            fig, ax = plt.subplots(figsize=(7, 4))
            for model, m in frame.groupby("model"):
                m = m.sort_values("level"); ax.plot(m.level.astype(str), m.recall_sensitivity, marker="o", label=model)
            ax.set(xlabel="Corruption severity", ylabel="Sensitivity", title=f"Robustness: {corruption}")
            ax.legend(fontsize=7); savefig_multi(fig, dirs["figures"] / f"robustness_{corruption}"); plt.close(fig)
    return out


def _last_conv_layer(model):
    import torch.nn as nn
    layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not layers: raise RuntimeError("No convolution layer found for CAM.")
    return layers[-1]

def _tensor_to_display(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x); x = x - np.nanmin(x)
    return x / max(float(np.nanmax(x)), 1e-8)

def run_xai(manifest: pd.DataFrame, predictions: pd.DataFrame, cfg: Config, dirs: dict[str, Path]) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    import torch
    from torchvision import transforms
    proposed = predictions[predictions.model == cfg.PRIMARY_MODEL_NAME].copy()
    proposed["case"] = np.select([
        (proposed.label == 1) & (proposed.pred_calibrated == 1),
        (proposed.label == 0) & (proposed.pred_calibrated == 0),
        (proposed.label == 0) & (proposed.pred_calibrated == 1),
        (proposed.label == 1) & (proposed.pred_calibrated == 0),
    ], ["true_positive", "true_negative", "false_positive", "false_negative"], default="other")
    chosen = []
    for case in ["true_positive", "true_negative", "false_positive", "false_negative"]:
        f = proposed[proposed.case == case]
        if not f.empty: chosen.append(f.sort_values("confidence", ascending=False).iloc[0])
    errors = proposed[proposed.correct_calibrated == 0]
    if not errors.empty:
        chosen.append(errors.sort_values("confidence", ascending=False).iloc[0].copy()); chosen[-1]["case"] = "high_confidence_error"
    uncertain = proposed.sort_values("predictive_entropy", ascending=False)
    if not uncertain.empty:
        chosen.append(uncertain.iloc[0].copy()); chosen[-1]["case"] = "uncertain_prediction"
    chosen = chosen[:cfg.XAI_CASES]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    records = []
    for row in chosen:
        fold = int(row.outer_fold); ckpt = dirs["models"] / f"dfu_imageguard_fold_{fold + 1}.pt"
        model = load_checkpoint_model(lambda: build_proposed_model(cfg), ckpt, device)
        with Image.open(row.image_path) as im: original = im.convert("RGB").resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
        x = tf(original).unsqueeze(0).to(device); methods: dict[str, Optional[np.ndarray]] = {}; errors_for_case = {}
        try:
            from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
            from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
            target_layer = _last_conv_layer(model)
            for name, cls in [("gradcam", GradCAM), ("gradcam_plus_plus", GradCAMPlusPlus)]:
                with cls(model=model, target_layers=[target_layer]) as cam:
                    mask = cam(input_tensor=x, targets=[BinaryClassifierOutputTarget(1)])[0]
                methods[name] = mask
        except Exception as exc: errors_for_case["gradcam"] = repr(exc)
        try:
            with torch.inference_mode(): base_p = torch.sigmoid(model(x)).item()
            heat = np.zeros((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), dtype=np.float32); counts = np.zeros_like(heat)
            patch, stride = 48, 24
            for y0 in range(0, cfg.IMAGE_SIZE - patch + 1, stride):
                for x0 in range(0, cfg.IMAGE_SIZE - patch + 1, stride):
                    xo = x.clone(); xo[:, :, y0:y0 + patch, x0:x0 + patch] = 0
                    with torch.inference_mode(): p = torch.sigmoid(model(xo)).item()
                    heat[y0:y0 + patch, x0:x0 + patch] += base_p - p; counts[y0:y0 + patch, x0:x0 + patch] += 1
            methods["occlusion"] = heat / np.maximum(counts, 1)
        except Exception as exc: errors_for_case["occlusion"] = repr(exc)
        try:
            import shap
            import torch.nn as nn
            class TwoClassWrapper(nn.Module):
                def __init__(self, base): super().__init__(); self.base = base
                def forward(self, batch):
                    z = self.base(batch).reshape(-1, 1); return torch.cat([-z, z], dim=1)
            background_rows = manifest[(manifest.outer_fold != fold)].sample(
                n=min(8, int((manifest.outer_fold != fold).sum())), random_state=cfg.SEED + fold)
            bg = []
            for p in background_rows.image_path:
                with Image.open(p) as im: bg.append(tf(im.convert("RGB")))
            wrapped = TwoClassWrapper(model).to(device).eval(); explainer = shap.GradientExplainer(wrapped, torch.stack(bg).to(device))
            sv = explainer.shap_values(x)
            if isinstance(sv, list): arr = np.asarray(sv[1])[0]
            else:
                arr = np.asarray(sv)[0]
                if arr.ndim == 4 and arr.shape[-1] == 2: arr = arr[..., 1]
            while arr.ndim > 3: arr = np.squeeze(arr, axis=-1)
            methods["shap"] = np.mean(np.abs(arr), axis=0) if arr.ndim == 3 else np.abs(arr)
        except Exception as exc: errors_for_case["shap"] = repr(exc)
        try:
            from lime import lime_image
            explainer = lime_image.LimeImageExplainer(random_state=cfg.SEED + fold)
            def predict_np(images):
                batch = torch.stack([tf(Image.fromarray(np.uint8(a))) for a in images]).to(device)
                with torch.inference_mode(): p = torch.sigmoid(model(batch)).cpu().numpy().reshape(-1)
                return np.c_[1 - p, p]
            exp = explainer.explain_instance(np.asarray(original), predict_np, labels=(1,),
                                             num_samples=cfg.XAI_LIME_SAMPLES, hide_color=0)
            _, mask = exp.get_image_and_mask(1, positive_only=True, num_features=8, hide_rest=False)
            methods["lime"] = (mask > 0).astype(float)
        except Exception as exc: errors_for_case["lime"] = repr(exc)
        for method, heat in methods.items():
            if heat is None: continue
            heat = _tensor_to_display(heat); out_prefix = f"{row.image_id}_{row['case']}_{method}"
            original_path = dirs["xai"] / f"{out_prefix}_original.png"; original.save(original_path)
            heat_fig, heat_ax = plt.subplots(figsize=(5, 5)); heat_ax.imshow(heat); heat_ax.axis("off"); heat_ax.set_title(f"{method} heatmap")
            heat_base = dirs["xai"] / f"{out_prefix}_heatmap"; savefig_multi(heat_fig, heat_base); plt.close(heat_fig)
            fig, ax = plt.subplots(figsize=(5, 5)); ax.imshow(original); ax.imshow(heat, alpha=0.45); ax.axis("off")
            ax.set_title(f"{method} | true={row.label_name} | p={row.prob_calibrated:.3f} | fold={fold + 1}")
            out_base = dirs["xai"] / f"{out_prefix}_overlay"; savefig_multi(fig, out_base); plt.close(fig)
            records.append({
                "image_id": row.image_id, "case": row["case"], "method": method,
                "true_label": int(row.label), "prediction": int(row.pred_calibrated),
                "probability": float(row.prob_calibrated), "outer_fold": fold + 1,
                "checkpoint": str(ckpt), "original_png": str(original_path),
                "heatmap_png": str(heat_base.with_suffix('.png')), "overlay_png": str(out_base.with_suffix('.png')),
                "causality_claim": False, "clinical_correctness_claim": False,
            })
        if errors_for_case: write_json(dirs["logs"] / f"xai_errors_{row.image_id}.json", errors_for_case)
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    out = pd.DataFrame(records); out.to_csv(dirs["tables"] / "xai_metadata.csv", index=False); return out


def error_analysis(predictions: pd.DataFrame, cfg: Config, dirs: dict[str, Path]) -> pd.DataFrame:
    p = predictions[predictions.model == cfg.PRIMARY_MODEL_NAME].copy()
    p["error_type"] = np.select([
        (p.label == 1) & (p.pred_calibrated == 0), (p.label == 0) & (p.pred_calibrated == 1),
    ], ["false_negative", "false_positive"], default="correct")
    p["high_confidence_error"] = ((p.error_type != "correct") & (p.confidence >= 0.90))
    p["uncertain_prediction"] = p.predictive_entropy >= p.predictive_entropy.quantile(0.90)
    p["low_confidence_correct"] = ((p.error_type == "correct") & (p.confidence <= p.confidence.quantile(0.10)))
    p.to_csv(dirs["tables"] / "error_uncertainty_analysis.csv", index=False); return p


def uncertainty_summary(predictions: pd.DataFrame, dirs: dict[str, Path]) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score
    rows = []
    for model, f in predictions.groupby("model"):
        error = (f.correct_calibrated == 0).astype(int)
        rows.append({
            "model": model,
            "error_detection_auroc_entropy": roc_auc_score(error, f.predictive_entropy) if error.nunique() == 2 else np.nan,
            "error_detection_auroc_low_margin": roc_auc_score(error, -f.top_two_margin) if error.nunique() == 2 else np.nan,
            "mean_confidence": f.confidence.mean(), "mean_entropy": f.predictive_entropy.mean(),
        })
    out = pd.DataFrame(rows); out.to_csv(dirs["tables"] / "uncertainty_summary.csv", index=False); return out


def make_error_gallery(error_df: pd.DataFrame, dirs: dict[str, Path], max_cases: int = 12) -> None:
    import matplotlib.pyplot as plt
    candidates = pd.concat([
        error_df[error_df.error_type == "false_negative"].sort_values("confidence", ascending=False).head(max_cases // 3),
        error_df[error_df.error_type == "false_positive"].sort_values("confidence", ascending=False).head(max_cases // 3),
        error_df[error_df.uncertain_prediction].sort_values("predictive_entropy", ascending=False).head(max_cases // 3),
    ], ignore_index=True).drop_duplicates("image_id").head(max_cases)
    if candidates.empty: return
    cols = 4; rows = math.ceil(len(candidates) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows)); axes = np.atleast_1d(axes).reshape(-1)
    for ax in axes: ax.axis("off")
    for ax, (_, row) in zip(axes, candidates.iterrows()):
        with Image.open(row.image_path) as im: ax.imshow(im.convert("RGB"))
        ax.set_title(f"{row.error_type} | y={row.label} p={row.prob_calibrated:.2f}", fontsize=8); ax.axis("off")
    savefig_multi(fig, dirs["figures"] / "error_case_gallery"); plt.close(fig)
