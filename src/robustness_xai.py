from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

from .config_data import Config, hash_text, seed_worker, write_json
from .evaluation import metric_dict, sigmoid_np
from .models_training import (
    build_baseline_model,
    build_proposed_model,
    load_checkpoint_model,
    predict_logits,
)
from .statistics_figures import savefig_multi


def corruption_fn(name: str, level: float, image_id: str) -> Callable[[Image.Image], Image.Image]:
    rng = np.random.default_rng(hash_text(f"{image_id}:{name}:{level}"))

    def apply(image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        if name == "gaussian_noise":
            array = np.asarray(image).astype(np.float32) / 255.0
            array = np.clip(array + rng.normal(0, float(level), array.shape), 0, 1)
            return Image.fromarray((array * 255).astype(np.uint8))
        if name == "gaussian_blur":
            return image.filter(ImageFilter.GaussianBlur(radius=float(level)))
        if name == "brightness":
            return ImageEnhance.Brightness(image).enhance(float(level))
        if name == "contrast":
            return ImageEnhance.Contrast(image).enhance(float(level))
        if name == "jpeg":
            import io

            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=int(level))
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                return decoded.convert("RGB").copy()
        if name == "rotation":
            return image.rotate(
                float(level),
                resample=Image.Resampling.BILINEAR,
                fillcolor=(0, 0, 0),
            )
        if name == "occlusion":
            array = np.asarray(image).copy()
            height, width = array.shape[:2]
            side = max(1, int(math.sqrt(float(level)) * min(height, width)))
            y = int(rng.integers(0, max(1, height - side + 1)))
            x = int(rng.integers(0, max(1, width - side + 1)))
            array[y:y + side, x:x + side] = np.median(array.reshape(-1, 3), axis=0)
            return Image.fromarray(array.astype(np.uint8))
        raise ValueError(f"Unknown corruption: {name}")

    return apply


def run_robustness(
    manifest: pd.DataFrame,
    predictions: pd.DataFrame,
    cfg: Config,
    dirs: dict[str, Path],
) -> pd.DataFrame:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_transform = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class CorruptedDataset(Dataset):
        def __init__(self, frame: pd.DataFrame, corruption: str, level: float):
            self.frame = frame.reset_index(drop=True)
            self.corruption = corruption
            self.level = level

        def __len__(self) -> int:
            return len(self.frame)

        def __getitem__(self, index: int):
            row = self.frame.iloc[index]
            with Image.open(row.image_path) as image:
                corrupted = corruption_fn(
                    self.corruption,
                    self.level,
                    str(row.image_id),
                )(image)
                x = base_transform(corrupted)
            return x, torch.tensor(float(row.label)), index

    model_factories: dict[str, Callable[..., Any]] = {
        cfg.PRIMARY_MODEL_NAME: lambda pretrained=True: build_proposed_model(cfg, pretrained=pretrained),
    }
    for name, timm_name in cfg.BASELINE_MODELS.items():
        model_factories[name] = (
            lambda pretrained=True, model_name=timm_name: build_baseline_model(
                model_name,
                pretrained=pretrained,
            )
        )

    rows: list[dict[str, Any]] = []
    for model_name, factory in model_factories.items():
        model_predictions = predictions[predictions.model == model_name]
        if model_predictions.empty:
            continue
        safe_name = model_name.lower().replace("-", "_").replace(" ", "_")
        for corruption, levels in cfg.ROBUSTNESS_LEVELS.items():
            for level in levels:
                pieces: list[pd.DataFrame] = []
                for fold in range(cfg.N_FOLDS):
                    fold_frame = manifest[manifest.outer_fold == fold].copy().reset_index(drop=True)
                    checkpoint = dirs["models"] / (
                        f"dfu_imageguard_fold_{fold + 1}.pt"
                        if model_name == cfg.PRIMARY_MODEL_NAME
                        else f"{safe_name}_fold_{fold + 1}.pt"
                    )
                    if not checkpoint.exists():
                        continue
                    model = load_checkpoint_model(factory, checkpoint, device)
                    generator = torch.Generator().manual_seed(
                        cfg.SEED + fold + hash_text(f"{model_name}:{corruption}:{level}")
                    )
                    workers = max(0, int(cfg.NUM_WORKERS))
                    loader_kwargs: dict[str, Any] = {
                        "batch_size": cfg.BATCH_SIZE,
                        "shuffle": False,
                        "num_workers": workers,
                        "pin_memory": device.type == "cuda",
                        "worker_init_fn": seed_worker,
                        "generator": generator,
                    }
                    if workers > 0:
                        loader_kwargs["persistent_workers"] = True
                    loader = DataLoader(
                        CorruptedDataset(fold_frame, corruption, level),
                        **loader_kwargs,
                    )
                    logits, y, _ = predict_logits(model, loader, device)
                    fold_rows = model_predictions[model_predictions.outer_fold == fold]
                    if fold_rows.empty:
                        raise RuntimeError(
                            f"Missing OOF metadata for {model_name}, fold {fold + 1}"
                        )
                    temperature = float(fold_rows.temperature.iloc[0])
                    threshold = float(fold_rows.threshold.iloc[0])
                    probability = sigmoid_np(logits / temperature)
                    pieces.append(pd.DataFrame({
                        "label": y,
                        "prob": probability,
                        "pred": (probability >= threshold).astype(int),
                    }))
                    del model, loader
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                if not pieces:
                    continue
                combined = pd.concat(pieces, ignore_index=True)
                metrics = metric_dict(combined.label, combined.prob, 0.5)
                from sklearn.metrics import balanced_accuracy_score, confusion_matrix, fbeta_score

                tn, fp, fn, tp = confusion_matrix(
                    combined.label,
                    combined.pred,
                    labels=[0, 1],
                ).ravel()
                metrics.update({
                    "accuracy": float((combined.pred == combined.label).mean()),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(combined.label, combined.pred)
                    ),
                    "recall_sensitivity": float(tp / max(tp + fn, 1)),
                    "specificity": float(tn / max(tn + fp, 1)),
                    "f1": float(
                        fbeta_score(combined.label, combined.pred, beta=1, zero_division=0)
                    ),
                    "fpr": float(fp / max(fp + tn, 1)),
                    "fnr": float(fn / max(fn + tp, 1)),
                })
                rows.append({
                    "model": model_name,
                    "corruption": corruption,
                    "level": level,
                    **metrics,
                })

    output = pd.DataFrame(rows)
    output.to_csv(dirs["tables"] / "robustness_results.csv", index=False)
    if not output.empty:
        import matplotlib.pyplot as plt

        for corruption, frame in output.groupby("corruption"):
            fig, ax = plt.subplots(figsize=(7, 4))
            for model_name, model_frame in frame.groupby("model"):
                model_frame = model_frame.sort_values("level")
                ax.plot(
                    model_frame.level.astype(str),
                    model_frame.recall_sensitivity,
                    marker="o",
                    label=model_name,
                )
            ax.set(
                xlabel="Corruption severity",
                ylabel="Sensitivity",
                title=f"Robustness: {corruption}",
            )
            ax.legend(fontsize=7)
            savefig_multi(fig, dirs["figures"] / f"robustness_{corruption}")
            plt.close(fig)
    return output


def _cam_target_layer(model):
    import torch.nn as nn

    target_module = getattr(model, "backbone", model)
    layers = [module for module in target_module.modules() if isinstance(module, nn.Conv2d)]
    if not layers:
        raise RuntimeError("No convolution layer found for CAM")
    return layers[-1]


def _tensor_to_display(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.squeeze(values)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D explanation map; got shape {values.shape}")
    if not np.isfinite(values).all():
        raise FloatingPointError("Explanation map contains NaN or infinity")
    values = values - values.min()
    return values / max(float(values.max()), 1e-8)


def run_xai(
    manifest: pd.DataFrame,
    predictions: pd.DataFrame,
    cfg: Config,
    dirs: dict[str, Path],
) -> pd.DataFrame:
    import matplotlib.pyplot as plt
    import torch
    from torchvision import transforms

    proposed = predictions[predictions.model == cfg.PRIMARY_MODEL_NAME].copy()
    if proposed.empty:
        return pd.DataFrame()
    proposed["case"] = np.select([
        (proposed.label == 1) & (proposed.pred_calibrated == 1),
        (proposed.label == 0) & (proposed.pred_calibrated == 0),
        (proposed.label == 0) & (proposed.pred_calibrated == 1),
        (proposed.label == 1) & (proposed.pred_calibrated == 0),
    ], ["true_positive", "true_negative", "false_positive", "false_negative"], default="other")

    chosen: list[pd.Series] = []
    for case in ["true_positive", "true_negative", "false_positive", "false_negative"]:
        frame = proposed[proposed.case == case]
        if not frame.empty:
            chosen.append(frame.sort_values("confidence", ascending=False).iloc[0].copy())
    errors = proposed[proposed.correct_calibrated == 0]
    if not errors.empty:
        row = errors.sort_values("confidence", ascending=False).iloc[0].copy()
        row["case"] = "high_confidence_error"
        chosen.append(row)
    uncertain = proposed.sort_values("predictive_entropy", ascending=False)
    if not uncertain.empty:
        row = uncertain.iloc[0].copy()
        row["case"] = "uncertain_prediction"
        chosen.append(row)

    unique_chosen: list[pd.Series] = []
    seen: set[str] = set()
    for row in chosen:
        if str(row.image_id) not in seen:
            seen.add(str(row.image_id))
            unique_chosen.append(row)
    chosen = unique_chosen[:cfg.XAI_CASES]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    records: list[dict[str, Any]] = []

    for row in chosen:
        fold = int(row.outer_fold)
        checkpoint = dirs["models"] / f"dfu_imageguard_fold_{fold + 1}.pt"
        model = load_checkpoint_model(
            lambda pretrained=True: build_proposed_model(cfg, pretrained=pretrained),
            checkpoint,
            device,
        )
        with Image.open(row.image_path) as image:
            original = image.convert("RGB").resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
        x = transform(original).unsqueeze(0).to(device)
        methods: dict[str, Optional[np.ndarray]] = {}
        method_errors: dict[str, str] = {}

        try:
            from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
            from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

            target_layer = _cam_target_layer(model)
            for name, cam_class in [("gradcam", GradCAM), ("gradcam_plus_plus", GradCAMPlusPlus)]:
                with cam_class(model=model, target_layers=[target_layer]) as cam:
                    mask = cam(
                        input_tensor=x,
                        targets=[BinaryClassifierOutputTarget(1)],
                    )[0]
                methods[name] = mask
        except Exception as exc:
            method_errors["gradcam"] = repr(exc)

        try:
            with torch.inference_mode():
                base_probability = torch.sigmoid(model(x)).item()
            heat = np.zeros((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), dtype=np.float32)
            counts = np.zeros_like(heat)
            patch, stride = 48, 24
            for y0 in range(0, cfg.IMAGE_SIZE - patch + 1, stride):
                for x0 in range(0, cfg.IMAGE_SIZE - patch + 1, stride):
                    occluded = x.clone()
                    occluded[:, :, y0:y0 + patch, x0:x0 + patch] = 0
                    with torch.inference_mode():
                        probability = torch.sigmoid(model(occluded)).item()
                    heat[y0:y0 + patch, x0:x0 + patch] += base_probability - probability
                    counts[y0:y0 + patch, x0:x0 + patch] += 1
            methods["occlusion"] = heat / np.maximum(counts, 1)
        except Exception as exc:
            method_errors["occlusion"] = repr(exc)

        try:
            import shap
            import torch.nn as nn

            class TwoClassWrapper(nn.Module):
                def __init__(self, base):
                    super().__init__()
                    self.base = base

                def forward(self, batch):
                    logits = self.base(batch).reshape(-1, 1)
                    return torch.cat([-logits, logits], dim=1)

            available = manifest[manifest.outer_fold != fold]
            background_rows = available.sample(
                n=min(8, len(available)),
                random_state=cfg.SEED + fold,
            )
            background = []
            for path in background_rows.image_path:
                with Image.open(path) as image:
                    background.append(transform(image.convert("RGB")))
            if not background:
                raise RuntimeError("No training-fold background images available for SHAP")
            wrapped = TwoClassWrapper(model).to(device).eval()
            explainer = shap.GradientExplainer(wrapped, torch.stack(background).to(device))
            shap_values = explainer.shap_values(x)
            if isinstance(shap_values, list):
                array = np.asarray(shap_values[1])[0]
            else:
                array = np.asarray(shap_values)[0]
                if array.ndim == 4 and array.shape[-1] == 2:
                    array = array[..., 1]
            array = np.squeeze(array)
            if array.ndim == 3:
                if array.shape[0] in {1, 3}:
                    array = np.mean(np.abs(array), axis=0)
                elif array.shape[-1] in {1, 3}:
                    array = np.mean(np.abs(array), axis=-1)
            methods["shap"] = np.abs(array)
        except Exception as exc:
            method_errors["shap"] = repr(exc)

        try:
            from lime import lime_image

            explainer = lime_image.LimeImageExplainer(random_state=cfg.SEED + fold)

            def predict_numpy(images):
                batch = torch.stack([
                    transform(Image.fromarray(np.uint8(array)))
                    for array in images
                ]).to(device)
                with torch.inference_mode():
                    probability = torch.sigmoid(model(batch)).cpu().numpy().reshape(-1)
                return np.c_[1 - probability, probability]

            explanation = explainer.explain_instance(
                np.asarray(original),
                predict_numpy,
                labels=(1,),
                num_samples=cfg.XAI_LIME_SAMPLES,
                hide_color=0,
            )
            _, mask = explanation.get_image_and_mask(
                1,
                positive_only=True,
                num_features=8,
                hide_rest=False,
            )
            methods["lime"] = (mask > 0).astype(float)
        except Exception as exc:
            method_errors["lime"] = repr(exc)

        for method, heat in methods.items():
            if heat is None:
                continue
            try:
                heat = _tensor_to_display(heat)
            except Exception as exc:
                method_errors[f"{method}_postprocess"] = repr(exc)
                continue
            prefix = f"{row.image_id}_{row['case']}_{method}"
            original_path = dirs["xai"] / f"{prefix}_original.png"
            original.save(original_path)

            heat_figure, heat_axis = plt.subplots(figsize=(5, 5))
            heat_axis.imshow(heat)
            heat_axis.axis("off")
            heat_axis.set_title(f"{method} heatmap")
            heat_base = dirs["xai"] / f"{prefix}_heatmap"
            savefig_multi(heat_figure, heat_base)
            plt.close(heat_figure)

            figure, axis = plt.subplots(figsize=(5, 5))
            axis.imshow(original)
            axis.imshow(heat, alpha=0.45)
            axis.axis("off")
            axis.set_title(
                f"{method} | true={row.label_name} | p={row.prob_calibrated:.3f} | fold={fold + 1}"
            )
            overlay_base = dirs["xai"] / f"{prefix}_overlay"
            savefig_multi(figure, overlay_base)
            plt.close(figure)
            records.append({
                "image_id": row.image_id,
                "case": row["case"],
                "method": method,
                "true_label": int(row.label),
                "prediction": int(row.pred_calibrated),
                "probability": float(row.prob_calibrated),
                "outer_fold": fold + 1,
                "checkpoint": str(checkpoint),
                "original_png": str(original_path),
                "heatmap_png": str(heat_base.with_suffix(".png")),
                "overlay_png": str(overlay_base.with_suffix(".png")),
                "causality_claim": False,
                "clinical_correctness_claim": False,
            })
        if method_errors:
            write_json(dirs["logs"] / f"xai_errors_{row.image_id}.json", method_errors)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output = pd.DataFrame(records)
    output.to_csv(dirs["tables"] / "xai_metadata.csv", index=False)
    return output


def error_analysis(
    predictions: pd.DataFrame,
    cfg: Config,
    dirs: dict[str, Path],
) -> pd.DataFrame:
    output = predictions[predictions.model == cfg.PRIMARY_MODEL_NAME].copy()
    output["error_type"] = np.select([
        (output.label == 1) & (output.pred_calibrated == 0),
        (output.label == 0) & (output.pred_calibrated == 1),
    ], ["false_negative", "false_positive"], default="correct")
    output["high_confidence_error"] = (
        (output.error_type != "correct") & (output.confidence >= 0.90)
    )
    output["uncertain_prediction"] = (
        output.predictive_entropy >= output.predictive_entropy.quantile(0.90)
    )
    output["low_confidence_correct"] = (
        (output.error_type == "correct")
        & (output.confidence <= output.confidence.quantile(0.10))
    )
    output.to_csv(dirs["tables"] / "error_uncertainty_analysis.csv", index=False)
    return output


def uncertainty_summary(
    predictions: pd.DataFrame,
    dirs: dict[str, Path],
) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score

    rows: list[dict[str, Any]] = []
    for model_name, frame in predictions.groupby("model"):
        error = (frame.correct_calibrated == 0).astype(int)
        rows.append({
            "model": model_name,
            "error_detection_auroc_entropy": (
                float(roc_auc_score(error, frame.predictive_entropy))
                if error.nunique() == 2
                else np.nan
            ),
            "error_detection_auroc_low_margin": (
                float(roc_auc_score(error, -frame.top_two_margin))
                if error.nunique() == 2
                else np.nan
            ),
            "mean_confidence": float(frame.confidence.mean()),
            "mean_entropy": float(frame.predictive_entropy.mean()),
        })
    output = pd.DataFrame(rows)
    output.to_csv(dirs["tables"] / "uncertainty_summary.csv", index=False)
    return output


def make_error_gallery(
    error_df: pd.DataFrame,
    dirs: dict[str, Path],
    max_cases: int = 12,
) -> None:
    import matplotlib.pyplot as plt

    candidates = pd.concat([
        error_df[error_df.error_type == "false_negative"]
        .sort_values("confidence", ascending=False)
        .head(max_cases // 3),
        error_df[error_df.error_type == "false_positive"]
        .sort_values("confidence", ascending=False)
        .head(max_cases // 3),
        error_df[error_df.uncertain_prediction]
        .sort_values("predictive_entropy", ascending=False)
        .head(max_cases // 3),
    ], ignore_index=True).drop_duplicates("image_id").head(max_cases)
    if candidates.empty:
        return
    columns = 4
    rows = math.ceil(len(candidates) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows))
    axes = np.atleast_1d(axes).reshape(-1)
    for axis in axes:
        axis.axis("off")
    for axis, (_, row) in zip(axes, candidates.iterrows()):
        with Image.open(row.image_path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(
            f"{row.error_type} | y={row.label} p={row.prob_calibrated:.2f}",
            fontsize=8,
        )
        axis.axis("off")
    savefig_multi(figure, dirs["figures"] / "error_case_gallery")
    plt.close(figure)
