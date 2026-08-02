import torch

from src.reliability_models import MODEL_SPECS, create_classifier


def test_all_locked_models_produce_one_binary_logit():
    sample = torch.zeros(1, 3, 224, 224)
    for model_key in MODEL_SPECS:
        model = create_classifier(model_key, pretrained=False, dropout=0.2)
        model.eval()
        with torch.inference_mode():
            output = model(sample)
        assert output.shape == (1,)
        assert torch.isfinite(output).all()
