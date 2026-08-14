from src.reliability_io import ReliabilitySettings
from src.reliability_models import MODEL_SPECS


def test_final_protocol_is_locked():
    settings = ReliabilitySettings()
    settings.validate()
    assert settings.n_folds == 5
    assert settings.seeds == (2026, 2027, 2028)
    assert len(settings.model_keys) * len(settings.seeds) * settings.n_folds == 45
    assert settings.primary_model_key == "convnextv2_tiny"
    assert MODEL_SPECS[settings.primary_model_key].primary is True


def test_architecture_search_is_not_encoded_as_model_sweep():
    settings = ReliabilitySettings()
    assert set(settings.model_keys) == {
        "convnextv2_tiny",
        "mobilenetv3_large",
        "densenet121",
    }
