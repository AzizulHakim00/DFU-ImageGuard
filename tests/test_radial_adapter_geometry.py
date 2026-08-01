from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radial_adapter_model import RadialAdapterSpec, RadialMorphologyAdapter


def test_adapter_shapes_and_zero_gate():
    spec = RadialAdapterSpec(projection_dim=32, radial_rings=5, angular_sectors=16)
    adapter = RadialMorphologyAdapter.build(in_channels=48, spec=spec)
    feature = torch.randn(3, 48, 14, 14)
    output = adapter(feature)
    assert output["adapter_logit"].shape == (3,)
    assert output["adapter_contribution"].shape == (3,)
    assert output["attention"].shape == (3, 1, 14, 14)
    assert output["center"].shape == (3, 2)
    assert output["scale"].shape == (3,)
    assert output["polar_tokens"].shape == (3, 32, 5, 16)
    assert torch.allclose(output["adapter_contribution"], torch.zeros(3), atol=0, rtol=0)
    attention_sum = output["attention"].flatten(1).sum(1)
    assert torch.allclose(attention_sum, torch.ones_like(attention_sum), atol=1e-5)
    assert torch.isfinite(output["polar_tokens"]).all()


def test_gate_can_activate_and_backpropagate():
    spec = RadialAdapterSpec(projection_dim=16, radial_rings=4, angular_sectors=8)
    adapter = RadialMorphologyAdapter.build(in_channels=24, spec=spec)
    feature = torch.randn(2, 24, 10, 10, requires_grad=True)
    output = adapter(feature)
    loss = output["adapter_contribution"].sum()
    loss.backward()
    assert adapter.alpha.grad is not None
    assert torch.isfinite(adapter.alpha.grad)


def test_full_model_preserves_baseline_at_initialization(monkeypatch):
    import types
    import torch.nn as nn

    class FeatureInfo:
        def channels(self):
            return [24, 48]

    class FakeBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.stage3 = nn.Conv2d(3, 24, 3, stride=4, padding=1)
            self.stage4 = nn.Conv2d(24, 48, 3, stride=2, padding=1)
            self.feature_info = FeatureInfo()

        def forward(self, x):
            f3 = self.stage3(x)
            f4 = self.stage4(f3)
            return [f3, f4]

    fake_timm = types.SimpleNamespace(create_model=lambda *args, **kwargs: FakeBackbone())
    monkeypatch.setitem(sys.modules, "timm", fake_timm)

    from radial_adapter_model import ConvNeXtV2BinaryBaseline, DFURadialAdapterNet

    spec = RadialAdapterSpec(projection_dim=16, radial_rings=4, angular_sectors=8, dropout=0.0)
    torch.manual_seed(123)
    baseline = ConvNeXtV2BinaryBaseline.build(pretrained=False, spec=spec).eval()
    torch.manual_seed(123)
    adapter = DFURadialAdapterNet.build(pretrained=False, spec=spec).eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        baseline_logits = baseline(x)
        adapter_output = adapter(x, return_aux=True)
    assert torch.allclose(adapter_output["logits"], adapter_output["base_logits"], atol=0, rtol=0)
    assert torch.allclose(baseline_logits, adapter_output["base_logits"], atol=1e-6, rtol=1e-6)
