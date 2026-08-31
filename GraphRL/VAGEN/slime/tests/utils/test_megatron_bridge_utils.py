import types

import pytest
import torch

from slime.utils.megatron_bridge_utils import (
    _install_qwen25_vl_transformers5_instance_compat,
    patch_auto_bridge_hf_config,
    patch_hf_config_for_megatron_bridge,
)


@pytest.mark.unit
def test_patch_hf_config_adds_rope_theta_from_rope_parameters():
    hf_config = types.SimpleNamespace(rope_parameters={"rope_theta": 1000000})

    patched_config = patch_hf_config_for_megatron_bridge(hf_config)

    assert patched_config is hf_config
    assert hf_config.rope_theta == 1000000


@pytest.mark.unit
def test_patch_hf_config_does_not_override_existing_rope_theta():
    hf_config = types.SimpleNamespace(rope_theta=500000, rope_parameters={"rope_theta": 1000000})

    patch_hf_config_for_megatron_bridge(hf_config)

    assert hf_config.rope_theta == 500000


@pytest.mark.unit
def test_patch_hf_config_handles_nested_text_config():
    text_config = types.SimpleNamespace(
        rope_parameters={"rope_theta": 10000},
        num_hidden_layers=28,
        hidden_size=3584,
        intermediate_size=18944,
        num_attention_heads=28,
        num_key_value_heads=4,
        vocab_size=152064,
        bos_token_id=151643,
        eos_token_id=151645,
    )
    hf_config = types.SimpleNamespace(text_config=text_config)

    patch_hf_config_for_megatron_bridge(hf_config)

    assert text_config.rope_theta == 10000
    assert hf_config.rope_theta == 10000
    assert hf_config.num_hidden_layers == 28
    assert hf_config.hidden_size == 3584
    assert hf_config.intermediate_size == 18944
    assert hf_config.num_attention_heads == 28
    assert hf_config.num_key_value_heads == 4
    assert hf_config.vocab_size == 152064
    assert hf_config.bos_token_id == 151643
    assert hf_config.eos_token_id == 151645


@pytest.mark.unit
def test_patch_hf_config_preserves_top_level_multimodal_values():
    text_config = types.SimpleNamespace(tie_word_embeddings=True, num_attention_heads=28)
    hf_config = types.SimpleNamespace(
        text_config=text_config,
        tie_word_embeddings=False,
        num_attention_heads=32,
    )

    patch_hf_config_for_megatron_bridge(hf_config)

    assert hf_config.tie_word_embeddings is False
    assert hf_config.num_attention_heads == 32


@pytest.mark.unit
def test_patch_hf_config_handles_pretrained_wrapper_config():
    wrapped_config = types.SimpleNamespace(rope_parameters={"rope_theta": 10000})
    hf_pretrained = types.SimpleNamespace(config=wrapped_config)

    patch_hf_config_for_megatron_bridge(hf_pretrained)

    assert wrapped_config.rope_theta == 10000


@pytest.mark.unit
def test_patch_hf_config_uses_rope_scaling_fallback():
    hf_config = types.SimpleNamespace(rope_scaling={"rope_theta": 10000})

    patch_hf_config_for_megatron_bridge(hf_config)

    assert hf_config.rope_theta == 10000


@pytest.mark.unit
def test_patch_auto_bridge_hf_config_patches_hf_pretrained():
    hf_config = types.SimpleNamespace(rope_parameters={"rope_theta": 12345})
    bridge = types.SimpleNamespace(hf_pretrained=hf_config)

    patched_bridge = patch_auto_bridge_hf_config(bridge)

    assert patched_bridge is bridge
    assert bridge.hf_pretrained.rope_theta == 12345


@pytest.mark.unit
def test_patch_auto_bridge_registers_new_megatron_output_layers():
    from megatron.bridge.models.conversion.param_mapping import AutoMapping

    bridge = types.SimpleNamespace(hf_pretrained=types.SimpleNamespace())

    patch_auto_bridge_hf_config(bridge)

    assert "LinearCrossEntropyModule" in AutoMapping._MODULE_TYPE_REGISTRY["column"]
    assert "LinearForLastLayer" in AutoMapping._MODULE_TYPE_REGISTRY["replicated"]


@pytest.mark.unit
def test_qwen25_vl_transformers5_instance_compat_restores_bridge03_contract():
    calls = {}

    class DummyModel:
        def __init__(self):
            self.config = types.SimpleNamespace(image_token_id=11, video_token_id=12)

        def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
            calls["image_return_dict"] = kwargs["return_dict"]
            return types.SimpleNamespace(pooler_output=(pixel_values,))

        def get_video_features(self, pixel_values, video_grid_thw=None, **kwargs):
            calls["video_return_dict"] = kwargs["return_dict"]
            return types.SimpleNamespace(pooler_output=(pixel_values,))

        def get_rope_index(self, input_ids, mm_token_type_ids, *args, **kwargs):
            calls["mm_token_type_ids"] = mm_token_type_ids
            return "positions", "deltas"

    class DummyHFModel:
        def get_vision_position_ids(self):
            return "vision-positions"

    model = DummyModel()
    _install_qwen25_vl_transformers5_instance_compat(model, DummyHFModel)

    image = torch.ones(1)
    video = torch.ones(2)
    assert model.config.return_dict is True
    image_features = model.get_image_features(image)
    video_features = model.get_video_features(video)
    assert image_features[0] is image
    assert video_features[0] is video
    assert calls["image_return_dict"] is True
    assert calls["video_return_dict"] is True

    input_ids = torch.tensor([[0, 11, 12]])
    assert model.get_rope_index(input_ids) == ("positions", "deltas")
    assert torch.equal(calls["mm_token_type_ids"], torch.tensor([[0, 1, 2]], dtype=torch.int))
    assert model.get_vision_position_ids() == "vision-positions"
