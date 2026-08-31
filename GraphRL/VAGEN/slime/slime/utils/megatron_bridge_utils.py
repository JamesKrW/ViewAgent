import inspect
import types
from contextlib import contextmanager


# Transformers 5 moved the language-model fields of multimodal configs (for
# example Qwen2.5-VL) under ``text_config``.  Older Megatron-Bridge releases
# still read those fields from the top-level config.  Keep the compatibility
# shim here, next to the other Bridge normalisation, so both model creation and
# checkpoint conversion see the same config.
_TEXT_CONFIG_FIELDS = (
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "max_position_embeddings",
    "rms_norm_eps",
    "initializer_range",
    "attention_dropout",
    "hidden_dropout",
    "attention_bias",
    "mlp_bias",
    "use_qk_norm",
    "rope_theta",
    "rope_scaling",
    "rope_parameters",
    "partial_rotary_factor",
    "bos_token_id",
    "eos_token_id",
)


def patch_qwen3_vl_rotary_embedding() -> None:
    """Make Megatron-Bridge Qwen3-VL rotary modules accept packed sequences."""
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl import text_model

    for name in (
        "Qwen3VLTextRotaryEmbedding",
        "Qwen3VLMoETextRotaryEmbedding",
        "Qwen3VLMultimodalRotaryEmbedding",
    ):
        cls = getattr(text_model, name, None)
        if cls is None or "packed_seq_params" in inspect.signature(cls.forward).parameters:
            continue
        if getattr(cls, "_slime_accepts_packed_seq_params", False):
            continue
        original_forward = cls.forward

        def patched_forward(self, *args, _original_forward=original_forward, packed_seq_params=None, **kwargs):
            return _original_forward(self, *args, **kwargs)

        cls.forward = patched_forward
        cls._slime_accepts_packed_seq_params = True


def _install_qwen25_vl_transformers5_instance_compat(model, hf_model_cls) -> None:
    """Restore the Transformers 4.x method contract expected by Bridge 0.3.

    Transformers 5 changed Qwen2.5-VL feature helpers to return a
    ``BaseModelOutputWithPooling`` and added ``mm_token_type_ids`` to
    ``get_rope_index``.  Megatron-Bridge 0.3 binds those methods dynamically but
    its forward still expects the old tuple-of-image-chunks/signature contract.
    This is the same adaptation shipped by Bridge 0.4, applied per model so it
    does not alter Hugging Face or SGLang globally.
    """
    import torch

    if not hasattr(model.config, "return_dict"):
        model.config.return_dict = True

    original_get_image_features = model.get_image_features
    original_get_video_features = model.get_video_features
    original_get_rope_index = model.get_rope_index

    def get_image_features_legacy(self, pixel_values, image_grid_thw=None, **kwargs):
        kwargs["return_dict"] = True
        output = original_get_image_features(pixel_values, image_grid_thw, **kwargs)
        return getattr(output, "pooler_output", output)

    def get_video_features_legacy(self, pixel_values_videos, video_grid_thw=None, **kwargs):
        kwargs["return_dict"] = True
        output = original_get_video_features(pixel_values_videos, video_grid_thw, **kwargs)
        return getattr(output, "pooler_output", output)

    def get_rope_index_legacy(
        self,
        input_ids,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        attention_mask=None,
        **kwargs,
    ):
        mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int)
        mm_token_type_ids[input_ids == self.config.image_token_id] = 1
        mm_token_type_ids[input_ids == self.config.video_token_id] = 2
        return original_get_rope_index(
            input_ids,
            mm_token_type_ids,
            image_grid_thw,
            video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            attention_mask=attention_mask,
            **kwargs,
        )

    model.get_image_features = types.MethodType(get_image_features_legacy, model)
    model.get_video_features = types.MethodType(get_video_features_legacy, model)
    model.get_rope_index = types.MethodType(get_rope_index_legacy, model)
    if hasattr(hf_model_cls, "get_vision_position_ids"):
        model.get_vision_position_ids = types.MethodType(hf_model_cls.get_vision_position_ids, model)


def patch_qwen25_vl_transformers5() -> None:
    """Backport Bridge 0.4's Qwen2.5-VL support while retaining Bridge 0.3."""
    from megatron.bridge.models.qwen_vl.modeling_qwen25_vl import Qwen25VLModel
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel

    if "mm_token_type_ids" not in inspect.signature(Qwen2_5_VLModel.get_rope_index).parameters:
        return
    if getattr(Qwen25VLModel, "_slime_transformers5_compat", False):
        return

    original_init = Qwen25VLModel.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _install_qwen25_vl_transformers5_instance_compat(self, Qwen2_5_VLModel)

    Qwen25VLModel.__init__ = patched_init
    Qwen25VLModel._slime_transformers5_compat = True


def patch_hf_config_for_megatron_bridge(hf_config):
    """Normalize nested Hugging Face configs before Megatron-Bridge reads them."""
    configs = []
    seen_config_ids = set()

    def add_config(config):
        if config is None or id(config) in seen_config_ids:
            return
        seen_config_ids.add(id(config))
        configs.append(config)

    add_config(hf_config)
    add_config(getattr(hf_config, "config", None))

    for config in list(configs):
        add_config(getattr(config, "text_config", None))

    for config in configs:
        rope_params = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None)
        if isinstance(rope_params, dict) and "rope_theta" in rope_params and not hasattr(config, "rope_theta"):
            config.rope_theta = rope_params["rope_theta"]

    for config in configs:
        text_config = getattr(config, "text_config", None)
        if text_config is None:
            continue
        for field in _TEXT_CONFIG_FIELDS:
            if getattr(config, field, None) is not None:
                continue
            value = getattr(text_config, field, None)
            if value is not None:
                setattr(config, field, value)

    return hf_config


def patch_auto_bridge_hf_config(bridge):
    # Megatron 0.16 introduced a fused output projection/cross-entropy module.
    # Bridge 0.4+ knows that it is column-parallel; older Bridge versions need
    # the registration explicitly.  LinearForLastLayer is slime's scalar
    # critic head and is replicated.
    from megatron.bridge.models.conversion.param_mapping import AutoMapping

    AutoMapping.register_module_type("LinearCrossEntropyModule", "column")
    AutoMapping.register_module_type("LinearForLastLayer", "replicated")
    patch_qwen25_vl_transformers5()

    hf_pretrained = getattr(bridge, "hf_pretrained", None)
    if hf_pretrained is not None:
        patch_hf_config_for_megatron_bridge(hf_pretrained)
    return bridge


@contextmanager
def patch_megatron_model(model):
    """Supply the config attribute expected by Megatron-Bridge while loading."""
    try:
        from megatron.core.pipeline_parallel.utils import unwrap_model
    except ImportError:
        from megatron.core.utils import unwrap_model

    unwrapped_model = unwrap_model(model)[0]
    model_config = unwrapped_model.config
    attribute_was_added = False
    if not hasattr(model_config, "share_embeddings_and_output_weights"):
        model_config.share_embeddings_and_output_weights = unwrapped_model.share_embeddings_and_output_weights
        attribute_was_added = True

    try:
        yield
    finally:
        if attribute_was_added:
            delattr(model_config, "share_embeddings_and_output_weights")
