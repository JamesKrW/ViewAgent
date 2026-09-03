from __future__ import annotations

from argparse import Namespace

from slime.backends.megatron_utils import data as data_module


def test_structured_prompt_is_transport_metadata_not_a_numeric_metric(monkeypatch):
    captured = {}
    monkeypatch.setattr(data_module.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(data_module.mpu, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        data_module.mpu,
        "get_data_parallel_world_size",
        lambda with_context_parallel=False: 1,
    )

    def fake_gather(_prefix, _args, _rollout_id, metrics):
        captured.update(metrics)
        return metrics

    monkeypatch.setattr(data_module, "gather_log_data", fake_gather)
    data_module.log_rollout_data(
        0,
        Namespace(
            ci_test=False,
            log_multi_turn=False,
            log_passrate=False,
            log_correct_samples=False,
        ),
        {
            "response_lengths": [1],
            "loss_masks": [[1]],
            "total_lengths": [2],
            "rollout_mask_sums": [1],
            "global_batch_sizes": [1],
            "rewards": [0.5],
            "prompt": [[{"role": "user", "content": "structured"}]],
        },
    )

    assert "prompt" not in captured
    assert captured["rewards"] == (0.5, 1)


def test_token_level_reward_vectors_log_mean_sample_total(monkeypatch):
    captured = {}
    monkeypatch.setattr(data_module.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(data_module.mpu, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        data_module.mpu,
        "get_data_parallel_world_size",
        lambda with_context_parallel=False: 1,
    )

    def fake_gather(_prefix, _args, _rollout_id, metrics):
        captured.update(metrics)
        return metrics

    monkeypatch.setattr(data_module, "gather_log_data", fake_gather)
    data_module.log_rollout_data(
        0,
        Namespace(
            ci_test=False,
            log_multi_turn=False,
            log_passrate=False,
            log_correct_samples=False,
        ),
        {
            "response_lengths": [2, 3],
            "loss_masks": [[1, 1], [1, 1, 1]],
            "total_lengths": [3, 4],
            "rollout_mask_sums": [2, 3],
            "global_batch_sizes": [2],
            "rewards": [[0.25, 0.75], [0.0, -0.5, 2.0]],
        },
    )

    # gather_log_data divides the summed sample totals by this sample count.
    assert captured["rewards"] == (2.5, 2)
