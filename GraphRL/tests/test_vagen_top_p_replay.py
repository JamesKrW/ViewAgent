from __future__ import annotations

import asyncio
from types import SimpleNamespace

import torch

from vagen_agent.rollout import client as client_module
from vagen_agent.rollout.client import RolloutClient
from vagen_agent.rollout.trajectory import Generated, Record, Span, assemble, dropped_sample


class _Source:
    index = 7
    group_index = 3
    prompt = ""
    label = None


def test_rollout_client_preserves_sglang_top_p_metadata(monkeypatch):
    async def fake_post(_url, payload):
        assert payload["sampling_params"]["custom_params"]["return_top_p_token_ids"]
        return {
            "text": "answer",
            "meta_info": {
                "output_token_logprobs": [[-0.1, 10, None], [-0.2, 11, None]],
                "top_p_token_ids": [10, 20, 11],
                "top_p_token_offsets": [0, 2, 3],
                "finish_reason": {"type": "stop"},
            },
        }

    monkeypatch.setattr("slime.utils.http_utils.post", fake_post)
    monkeypatch.setattr(client_module, "_router_preserves_custom_params", lambda: True)
    client = RolloutClient(
        model=None,
        record=Record(),
        args=SimpleNamespace(sglang_router_ip="127.0.0.1", sglang_router_port=30000),
    )
    out = asyncio.run(
        # The custom VAGEN client must request replay metadata itself.  Slime adds
        # custom_params in its default generation path, which custom generators do not
        # own and must not depend on.
        client._post([1, 2], [], {"top_p": 0.9})
    )

    torch.testing.assert_close(out["top_p_token_ids"], torch.tensor([10, 20, 11], dtype=torch.int32))
    torch.testing.assert_close(out["top_p_token_offsets"], torch.tensor([0, 2, 3], dtype=torch.int32))


def test_rollout_client_bypasses_public_router_without_losing_stickiness(monkeypatch):
    posted_urls = []

    async def fake_get(url):
        assert url == "http://127.0.0.1:30000/workers"
        return {
            "workers": [
                {"url": "http://127.0.0.1:31002", "worker_type": "regular", "is_healthy": True},
                {"url": "http://127.0.0.1:31001", "worker_type": "regular", "is_healthy": True},
            ]
        }

    async def fake_post(url, payload):
        posted_urls.append(url)
        assert payload["sampling_params"]["custom_params"]["return_top_p_token_ids"]
        return {
            "text": "answer",
            "meta_info": {
                "output_token_logprobs": [[-0.1, 10, None]],
                "top_p_token_ids": [10],
                "top_p_token_offsets": [0, 1],
                "finish_reason": {"type": "stop"},
            },
        }

    client_module._direct_worker_cursors.clear()
    client_module._warned_router_bypasses.clear()
    monkeypatch.setattr(client_module, "_router_preserves_custom_params", lambda: False)
    monkeypatch.setattr("slime.utils.http_utils.get", fake_get)
    monkeypatch.setattr("slime.utils.http_utils.post", fake_post)
    client = RolloutClient(
        model=None,
        record=Record(),
        args=SimpleNamespace(sglang_router_ip="127.0.0.1", sglang_router_port=30000),
    )

    asyncio.run(client._post([1, 2], [], {"top_p": 0.9}))
    asyncio.run(client._post([1, 2], [], {"top_p": 0.9}))

    assert posted_urls == ["http://127.0.0.1:31001/generate"] * 2


def _context(record, parent, ids):
    return record.add_context_node(
        parent, {"role": "user", "content": "context"}, Span(ids=list(ids), engine_ids=list(ids))
    )


def _generated(record, parent, call_id, ids, candidates, offsets):
    node = record.add_generated(
        parent,
        {"role": "assistant", "content": "answer"},
        Generated(
            call_id=call_id,
            output_ids=list(ids),
            logprobs=[-0.5] * len(ids),
            finish_reason="stop",
            top_p_token_ids=torch.tensor(candidates, dtype=torch.int32),
            top_p_token_offsets=torch.tensor(offsets, dtype=torch.int32),
        ),
        Span(ids=list(ids), engine_ids=list(ids)),
    )
    record.rewards[call_id] = [0.0] * (len(ids) - 1) + [1.0]
    return node


def test_concat_merges_top_p_replay_and_pads_context_tokens():
    record = Record()
    prompt = _context(record, record.root, [1, 2])
    first = _generated(record, prompt, 1, [10, 11], [10, 20, 11], [0, 2, 3])
    context = _context(record, first, [3])
    _generated(record, context, 2, [12], [12, 22], [0, 2])

    [sample] = assemble(record, _Source, algorithm="token_level_gae")

    assert sample.response_length == 4
    torch.testing.assert_close(
        sample.rollout_top_p_token_ids,
        torch.tensor([10, 20, 11, 12, 22], dtype=torch.int32),
    )
    torch.testing.assert_close(
        sample.rollout_top_p_token_offsets,
        torch.tensor([0, 2, 3, 3, 5], dtype=torch.int32),
    )


def test_shared_generated_prefix_gets_empty_top_p_spans_on_second_row():
    record = Record()
    prompt = _context(record, record.root, [1])
    shared = _generated(record, prompt, 1, [10], [10, 20], [0, 2])
    left_context = _context(record, shared, [2])
    _generated(record, left_context, 2, [11], [11], [0, 1])
    right_context = _context(record, shared, [3])
    _generated(record, right_context, 3, [12], [12, 22], [0, 2])

    first, second = assemble(record, _Source, algorithm="token_level_gae")

    torch.testing.assert_close(
        first.rollout_top_p_token_offsets, torch.tensor([0, 2, 2, 3], dtype=torch.int32)
    )
    assert second.loss_mask == [0, 0, 1]
    torch.testing.assert_close(
        second.rollout_top_p_token_ids, torch.tensor([12, 22], dtype=torch.int32)
    )
    torch.testing.assert_close(
        second.rollout_top_p_token_offsets, torch.tensor([0, 0, 0, 2], dtype=torch.int32)
    )


def test_dropped_sample_keeps_valid_empty_top_p_shape():
    sample = dropped_sample(_Source, "no trainable response")

    assert sample.remove_sample
    assert sample.response_length == 1
    torch.testing.assert_close(sample.rollout_top_p_token_ids, torch.empty(0, dtype=torch.int32))
    torch.testing.assert_close(
        sample.rollout_top_p_token_offsets, torch.tensor([0, 0], dtype=torch.int32)
    )
