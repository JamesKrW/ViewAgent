"""Prove the Qwen adapter tokenizes a conversation the way the chat template does.

This is the test that matters most for the model axis, because the failure it catches is
invisible at runtime: if the incremental render disagrees with the template by a token or
two at each turn boundary, the rollout and the training pass *both* see the same seam, so
no loss, metric or assertion downstream can tell. It is off-distribution input, not a
mismatch. The only place to catch it is here, against a real tokenizer.

Runs on CPU, needs no slime and no GPU -- just a cached tokenizer.

    python tests/test_model_adapter.py                    # every cached family
    python tests/test_model_adapter.py Qwen/Qwen3.5-4B    # one
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vagen_agent.models import ModelAdapterError, build_model_adapter  # noqa: E402

#: Families to check, and whether we expect a processor (i.e. a multimodal checkpoint).
CANDIDATES = [
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
]


def load(name: str):
    """Tokenizer plus processor if the checkpoint has one. Local cache only -- this test
    must not depend on the network."""
    from transformers import AutoProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True, trust_remote_code=True)
    processor = None
    try:
        candidate = AutoProcessor.from_pretrained(name, local_files_only=True, trust_remote_code=True)
        # AutoProcessor happily returns a *tokenizer* for text-only checkpoints. Treating
        # that as a processor would make `supports_images` true for a model that cannot see,
        # and the modality check would then wave through a vision environment.
        if hasattr(candidate, "image_processor"):
            processor = candidate
    except Exception:
        processor = None
    return tokenizer, processor


def check(name: str) -> tuple[bool, str]:
    tokenizer, processor = load(name)
    adapter = build_model_adapter("qwen", tokenizer, processor)

    adapter.self_test()

    facts = [
        f"images={adapter.supports_images}",
        f"separator={adapter.message_separator()}",
        f"prefix_len={len(adapter._continuation_prefix())}",
        f"placeholders={sorted(adapter.image_placeholder_ids()) or '-'}",
        f"sentinels={sorted(adapter.vision_sentinel_ids()) or '-'}",
        f"isolated_user_turn={adapter.can_render_isolated_user_turn()}",
    ]

    # A second, independent shape: a deeper exchange, so the agreement is not an accident
    # of the two-turn case the self-test uses. Same suffix property -- the render of the
    # last observation must be exactly what the template puts at the end.
    u1 = {"role": "user", "content": "first"}
    a1 = {"role": "assistant", "content": "reply one"}
    u2 = {"role": "user", "content": "second"}
    a2 = {"role": "assistant", "content": "reply two"}
    u3 = {"role": "user", "content": "third observation, longer than the others"}
    canonical = adapter._render_canonical([u1, a1, u2, a2, u3])
    continuation = adapter.block(u3, generation_prompt=True).ids
    if canonical[-len(continuation):] != continuation:
        return False, (f"deep-conversation suffix differs ({len(continuation)} tokens)\n"
                       f"      {' '.join(facts)}")

    # And the system-prompt shape the runner actually opens with.
    sysmsg = {"role": "system", "content": "You play Sokoban."}
    opening = adapter.render_group([sysmsg, u1]).ids
    full = adapter._render_canonical([sysmsg, u1])
    if opening != full:
        return False, (f"system+user opening differs: {len(opening)} vs {len(full)}\n"
                       f"      {' '.join(facts)}")

    facts.append(f"deep={len(canonical)}tok OK")
    return True, "  ".join(facts)


def main() -> int:
    names = sys.argv[1:] or CANDIDATES
    failures = 0
    skipped = 0
    for name in names:
        try:
            ok, detail = check(name)
        except (OSError, ValueError) as exc:
            if "local_files_only" in str(exc) or "not a local folder" in str(exc) or "Can't load" in str(exc):
                print(f"SKIP  {name:32s} not in the local cache")
                skipped += 1
                continue
            if "chat_template is not set" in str(exc):
                # A base checkpoint has no chat template, which is not an adapter fault --
                # nothing in this port can run against one anyway, since every turn is a
                # chat turn.
                print(f"SKIP  {name:32s} no chat template (base checkpoint)")
                skipped += 1
                continue
            print(f"ERROR {name:32s} {type(exc).__name__}: {exc}")
            failures += 1
            continue
        except ModelAdapterError as exc:
            if "chat_template is not set" in str(exc):
                print(f"SKIP  {name:32s} no chat template (base checkpoint)")
                skipped += 1
                continue
            print(f"FAIL  {name:32s} {exc}")
            failures += 1
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {name:32s} {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failures += 1
            continue
        print(f"{'PASS ' if ok else 'FAIL '} {name:32s} {detail}")
        failures += 0 if ok else 1
    print(f"\n{len(names) - failures - skipped} passed, {failures} failed, {skipped} skipped")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
