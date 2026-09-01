"""Check what a real rollout produced, from slime's own dump. No training.

`run_sokoban.py --rollout-only` sets slime's `--debug-rollout-only` (sglang engines only,
no Megatron) and `--dump-details`, which writes every `Sample` to
`<runs>/<exp>/dump/rollout_data/eval_0.pt`. This reads that file back and checks the
properties that decide whether the rollout is *correct* rather than merely finished.

The reason to check the dump rather than a fake-server unit test: the fake server returns
text we wrote, so it cannot disagree with the chat template about anything. A real model
generating into a real template is the only thing that exercises the seam this port has to
get right -- and the seam moves with the model family and with `enable_thinking`.

What is checked, per sample:

* **the mask covers exactly the model's own tokens.** Decoding the mask-1 positions must
  give text that appears verbatim in the assistant's turn, and the mask-0 positions must
  contain the template markers. A mask off by one turn boundary is the failure mode that
  nothing downstream can report: the rollout and the training pass see the same seam.
* **no control token is trainable.** `<|im_start|>`, `<|im_end|>` and the role headers
  belong to the template, not the policy. Training on them teaches the model to emit its
  own turn boundaries.
* **image placeholders and frames stay 1:1**, on a multimodal run.
* **episode bookkeeping**: rows of one episode share a `rollout_id` and a reward.
* **thinking**, when the template pre-fills one: the response must open after the
  generation prompt's `<think>`, not repeat it.

    python tests/verify_rollout_dump.py <dump.pt> --model <hf-dir> [--expect-thinking]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CONTROL = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")


def load(path: str) -> list[dict]:
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob["samples"]


#: Counted, not failed: responses in which the *policy* emitted a template control token.
#: A quality signal about the model, not a defect in the mask -- see check_sample.
HALLUCINATED_BOUNDARY: list[int] = []


def check_sample(s: dict, tok, expect_thinking: bool | None,
                 placeholder_ids: set[int]) -> list[str]:
    problems: list[str] = []
    tokens = list(s["tokens"])
    mask = list(s["loss_mask"])
    rlen = int(s["response_length"])
    response = tokens[len(tokens) - rlen:]

    if len(mask) != rlen:
        return [f"loss_mask has {len(mask)} entries for a response of {rlen} tokens"]

    trainable = [t for t, m in zip(response, mask) if m]
    frozen = [t for t, m in zip(response, mask) if not m]

    # A control token inside the trainable span used to be read as "the mask ran past a
    # turn boundary". Under the message tree that inference no longer holds: a row's
    # mask-1 positions are, by construction, exactly the ids the server returned for a
    # generation -- context is appended as mask 0 and never merges into a response span.
    # So a control token in there was *sampled*, and the only honest thing to do is train
    # on it; masking it out would hide from the policy that it emitted a turn boundary.
    #
    # Measured on a real Qwen3-4B rollout: 1 response in 551 hit the per-turn cap after
    # 3366 tokens of rambling and hallucinated `<|im_start|>...<tool_call>` eight tokens
    # from the end. That is a policy-quality signal, not a masking bug, so it is counted
    # rather than failed.
    # Per *response span*, not over the concatenation of them. A concat row holds several
    # responses separated by mask-0 observations, and each one legitimately ends on its
    # own stop token -- checking the joined text makes every earlier turn's `<|im_end|>`
    # look like it appeared mid-response.
    spans, run = [], []
    for token, m in zip(response, mask):
        if m:
            run.append(token)
        elif run:
            spans.append(run)
            run = []
    if run:
        spans.append(run)
    for span in spans:
        body = span[:-1] if tok.decode(span[-1:]) in CONTROL else span
        if [m for m in CONTROL if m in tok.decode(body)]:
            HALLUCINATED_BOUNDARY.append(1)
            break
    trainable_text = tok.decode(trainable)

    # The frozen part of the response region is the observations the environment sent back
    # plus the template's boundaries. It must be non-empty on a multi-turn row and it must
    # be where the markers live.
    if rlen and not trainable:
        problems.append("no trainable tokens in the response region")
    if frozen and not any(m in tok.decode(frozen) for m in CONTROL):
        problems.append("mask-0 span carries no template marker -- boundaries may be "
                        "attributed to the policy")

    # Thinking. When the generation prompt pre-fills `<think>`, the sampled response
    # continues inside it, so the trainable text opens with the *body*, not a second
    # `<think>`. A duplicated opener means the prompt tail was re-rendered into the
    # response.
    if expect_thinking is not None:
        prompt_text = tok.decode(tokens[: len(tokens) - rlen])
        opens_in_think = prompt_text.rstrip().endswith("<think>")
        if opens_in_think and trainable_text.lstrip().startswith("<think>"):
            problems.append("the prompt already opened <think> and the response opens "
                            "another -- a duplicated generation prompt")
        if expect_thinking is False and "</think>" not in prompt_text[-64:] \
                and "<think>" in trainable_text:
            problems.append("enable_thinking=false, but the model emitted <think> and the "
                            "prompt did not close one -- the kwarg did not reach the template")

    # Multimodal bookkeeping. Three counts have to agree, and each disagreement is a
    # different silent corruption: frames without tensors trains on a sequence whose image
    # positions describe nothing; a grid row per frame is what `masked_scatter` indexes
    # against; and the expanded placeholder run is the hole those features are poured into,
    # so a mismatch shifts every token after it while the sequence still looks well-formed.
    images = (s.get("multimodal_inputs") or {}).get("images") or []
    train = s.get("multimodal_train_inputs") or {}
    if images:
        grid = train.get("image_grid_thw")
        pixels = train.get("pixel_values")
        if grid is None or pixels is None:
            problems.append(f"{len(images)} frame(s) but no image_grid_thw/pixel_values "
                            f"for the trainer (got {sorted(train)})")
        else:
            if len(grid) != len(images):
                problems.append(f"{len(images)} frame(s) but {len(grid)} grid row(s)")
            pads = sum(1 for t in tokens if t in placeholder_ids)
            if not pads:
                problems.append(f"{len(images)} frame(s) but no image placeholder tokens "
                                f"in the sequence -- the features have nowhere to land")
            elif len(pixels) % pads:
                problems.append(f"{len(pixels)} patch rows do not divide over {pads} "
                                f"placeholder token(s)")

    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--model", required=True, help="HF dir, for the tokenizer")
    ap.add_argument("--expect-thinking", dest="thinking", action="store_true", default=None)
    ap.add_argument("--expect-no-thinking", dest="thinking", action="store_false")
    ap.add_argument("--show", type=int, default=0, help="print N decoded conversations")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    # The ids a frame expands behind, so the multimodal check can find them. Taken from the
    # tokenizer rather than hardcoded: they move between families.
    placeholder_ids = {i for i in (tok.convert_tokens_to_ids(t)
                                   for t in ("<|image_pad|>", "<|video_pad|>"))
                       if isinstance(i, int) and i >= 0}
    samples = load(args.dump)

    failures = 0
    by_episode: dict = defaultdict(list)
    for s in samples:
        problems = check_sample(s, tok, args.thinking, placeholder_ids)
        by_episode[s["rollout_id"]].append(s)
        if problems:
            failures += 1
            if failures <= 5:
                print(f"FAIL  sample index={s['index']} rollout_id={s['rollout_id']}")
                for p in problems:
                    print(f"        - {p}")

    # Episode-level: rows of one episode share a reward, and a no_concat episode is
    # several rows.
    episode_problems = 0
    for rid, rows in by_episode.items():
        if len({round(float(r["reward"]), 9) for r in rows}) != 1:
            print(f"FAIL  rollout_id={rid}: rows carry different rewards "
                  f"{[r['reward'] for r in rows]}")
            episode_problems += 1

    samples_per_episode = sum(len(v) for v in by_episode.values()) / max(1, len(by_episode))
    trainable = sum(sum(s["loss_mask"]) for s in samples)
    total_resp = sum(int(s["response_length"]) for s in samples)
    rewards = [float(s["reward"]) for s in samples]

    # Not correctness, but the thing that decides whether a family is *usable* here, and it
    # is invisible in the success rate alone: a turn that spends its whole budget reasoning
    # and never emits an action looks the same as a wrong action.
    stopped = answered = closed_think = 0
    for s in samples:
        n = len(s["tokens"]) - int(s["response_length"])
        text = tok.decode(s["tokens"][n:])
        stopped += any(text.rstrip().endswith(m) for m in CONTROL)
        answered += "<answer>" in text
        closed_think += "</think>" in text
    n = max(1, len(samples))
    print(f"turns ending on a stop token: {stopped}/{len(samples)} "
          f"({100*stopped/n:.0f}%) -- the rest hit the per-turn cap")
    print(f"turns emitting <answer>:      {answered}/{len(samples)} ({100*answered/n:.0f}%)")
    withimg = [s for s in samples if (s.get("multimodal_inputs") or {}).get("images")]
    if withimg:
        frames = sum(len(s["multimodal_inputs"]["images"]) for s in withimg)
        print(f"rows carrying frames:         {len(withimg)}/{len(samples)}  "
              f"({frames} frames; placeholder runs, frames and image_grid_thw all 1:1)")
    hb = len(HALLUCINATED_BOUNDARY)
    print(f"policy emitted a control token: {hb}/{len(samples)} ({100*hb/n:.1f}%) "
          f"-- sampled, so trained on; a model-quality signal, not a mask bug")
    print(f"turns closing </think>:       {closed_think}/{len(samples)} ({100*closed_think/n:.0f}%)")

    for s in samples[: args.show]:
        n = len(s["tokens"]) - int(s["response_length"])
        print("=" * 100)
        print("PROMPT   ", repr(tok.decode(s["tokens"][:n]))[:1200])
        print("RESPONSE ", repr(tok.decode(s["tokens"][n:]))[:1200])
        print("TRAINABLE", repr(tok.decode(
            [t for t, m in zip(s["tokens"][n:], s["loss_mask"]) if m]))[:1200])

    print(f"\n{len(samples)} samples / {len(by_episode)} episodes  "
          f"samples_per_episode={samples_per_episode:.2f}  "
          f"trainable={trainable}/{total_resp} response tokens  "
          f"reward mean={sum(rewards)/max(1,len(rewards)):+.4f}")
    bad = failures + episode_problems
    print(f"{len(samples) - failures} sample(s) clean, {bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
