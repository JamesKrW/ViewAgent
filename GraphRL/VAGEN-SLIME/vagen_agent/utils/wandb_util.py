"""Validation episodes as HTML tables, for wandb.

Reporting only -- nothing here is on the rollout path, and `log_eval_episodes` swallows
its own failures: a picture is not worth a training step.

An eval number tells you the policy got worse; it does not tell you that it started
emitting two actions per turn, or that the frame stopped arriving, or that compaction ate
the goal state. Reading a handful of episodes does.

What is rendered is the **training sequence itself** -- the token stream of the Sample,
with each run of image-placeholder tokens replaced by the frame it stands for and each
model-generated span marked. Not a reconstruction from text columns: a frame appears where
the sequence says it appears, and the highlighted spans are exactly the mask-1 positions,
so a mask that ran past a turn boundary is visible as highlighted template text.

Naming, since the three levels are easy to conflate:

* **episode** -- one whole agent/environment interaction; every row shares its rollout_id
* **conversation** -- one continuous exchange, and one training row
* **turn** -- one model call inside a conversation
"""

from __future__ import annotations

import base64
import html
import io
import logging
from typing import Any

logger = logging.getLogger(__name__)

MAX_IMAGE_WIDTH = 192
_STYLE = ("font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;"
          "line-height:1.45;white-space:pre-wrap;word-break:break-word")
_GENERATED = "background:#e8f5e9;border-left:3px solid #2e7d32;padding:2px 6px;margin:4px 0"
_CONTEXT = "color:#555;padding:2px 6px;margin:4px 0"
#: A token the environment paid on. Worth marking now that this is the *raw* reward:
#: it is sparse and it says which turn earned what. Marking the folded vector was not --
#: under default_gae that is the same value on the last token of every row.
_PAID = ("background:#fff3cd;border-left:3px solid #b8860b;padding:2px 6px;margin:4px 0;"
         "font-weight:600")
#: What the environment paid for the turn, on its own line under the turn. The number goes
#: here rather than beside the token: a dense reward vector would print it once per token
#: and drown the transcript, and the question a reader has is what this turn earned, not
#: which position in the span it was attached to. The colour already says that.
_TURN_REWARD = ("color:#8a6d00;font-size:11px;padding:1px 6px;margin:0 0 8px;"
                "border-left:3px solid #e8d9a0")
_LEGEND = (
    '<div style="font-size:11px;margin:0 0 10px;padding:6px 8px;background:#fafafa;'
    'border:1px solid #ddd">'
    '<span style="background:#e8f5e9;border-left:3px solid #2e7d32;padding:1px 5px">'
    'trained</span> &nbsp;mask=1, carries gradient &mdash; exactly the ids the server '
    'returned &nbsp;&nbsp;'
    '<span style="color:#555;padding:1px 5px">context</span> &nbsp;mask=0, prompt / '
    'observation / template &nbsp;&nbsp;'
    '<span style="background:#fff3cd;border-left:3px solid #b8860b;padding:1px 5px">'
    'paid</span> &nbsp;a token the environment gave reward on; the amount is under the '
    'turn, raw, before any estimator folded it &nbsp;&nbsp;'
    '<br><span style="color:#777">A frame replaces the run of placeholder tokens it '
    'stands for. Highlighted template text would mean the mask ran past a turn '
    'boundary.</span></div>')


def _img_tag(image: Any) -> str:
    """A PIL frame inline. Downscaled before encoding, not merely styled down: the payload
    is otherwise whatever the environment rendered, which is invisible at Sokoban's 192px
    and costs in proportion to resolution for anything larger."""
    try:
        frame = image.convert("RGB")
        if frame.width > MAX_IMAGE_WIDTH:
            height = max(1, round(frame.height * MAX_IMAGE_WIDTH / frame.width))
            frame = frame.resize((MAX_IMAGE_WIDTH, height))
        buf = io.BytesIO()
        frame.save(buf, format="PNG", optimize=True)
        payload = base64.b64encode(buf.getvalue()).decode()
    except Exception:  # noqa: BLE001 - a frame that will not encode must not lose the text
        return ""
    return (f'<img src="data:image/png;base64,{payload}" '
            f'style="max-width:{MAX_IMAGE_WIDTH}px;display:block;margin:6px 0">')


def _turn_html(run: list[int], base: int, raw: list[float], tokenizer) -> list[str]:
    """One turn: its generated text, the paid tokens in yellow, and what it earned.

    ``base`` is the run's offset into the response region, so ``raw[base + k]`` is the
    reward on the run's k-th token. A generated run *is* a turn -- the mask breaks at
    every observation -- so summing the vector over the run gives the turn's reward.
    """
    out, buffer = [], []

    def flush():
        if not buffer:
            return
        text = tokenizer.decode(buffer, skip_special_tokens=False)
        buffer.clear()
        if text.strip():
            out.append(f'<div style="{_GENERATED}">{html.escape(text)}</div>')

    total = 0.0
    for k, token in enumerate(run):
        index = base + k
        value = raw[index] if 0 <= index < len(raw) else 0.0
        if not value:
            buffer.append(token)
            continue
        total += value
        flush()
        out.append(f'<div style="{_PAID}">'
                   f'{html.escape(tokenizer.decode([token], skip_special_tokens=False))}</div>')
    flush()
    out.append(f'<div style="{_TURN_REWARD}">Summed Turn Reward: {total:+.4f}</div>')
    return out


def _runs(tokens: list[int], mask: list[int], prompt_len: int, placeholders: set[int]):
    """Split a row into ``(kind, payload)`` runs: 'gen' / 'ctx' text, or 'img'.

    ``mask`` covers only the response region, so positions before ``prompt_len`` are
    context by construction.
    """
    full_mask = [0] * prompt_len + list(mask or [])
    full_mask += [0] * (len(tokens) - len(full_mask))
    out, run, kind = [], [], None
    for token, m in zip(tokens, full_mask):
        this = "img" if token in placeholders else ("gen" if m else "ctx")
        if this != kind:
            if run:
                out.append((kind, run))
            run, kind = [], this
        run.append(token)
    if run:
        out.append((kind, run))
    return out


def episode_html(samples, tokenizer, placeholders: set[int]) -> str:
    """One episode, in the order it happened. ``samples`` are its rows, in row order."""
    parts = [f'<div style="{_STYLE}">', _LEGEND]
    for n, sample in enumerate(samples):
        if n:
            parts.append('<hr style="border:0;border-top:2px solid #c00;margin:18px 0 6px">')
        meta = sample.metadata or {}
        # Position in the list first. An episode is a list of Samples in order, whatever
        # the harness did to produce it -- one under concat, one per turn under no_concat,
        # one per compaction under compact -- so the renderer numbers the list and never
        # asks which policy it was. `round_number` is shown only when it disagrees, which
        # would mean the transcript is not the list that was returned.
        ordinal = meta.get("round_number", n)
        drift = "" if ordinal == n else f' <span style="color:#c00">round_number {ordinal}</span>'
        parts.append(f'<div style="color:#c00;font-size:13px">'
                     f'<b>#{n} conversation</b> '
                     f'<span style="font-weight:400;color:#777">({n + 1} of {len(samples)})'
                     f'</span>{drift}</div>')
        frames = list((sample.multimodal_inputs or {}).get("images") or [])
        prompt_len = len(sample.tokens) - int(sample.response_length or 0)
        mask = list(sample.loss_mask or [])
        # What the *environment* paid, before any estimator folded it. Showing the folded
        # vector instead makes the transcript algorithm-dependent: one episode reads as
        # the same number on every row under default_gae and as a different set under
        # token_level_gae, while the environment did exactly the same thing both times.
        raw = list(getattr(sample, "vagen_raw_scores", None) or [])
        parts.append(f'<div style="color:#777;font-size:11px">'
                     f'{sum(mask)} trained / {len(sample.tokens)} tokens'
                     f'{"" if not frames else f", {len(frames)} frame(s)"}'
                     f' &middot; summed up raw reward: {sum(raw):+.4f}</div>')
        if raw and len(raw) != len(mask):
            # The reward vector is indexed by response position, so a length that
            # disagrees with the mask means the marks below are on the wrong tokens.
            parts.append(f'<div style="color:#c00"><b>reward vector is {len(raw)} long '
                         f'against {len(mask)} response tokens</b></div>')
        used = 0
        offset = 0
        for kind, run in _runs(list(sample.tokens), sample.loss_mask, prompt_len, placeholders):
            start, offset = offset, offset + len(run)
            if kind == "img":
                # One run of placeholder tokens is one frame.
                if used < len(frames):
                    parts.append(_img_tag(frames[used]))
                used += 1
                continue
            if kind == "gen":
                parts.extend(_turn_html(run, start - prompt_len, raw, tokenizer))
                continue
            text = tokenizer.decode(run, skip_special_tokens=False)
            if not text.strip():
                continue
            parts.append(f'<div style="{_CONTEXT}">{html.escape(text)}</div>')
        if used != len(frames):
            parts.append(f'<div style="color:#c00"><b>frame/placeholder mismatch: '
                         f'{used} runs against {len(frames)} frames</b></div>')
    parts.append("</div>")
    return "".join(parts)


def select_episodes(episodes: list[dict], n: int, success_ratio: float = 0.5) -> list[dict]:
    """A balanced sample: some the policy solved, some it did not.

    All-success is the usual accident and it is the least informative set there is -- the
    failures are what say *how* it fails.
    """
    if len(episodes) <= n:
        return episodes
    solved = [e for e in episodes if e["success"]]
    failed = [e for e in episodes if not e["success"]]
    want_ok = min(len(solved), max(0, round(n * success_ratio)))
    want_bad = min(len(failed), n - want_ok)
    want_ok = min(len(solved), n - want_bad)          # backfill if one side is short
    return solved[:want_ok] + failed[:want_bad]


#: Fields shown per episode, in order. Keep in step with ``build_table``.
_PER_EPISODE = ("episode_id", "turns", "reward", "success", "transcript")

#: The growing table, per eval dataset. wandb tables are immutable once logged, so a
#: table that grows has to be rebuilt from the previous rows -- the documented workaround.
_TABLES: dict[str, Any] = {}


def build_table(episodes: list[dict], step: int, previous=None):
    """**One row per step**, with the chosen episodes side by side across the columns.

    Not one row per episode. Both fit in a table, and the difference is what you can read
    off it: with a row per step you scroll down and watch the same slot evolve, and step 0
    sits above step 20 in one view. With a row per episode each eval is a separate table
    and comparing two steps means scrubbing a slider between them.
    """
    import wandb

    columns = ["step"] + [f"ep{i}_{f}" for i in range(len(episodes)) for f in _PER_EPISODE]
    table = wandb.Table(columns=columns,
                        data=list(previous.data) if previous is not None else [])
    row: list[Any] = [step]
    for e in episodes:
        row += [str(e["episode_id"]), e["turns"], e["reward"], e["success"],
                wandb.Html(e["html"])]
    table.add_data(*row)
    return table


def publish(name: str, episodes: list[dict], step: int) -> None:
    """Append this step's row to the dataset's table and log it."""
    import wandb

    if not episodes:
        return
    # `transcripts`, not `episodes`: `episode_metrics` already logs
    # `eval/<name>-episodes` as the episode *count*, and a key that holds a number on one
    # log and a Table on the next does not surface as a table panel at all. The artifact
    # still uploads, which is what makes this look like a wandb problem rather than a
    # collision of our own making.
    previous = _TABLES.get(name)
    width = 1 + len(episodes) * len(_PER_EPISODE)
    if previous is not None and len(previous.columns) != width:
        # A step that selected fewer episodes cannot extend the same table -- the column
        # count is fixed by the first row. Give it its own key rather than dropping it.
        wandb.log({f"eval/{name}-transcripts_{len(episodes)}": build_table(episodes, step),
                   "rollout/step": step})
        return
    _TABLES[name] = build_table(episodes, step, previous=previous)
    wandb.log({f"eval/{name}-transcripts": _TABLES[name], "rollout/step": step})


def log_eval_episodes(args, data, step: int, n: int = 8) -> None:
    """Publish a few validation episodes per dataset. Never raises: a picture is not worth
    a training step."""
    try:
        import wandb

        if not getattr(args, "use_wandb", False) or wandb.run is None:
            return
        from slime.utils.processing_utils import load_processor, load_tokenizer

        from vagen_agent.models._common.image_tokens import image_token_ids

        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        placeholders = image_token_ids(processor or tokenizer) if (processor or tokenizer) else set()

        for name, info in (data or {}).items():
            by_episode: dict[Any, list] = {}
            for sample in info.get("samples") or []:
                key = getattr(sample, "rollout_id", None)
                by_episode.setdefault(key if key is not None else id(sample), []).append(sample)

            episodes = []
            for key, rows in by_episode.items():
                rows.sort(key=lambda s: (s.metadata or {}).get("round_number", 0))
                meta = rows[0].metadata or {}
                env_metrics = meta.get("metrics") or {}
                episodes.append({
                    "episode_id": key,
                    "turns": int(meta.get("episode_turns", len(rows))),
                    "reward": round(float(meta.get("episode_reward", rows[0].reward)), 4),
                    "success": bool(
                        env_metrics.get("success", meta.get("traj_success", False))
                    ),
                    "rows": rows,
                })
            chosen = select_episodes(episodes, n)
            for e in chosen:
                e["html"] = episode_html(e.pop("rows"), tokenizer, placeholders)
            publish(name, chosen, step)
    except Exception:  # noqa: BLE001
        logger.warning("could not publish eval episodes to wandb", exc_info=True)


__all__ = ["build_table", "episode_html", "log_eval_episodes", "publish",
           "select_episodes"]
