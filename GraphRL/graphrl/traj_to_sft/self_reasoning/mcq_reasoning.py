"""MCQ-style reasoning annotation: natural-conversation prompt + retry.

Same explode-and-reassemble framework as :mod:`single_turn` (one annotation
job per assistant turn, multi-process VAGEN rollout, reassemble into the
original multi-turn record), but each per-turn job is a **2-choice MCQ**:

  * The annotator sees the original navigation conversation up to the
    decision point AS A NATURAL CHAT TRANSCRIPT (system = original game
    system prompt; user = original first observation + images; "previous
    decisions" rendered inline as the GT action names; final user message
    = current observation with the MCQ question appended).
  * Two options: GT vs opposite/distractor (A/B order shuffled per turn).
  * Asked to output
    ``<observation>...</observation> reasoning <action>X</action>``
    where X is a LETTER. Not told which letter is correct.
  * Up to ``max_attempts`` tries. Wrong-letter or format-error feedback is
    fed back as the next user turn; same MCQ stays visible. On success the
    letter is substituted with the actual GT action and the reasoning body
    becomes the augmented per-turn content.
  * Salvage on final failure: if the last reply contains ``<action>X</action>``
    with the correct letter (i.e. answer right, format slightly wrong),
    keep the prefix as reasoning. Otherwise fall back to the original
    ``<action>...</action>`` content via the standard reassembly salvage.

Per-dataset MCQ construction (in :func:`_build_mcq_choices`):

  * ``multi_turn_action_gen``: GT vs. opposite primitive
    (``turn_left`` ↔ ``turn_right``, ``move_forward`` ↔ ``move_backward``,
    etc.). Turns whose action is ``answer(...)`` or any unknown primitive
    are skipped (those turns fall back to original content with no
    reasoning, since perturbing a 6-DoF pose into a "wrong but plausible"
    distractor isn't well-defined).
  * ``view_difference``: GT integer vs. random different integer (uniform
    over a small range around GT).
  * ``view_difference_mcq``: original prompt already lists multiple
    options (e.g. "A. 5 B. 8 C. 12 D. 15"); we keep the GT plus one
    randomly-sampled distractor option.

The final SFT record is bit-shape-identical to what
:class:`SingleTurnReasoner` produces (assistant content =
``<observation>...</observation> ... <action>{actual_value}</action>``)
so the downstream LLaMA-Factory SFT phase doesn't need any change.

Wired in via ``traj_to_sft.reasoning.reasoner_cls``::

    traj_to_sft:
      reasoning:
        enabled: true
        reasoner_cls: graphrl.traj_to_sft.self_reasoning.mcq_reasoning.MCQReasoner
        single_turn:
          recent_k: 2
        # max_turns = max number of MCQ attempts; each retry is one vagen
        # turn with the wrong/format-error feedback as the user message.
        max_turns: 3
"""
from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from view_suite.gym.gym_image_env import GymImageEnv

from .augment import run_vagen_eval_and_collect
from .base import BaseChecker, BaseDataset, Datapoint, TurnCheck
from .postprocess import _USE_ORIGINAL, _inject_system_suffix
from .reasoner import DEFAULT_PROMPTS_DIR
from .sglang_server import SGLangServer
from .single_turn import (
    SingleTurnReasoner,
    _build_subrecord,
    _kept_user_turns,
)

logger = logging.getLogger(__name__)


# ── Action opposites for multi_turn_action_gen MCQ construction ──────────
# When the GT is a primitive in this map, the opposite is used as the
# distractor. Actions outside this map (notably ``answer(...)``) are
# skipped — those turns fall back to original content with no reasoning.
_ACTION_OPPOSITES: Dict[str, str] = {
    "turn_left":     "turn_right",
    "turn_right":    "turn_left",
    "move_forward":  "move_backward",
    "move_backward": "move_forward",
    "move_left":     "move_right",
    "move_right":    "move_left",
    "move_up":       "move_down",
    "move_down":     "move_up",
    "look_up":       "look_down",
    "look_down":     "look_up",
    "rotate_cw":     "rotate_ccw",
    "rotate_ccw":    "rotate_cw",
}

_ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL)
_OBS_RE = re.compile(r"<observation>(.*?)</observation>", re.DOTALL)
# Used to parse view_difference_mcq's question — looks for "A. <text>" patterns
# in the user message body. Permissive: allows multi-line option content but
# stops at the next "X. " marker or end-of-text.
_MCQ_OPTION_RE = re.compile(r"\b([A-Z])\.\s+(.*?)(?=\s*[A-Z]\.|$)", re.DOTALL)


# ── MCQ choice construction (per-dataset heuristics) ──────────────────────


def _extract_action_value(text: str) -> Optional[str]:
    """Pull the content out of ``<action>X</action>``. Returns None when no tag."""
    m = _ACTION_RE.search(text or "")
    return m.group(1).strip() if m else None


def _build_mcq_choices(
    target_dataset: str,
    gt_action: str,
    user_message: str,
    rng: random.Random,
) -> Optional[Tuple[str, str]]:
    """Return ``(gt_value, neg_value)`` or None to skip MCQ for this turn.

    ``gt_action`` is the raw content of the GT ``<action>...</action>``.
    ``user_message`` is the FULL text of the most recent user turn (used by
    view_difference_mcq to recover the original distractors).

    The same value gets stored under whichever letter wins the shuffle, so
    the ordering decision is left to the caller.
    """
    if target_dataset.startswith("multi_turn_action_gen"):
        # Strip any whitespace/punct that crept in. Multi-action edges like
        # ``turn_left|move_forward`` are filtered upstream by the
        # ``single_mix_multi_ratio: "1:0:0"`` knob, so we expect a single
        # primitive here. Any leftover unexpected form → skip MCQ.
        primitive = gt_action.strip()
        if primitive in _ACTION_OPPOSITES:
            return primitive, _ACTION_OPPOSITES[primitive]
        # ``answer(...)`` and anything we don't have an opposite for: skip.
        return None

    if target_dataset.startswith("view_difference_mcq"):
        # Try to parse the original options from the user prompt; pick a
        # random distractor that isn't the GT.
        options = _parse_mcq_options(user_message)
        if not options:
            return None
        distractors = [v for v in options.values() if v.strip() != gt_action.strip()]
        if not distractors:
            return None
        return gt_action, rng.choice(distractors)

    if target_dataset.startswith("view_difference"):
        # GT is a number (integer typically). Generate a different number
        # in a small neighbourhood so the distractor is plausible.
        try:
            gt_num = int(gt_action.strip())
        except ValueError:
            # Free-form GT — fall back to no MCQ.
            return None
        # Distractor = GT ± rng[1, 4]; flip sign uniformly.
        delta = rng.randint(1, 4)
        if rng.random() < 0.5:
            delta = -delta
        neg_num = max(0, gt_num + delta) if delta < 0 else gt_num + delta
        if neg_num == gt_num:
            neg_num = gt_num + 1
        return str(gt_num), str(neg_num)

    return None


def _parse_mcq_options(user_message: str) -> Dict[str, str]:
    """Parse "A. <text> B. <text> ..." style options from a user message.

    Returns ``{letter: option_text}``. Empty dict if no parseable options.
    Best-effort — view_difference_mcq prompts can be slightly free-form.
    """
    out: Dict[str, str] = {}
    for m in _MCQ_OPTION_RE.finditer(user_message):
        letter = m.group(1)
        text = m.group(2).strip().rstrip(".,;:")
        # Filter out things like "Step 1." that match the regex but aren't
        # MCQ options (single-digit number followed by dot).
        if len(letter) == 1 and "A" <= letter <= "Z":
            out[letter] = text
    # Require at least 2 options to count as an MCQ.
    return out if len(out) >= 2 else {}


def _format_mcq_question(letter_to_value: Dict[str, str]) -> str:
    """Render a 2-choice MCQ block (or 1-choice for the reduced retry).

    Letters are emitted in alphabetical order so display is deterministic
    even after a shuffle has assigned which value goes to which letter.
    """
    lines = ["", "=== QUESTION ==="]
    if len(letter_to_value) == 2:
        lines.append("Which option should you choose? Answer with the LETTER inside <action>...</action>.")
    else:
        lines.append(
            "Only one option remains. Choose it and explain your reasoning in the same format."
        )
    for letter in sorted(letter_to_value):
        lines.append(f"{letter}. {letter_to_value[letter]}")
    return "\n".join(lines)


# ── Datapoint / dataset ───────────────────────────────────────────────────


@dataclass
class MCQDatapoint(Datapoint):
    """Datapoint extended with MCQ state for the env's state machine."""
    correct_letter: str = "A"
    letter_to_value: Dict[str, str] = field(default_factory=dict)
    base_user_text: str = ""           # last-user-turn text WITHOUT the MCQ question
    system_text: str = ""              # original game system prompt for this record


class MCQExplodedDataset(BaseDataset):
    """Reads the exploded SFT JSON written by :class:`MCQReasoner`.

    Each record carries the standard ShareGPT shape PLUS:

      * ``_correct_letter``: "A" or "B" — which letter the GT was
        randomly assigned to in the shuffle.
      * ``_letter_to_value``: full mapping (e.g. ``{"A": "turn_left",
        "B": "turn_right"}``).
      * ``_base_user_text``: the original last-user-turn text WITHOUT
        the MCQ question appended — used by the env to re-render the
        question for the reduced (attempt-2) prompt.

    Records that have no MCQ assignment (e.g. ``answer(...)`` turn
    skipped) just won't appear in the exploded list; their per-turn
    reasoning will fall back to the original ``<action>...</action>``
    content via the standard reassembly salvage.
    """

    def __init__(
        self,
        sft_path: str,
        image_root: Optional[str] = None,
        image_size: Optional[List[int]] = None,
    ):
        self.sft_path = Path(sft_path)
        self.image_root = Path(image_root) if image_root else self.sft_path.parent
        self.image_size = tuple(image_size) if image_size else None
        with open(self.sft_path, encoding="utf-8") as f:
            self._records: List[Dict[str, Any]] = json.load(f)

    def __len__(self) -> int:
        return len(self._records)

    def get(self, idx: int) -> Datapoint:
        rec = self._records[idx]
        imgs = [self._load(p) for p in rec.get("images", [])]
        correct_letter = rec.get("_correct_letter")
        letter_to_value = rec.get("_letter_to_value") or {}
        base_user_text = rec.get("_base_user_text", "")
        system_text = ""
        for m in rec.get("messages", []):
            if m.get("role") == "system":
                system_text = m.get("content", "")
                break
        if not correct_letter or not letter_to_value:
            raise ValueError(
                f"MCQExplodedDataset record {idx} is missing _correct_letter / _letter_to_value"
            )
        return MCQDatapoint(
            idx=idx,
            messages=rec["messages"],
            images=imgs,
            assistant_texts=[correct_letter],
            correct_letter=correct_letter,
            letter_to_value={str(k): str(v) for k, v in letter_to_value.items()},
            base_user_text=base_user_text,
            system_text=system_text,
        )

    def _load(self, rel: str) -> Image.Image:
        img = Image.open(self.image_root / rel).convert("RGB")
        if self.image_size:
            img = img.resize(self.image_size, Image.Resampling.LANCZOS)
        return img


# ── Checker ───────────────────────────────────────────────────────────────


class MCQChecker(BaseChecker):
    """Validates MCQ-format replies and reports whether the chosen letter
    matches the expected one.

    Expected reply shape::

        <observation>...</observation>
        free-form reasoning prose
        <action>X</action>

    where X is a single capital letter (e.g. ``A`` or ``B``). The checker
    reports per-turn:

      * ``ok=True``  — format clean AND chosen letter == expected
      * ``ok=False`` with ``error="format"``  — format malformed
      * ``ok=False`` with ``error="wrong_letter"`` — format clean but
        wrong choice; reply text is preserved on the TurnCheck so the
        env can decide whether to advance to attempt 2 or salvage.

    ``augmented`` on a successful TurnCheck holds the body with the
    chosen ``<action>X</action>`` REWRITTEN to use the actual value
    (e.g. ``<action>turn_left</action>``) — that's what gets stored as
    the SFT augmented body so the SFT model never sees the letters.
    """

    def __init__(self, max_turn_chars: int = 1500):
        self.max_turn_chars = int(max_turn_chars)

    def check(self, reply: str, expected_assistant_texts: List[str]) -> List[TurnCheck]:
        # We only check the single expected letter — explode flattens to one.
        if len(expected_assistant_texts) != 1:
            return [TurnCheck(1, False, f"unexpected n_expected={len(expected_assistant_texts)}", None)]
        expected_letter = expected_assistant_texts[0].strip()
        return [self._check_one(reply, expected_letter)]

    def _check_one(self, reply: str, expected_letter: str) -> TurnCheck:
        if not reply or len(reply) > self.max_turn_chars:
            return TurnCheck(1, False,
                             f"too long ({len(reply or '')} > {self.max_turn_chars})", None)
        obs = _OBS_RE.search(reply)
        act = _ACTION_RE.search(reply)
        if obs is None:
            return TurnCheck(1, False, "missing <observation>", None)
        if act is None:
            return TurnCheck(1, False, "missing <action>", None)
        chosen = act.group(1).strip()
        if not (len(chosen) == 1 and "A" <= chosen <= "Z"):
            return TurnCheck(1, False,
                             f"<action> must be a single capital letter, got {chosen!r}", None)
        if chosen != expected_letter:
            # Format clean, wrong letter — env decides next step.
            tc = TurnCheck(1, False, "wrong_letter", None)
            tc.chosen_letter = chosen  # type: ignore[attr-defined]
            return tc
        # Success: keep body as-is; the env will rewrite the <action>
        # letter into the real value before storing.
        return TurnCheck(1, True, "", reply.strip())


# ── Env (state machine: 2 attempts × up to 2 format-fix turns each) ──────


class MCQReasoningEnv(GymImageEnv):
    """Per-record MCQ annotation env with simple retry-on-failure logic.

    Conversation layout (vagen builds it incrementally):

      * system role  = original game system prompt (from dp.messages)
      * user role 1  = full conversation transcript (first user obs +
        all images + GT actions taken so far rendered inline) PLUS the
        MCQ question + format spec at the end
      * assistant 1  = model's answer
      * user role 2  = "wrong, try again" / "format error, retry" feedback
        (only if model's first answer failed)
      * assistant 2  = model's retry
      * ... up to ``max_attempts`` attempts total

    On final failure we attempt one salvage parse: if the last reply
    contains ``<action>X</action>`` with the correct letter (i.e. answer
    right but format somewhere off), we treat the prefix as reasoning
    and accept it. Otherwise the per-turn reassembly salvage falls back
    to the original ``<action>...</action>`` content with no reasoning.
    """

    _PKG = "graphrl.traj_to_sft.self_reasoning"

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__(env_config)
        # Annotator system prompt (mcq_system.md) replaces the dp's original
        # game system — see system_prompt() comment for why. The dp's system
        # is implicitly available in dp.messages for any future need.
        self._annotator_sys_text = Path(env_config["system_prompt_path"]).read_text(encoding="utf-8")
        self._format_spec_text = Path(env_config["user_prompt_path"]).read_text(encoding="utf-8")
        # Reduced-MCQ feedback prompt (only the correct option shown) for
        # the wrong-letter escalation path. Defaults to a sibling file if
        # caller didn't override.
        reduced_path = env_config.get("reduced_user_prompt_path") or str(
            Path(env_config["user_prompt_path"]).with_name("mcq_user_reduced.md")
        )
        try:
            self._reduced_text = Path(reduced_path).read_text(encoding="utf-8")
        except FileNotFoundError:
            self._reduced_text = (
                "Your previous answer was wrong. Only the correct option "
                "remains; pick it and write reasoning."
            )

        from .base import import_by_path
        ds_cls = import_by_path(env_config.get("dataset_cls", f"{self._PKG}.mcq_reasoning.MCQExplodedDataset"))
        ds_kwargs = dict(env_config.get("dataset_kwargs", {}))
        ds_kwargs.setdefault("sft_path", env_config["sft_path"])
        ds_kwargs.setdefault("image_root", env_config.get("image_root"))
        ds_kwargs.setdefault("image_size", env_config.get("image_size"))
        self._dataset: BaseDataset = ds_cls(**ds_kwargs)

        ck_cls = import_by_path(env_config.get("checker_cls", f"{self._PKG}.mcq_reasoning.MCQChecker"))
        self._checker: BaseChecker = ck_cls(**(env_config.get("checker_kwargs") or {}))

        self._dp: Optional[MCQDatapoint] = None
        self._attempt: int = 0
        self._last_reply: str = ""
        self._max_attempts: int = int(env_config.get("max_attempts", 3))

    async def close(self) -> None:
        self._dp = None

    async def system_prompt(self) -> Dict[str, Any]:
        # Use a CUSTOM annotator system prompt (mcq_system.md) — not the dp's
        # original game system. The original system has rules like "Do NOT
        # output any text outside the expected tags" that conflict with our
        # required <observation>...prose...<action>X</action> format and
        # cause the annotator to skip the <observation> tag entirely. The
        # annotator system explicitly OVERRIDES those rules and re-states the
        # action vocabulary so the model still has navigation context.
        return {"obs_str": self._annotator_sys_text}

    async def reset(self, seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        idx = seed % len(self._dataset)
        dp = self._dataset.get(idx)
        assert isinstance(dp, MCQDatapoint), \
            f"MCQReasoningEnv requires MCQExplodedDataset — got {type(dp).__name__}"
        self._dp = dp
        self._attempt = 1
        self._last_reply = ""

        obs_str = self._render_initial_user(dp)
        obs: Dict[str, Any] = {"obs_str": obs_str}
        if dp.images:
            obs["multi_modal_input"] = {"<image>": list(dp.images)}
        info = {
            "sft_idx": dp.idx,
            "n_images": len(dp.images),
            "correct_letter": dp.correct_letter,
            "success": False,
        }
        return obs, info

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        assert self._dp is not None, "step() called before reset()"
        self._last_reply = action_str or ""
        result = self._checker.check(action_str, [self._dp.correct_letter])[0]
        per_turn = [{"attempt": self._attempt, "ok": result.ok, "error": result.error}]
        max_attempts = int(self._max_turns())

        if result.ok:
            return self._success(result.augmented or "", per_turn)

        # Failure — out of attempts: try a final salvage.
        if self._attempt >= max_attempts:
            salvage = self._salvage(self._last_reply)
            if salvage is not None:
                return self._success(salvage, per_turn, salvaged=True)
            return {"obs_str": "Out of attempts."}, 0.0, True, {
                "attempt": self._attempt,
                "per_turn": per_turn,
                "sft_idx": self._dp.idx,
                "success": False,
            }

        # Failure with attempts remaining. Two distinct paths:
        #  * wrong_letter → ESCALATE to reduced-MCQ (only correct option shown).
        #    The model can't pick wrong if there's only one option, so the
        #    next attempt is essentially guaranteed to succeed (modulo
        #    format errors). Massively reduces the wrong-letter failure rate.
        #  * format error → keep the same MCQ but feed back what was wrong.
        self._attempt += 1
        if result.error == "wrong_letter":
            # Re-render the question with only the correct option, append
            # the reduced-prompt feedback. The model now sees a 1-choice
            # question and writes reasoning for the ONLY remaining option.
            reduced_q = _format_mcq_question(
                {self._dp.correct_letter: self._dp.letter_to_value[self._dp.correct_letter]}
            )
            feedback = self._reduced_text.strip() + "\n" + reduced_q
        else:
            feedback = (
                f"Format error: {result.error}. Re-output your reply: open with "
                "<observation>, write prose reasoning, then close with "
                "<action>X</action> where X is the LETTER (A or B)."
            )
        return {"obs_str": feedback}, 0.0, False, {
            "attempt": self._attempt,
            "per_turn": per_turn,
            "sft_idx": self._dp.idx,
            "success": False,
        }

    # ── helpers ───────────────────────────────────────────────────────────

    def _max_turns(self) -> int:
        return self._max_attempts

    def _success(self, body: str, per_turn, salvaged: bool = False):
        real_value = self._dp.letter_to_value.get(self._dp.correct_letter, "")
        body_with_real = _ACTION_RE.sub(
            lambda _m: f"<action>{real_value}</action>", body, count=1,
        )
        info = {
            "attempt": self._attempt,
            "augmented": [body_with_real],
            "sft_idx": self._dp.idx,
            "per_turn": per_turn,
            "salvaged": salvaged,
            "success": True,
        }
        return {"obs_str": "Correct."}, 1.0, True, info

    def _salvage(self, reply: str) -> Optional[str]:
        """Final-attempt salvage: if the reply contains ``<action>X</action>``
        with EITHER the correct letter OR the correct letter's mapped action
        name (the model often outputs the action name despite the LETTER
        instruction), accept the rest as reasoning even if ``<observation>``
        is missing or the format is otherwise loose. Returns the body to use,
        or None if not salvageable.

        Also requires that the reply has SOME content beyond just the
        ``<action>`` tag — otherwise we'd be storing reasoning-free fallback
        bodies, which the per-turn reassembly salvage already handles
        more cleanly with the original GT action.
        """
        if not reply or not self._dp:
            return None
        m = _ACTION_RE.search(reply)
        if not m:
            return None
        chosen = m.group(1).strip()
        correct_letter = self._dp.correct_letter
        correct_action = self._dp.letter_to_value.get(correct_letter, "")
        if chosen != correct_letter and chosen != correct_action:
            return None
        # Reject empty-prose salvage (just ``<action>X</action>`` with no
        # reasoning before it) — the standard reassembly fallback gives a
        # cleaner result in that case.
        body_no_act = _ACTION_RE.sub("", reply).strip()
        if not body_no_act:
            return None
        # Right answer, sloppy format. Use the reply verbatim — the
        # downstream letter→action substitution rewrites whatever's inside
        # <action>...</action> to the canonical action name.
        return reply.strip()

    def _render_initial_user(self, dp: "MCQDatapoint") -> str:
        """Render the conversation history as a clean transcript and
        append the MCQ question + format spec.

        Layout::

            {first user message — initial obs with images}

            [Action taken: turn_left]

            {next user message if its images survived single_turn pruning}

            [Action taken: look_up]

            ...

            {last user message — current obs with image}

            === Question ===
            Pick the better next action.
            A. ...
            B. ...

            {format spec from mcq_user.md}

        We do NOT put any other framing (`=== TASK SETUP ===` etc) — the
        model just sees natural alternating user/assistant content.
        Image placeholder ordering is preserved verbatim from dp.messages
        so vagen's reading-order substitution stays aligned with dp.images.
        """
        question_block = _format_mcq_question(dict(dp.letter_to_value))

        # Walk dp.messages and emit user content + GT-action breadcrumbs.
        # Skip system (handled by system_prompt()).
        parts: List[str] = []
        last_user_idx: Optional[int] = None
        user_contents: List[str] = []
        prev_actions: List[str] = []
        for m in dp.messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                user_contents.append(content)
            elif role == "assistant":
                act = _extract_action_value(content) or ""
                if act:
                    prev_actions.append(act)

        # Build the transcript: alternate user_contents[i] and prev_actions[i]
        # (action breadcrumbs), with the LAST user message getting the MCQ
        # question appended. prev_actions has one fewer entry than
        # user_contents because the trailing assistant was dropped during
        # explode (the model is being asked to fill it in).
        n_users = len(user_contents)
        for i, uc in enumerate(user_contents):
            is_last = (i == n_users - 1)
            if is_last:
                # Replace the embedded question with a fresh render (defensive
                # against re-runs where dp may already carry the embed).
                base = dp.base_user_text or uc
                parts.append(base + question_block)
            else:
                parts.append(uc)
            if i < len(prev_actions):
                parts.append(f"[Action taken: {prev_actions[i]}]")

        parts.append(self._format_spec_text.strip())
        return "\n\n".join(parts)


# ── Reasoner (drives the explode + multi-process vagen rollout) ──────────


class MCQReasoner(SingleTurnReasoner):
    """Single-turn explode + reassemble, but each per-turn job runs the
    MCQ retry state machine instead of the freeform-wrap task.

    Inherits all the explode / vagen-launch / multi-process / reassemble
    logic from :class:`SingleTurnReasoner`. Only differences:

      * Default prompts point to ``mcq_*.md`` instead of ``single_turn_*.md``.
      * Default ``env_name`` → distinct registry entry so it doesn't
        collide with single_turn's env.
      * Default ``max_turns`` = 4 (state-machine needs the full budget).
      * The exploded record builder enriches each per-turn sub-record
        with MCQ choice metadata (``_correct_letter`` /
        ``_letter_to_value`` / ``_base_user_text``) and skips turns
        whose action has no plausible distractor (those fall back to
        the original ``<action>...</action>`` content via reassembly
        salvage — same as single_turn's salvage path).
    """

    name = "MCQReasoner"

    # ── overrides ─────────────────────────────────────────────────────────

    def system_prompt_path(self) -> Path:
        p = self.config.get("system_prompt_path")
        return Path(p) if p else DEFAULT_PROMPTS_DIR / "mcq_system.md"

    def user_prompt_path(self) -> Path:
        p = self.config.get("user_prompt_path")
        return Path(p) if p else DEFAULT_PROMPTS_DIR / "mcq_user.md"

    def env_name(self) -> str:
        return self.config.get("env_name", "GraphRLMCQReasoningEnv")

    # ── per-target step (overrides single_turn's to inject MCQ env + dataset) ──

    def _augment_target(self, target, base_url, model, sys_p, usr_p) -> None:
        snapshot = self.snapshot_path(target)
        if not snapshot.exists():
            logger.warning(
                "[%s] no snapshot for %s at %s; skipping",
                self.parent_name, target, snapshot,
            )
            return

        with open(snapshot, encoding="utf-8") as f:
            originals = json.load(f)

        n_records_cap = self.n_records_for(target)
        if n_records_cap is not None:
            n_records_cap = min(int(n_records_cap), len(originals))
            sliced = originals[:n_records_cap]
        else:
            sliced = originals
        if not sliced:
            logger.warning("[%s] %s: no records to augment", self.parent_name, target)
            return

        recent_k = self.recent_k()
        exploded_path = self.exploded_path(target)
        mapping_path = self.mapping_path(target)
        exploded_path.parent.mkdir(parents=True, exist_ok=True)

        # Resume-stable explode.
        if exploded_path.exists() and mapping_path.exists():
            with open(mapping_path, encoding="utf-8") as f:
                mapping = json.load(f)
            with open(exploded_path, encoding="utf-8") as f:
                exploded = json.load(f)
            logger.info(
                "[%s] %s: reusing existing MCQ-exploded view (%d jobs from %d records)",
                self.parent_name, target, len(exploded), len(sliced),
            )
        else:
            exploded, mapping = self._explode_records_with_mcq(
                sliced, target_dataset=target, recent_k=recent_k,
                seed=int(self.config.get("seed", 1729)),
            )
            if not exploded:
                logger.warning(
                    "[%s] %s: MCQ explode produced no jobs; skipping",
                    self.parent_name, target,
                )
                return
            with open(exploded_path, "w", encoding="utf-8") as f:
                json.dump(exploded, f, ensure_ascii=False)
            with open(mapping_path, "w", encoding="utf-8") as f:
                json.dump(mapping, f, ensure_ascii=False)
            logger.info(
                "[%s] %s: MCQ-exploded %d records → %d jobs (recent_k=%d)",
                self.parent_name, target, len(sliced), len(exploded), recent_k,
            )

        # max_turns = max_attempts in the new flow (each attempt = 1 vagen
        # turn — model gets one MCQ retry per turn, with feedback in between).
        max_attempts = int(self.config.get("max_turns", 3))

        per_seed = run_vagen_eval_and_collect(
            sft_path=exploded_path,
            image_root=self.image_root(target),
            dump_dir=self.dump_dir(target),
            tag_id=target,
            base_url=base_url,
            model_name=model,
            system_prompt_path=sys_p,
            user_prompt_path=usr_p,
            image_size=self.image_size(),
            max_turns=max_attempts,
            max_concurrent_jobs=int(self.config.get("max_concurrent_jobs", 16)),
            max_retries=int(self.config.get("max_retries", 6)),
            chat_config=self.chat_config(),
            salvage_partial=bool(self.config.get("salvage_partial", True)),
            n_records=None,
            resume=bool(self.config.get("resume", True)),
            env_name=self.env_name(),
            checker_cls=f"{__name__}.MCQChecker",
            checker_kwargs=None,
            dataset_cls=f"{__name__}.MCQExplodedDataset",
            dataset_kwargs={},
            extra_env_config={"max_attempts": max_attempts},
            num_workers=self.num_workers(),
            # CRITICAL: register MCQReasoningEnv (not the default
            # ReasoningEnv) under the env_name we declared. Without this,
            # vagen instantiates ReasoningEnv with our MCQ dataset/checker
            # — but ReasoningEnv.step() doesn't do the letter→action
            # substitution, so the augmented body ends up with the raw
            # ``<action>A</action>`` letter instead of
            # ``<action>{actual_action}</action>``.
            env_cls=MCQReasoningEnv,
        )

        # Re-use single_turn's reassembly via the parent module's helper.
        from .single_turn import _reassemble
        out_records, (n_full, n_partial, n_dropped) = _reassemble(
            originals=originals,
            mapping=mapping,
            per_seed=per_seed,
            keep_unaugmented=bool(self.config.get("keep_unaugmented", False)),
            augmented_system_prompt_suffix=self.config.get("augmented_system_prompt_suffix"),
            raw_system_prompt_suffix=self.config.get("raw_system_prompt_suffix"),
        )

        out_path = self.output_path(target)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_records, f, ensure_ascii=False, indent=2)
        logger.info(
            "[%s] %s: wrote %d records → %s (full=%d partial=%d dropped=%d, jobs=%d)",
            self.parent_name, target, len(out_records), out_path,
            n_full, n_partial, n_dropped, len(mapping),
        )

    # ── MCQ-aware explode ─────────────────────────────────────────────────

    @staticmethod
    def _explode_records_with_mcq(
        records: List[Dict[str, Any]],
        target_dataset: str,
        recent_k: int,
        seed: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, int]]]:
        """Like ``_explode_records`` but, for each per-turn sub-record,
        attaches MCQ choice metadata (correct letter, letter-to-value map,
        base user text). Turns that don't have a plausible distractor are
        SKIPPED — their reassembled output will fall back to the original
        ``<action>X</action>`` content through the standard salvage path.
        """
        rng = random.Random(seed)
        exploded: List[Dict[str, Any]] = []
        mapping: List[Dict[str, int]] = []
        for rec_i, rec in enumerate(records):
            messages = rec.get("messages") or []
            image_paths = list(rec.get("images") or [])
            n_assistant = sum(1 for m in messages if m.get("role") == "assistant")
            if n_assistant == 0:
                continue
            for turn in range(1, n_assistant + 1):
                # Standard explode produces messages truncated to user_i.
                sub = _build_subrecord(messages, image_paths, turn, recent_k)
                if sub is None:
                    continue
                target_action = sub.get("_target_action") or ""
                gt_inner = _extract_action_value(target_action) or target_action
                # Find the last user message to use both as ``base_user_text``
                # AND as the source for view_difference_mcq option parsing.
                base_user_text = ""
                for m in reversed(sub["messages"]):
                    if m["role"] == "user":
                        base_user_text = m["content"]
                        break
                choices = _build_mcq_choices(target_dataset, gt_inner, base_user_text, rng)
                if choices is None:
                    # Skip MCQ for this turn — original action will be
                    # used as-is via reassembly salvage.
                    continue
                gt_value, neg_value = choices
                # Random shuffle: which letter holds the GT?
                if rng.random() < 0.5:
                    correct_letter = "A"
                    letter_to_value = {"A": gt_value, "B": neg_value}
                else:
                    correct_letter = "B"
                    letter_to_value = {"A": neg_value, "B": gt_value}
                # Embed the MCQ question into the last user message so the
                # exploded JSON is self-contained for vagen / dataset reader.
                question_block = _format_mcq_question(letter_to_value)
                new_messages = [dict(m) for m in sub["messages"]]
                for idx in range(len(new_messages) - 1, -1, -1):
                    if new_messages[idx]["role"] == "user":
                        new_messages[idx]["content"] = base_user_text + question_block
                        break
                # CRITICAL: drop the trailing assistant message entirely.
                # _build_subrecord appended the GT action there as the
                # "to be wrapped with reasoning" target — but for MCQ the
                # model must NOT see the answer; it has to deduce the
                # correct letter from the images alone. The dataset reader
                # (MCQExplodedDataset.get) gets the expected letter from
                # the explicit ``_correct_letter`` field, so removing the
                # trailing assistant doesn't break the checker.
                if new_messages and new_messages[-1]["role"] == "assistant":
                    new_messages.pop()
                exploded.append({
                    "messages": new_messages,
                    "images": sub["images"],
                    "_target_action": correct_letter,    # what the checker validates
                    "_correct_letter": correct_letter,
                    "_letter_to_value": letter_to_value,
                    "_base_user_text": base_user_text,
                    "_actual_gt_value": gt_value,        # for debugging / future use
                })
                mapping.append({
                    "record_idx": rec_i,
                    "turn_idx": turn,
                    "n_turns": n_assistant,
                })
        return exploded, mapping


__all__ = [
    "MCQChecker",
    "MCQDatapoint",
    "MCQExplodedDataset",
    "MCQReasoner",
    "MCQReasoningEnv",
]
