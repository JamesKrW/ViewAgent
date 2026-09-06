"""Shared formatting for the three ViewSuite navigation action spaces.

The environments intentionally keep their coordinate-system details local, but the
prompt structure should stay identical.  In particular, movement semantics, step
sizes, and rotation quantization belong in one action-space section rather than in
three partly overlapping sections.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

_DEFAULT_SIGNATURES = {
    "query_pose": "query_pose(view_name)",
    "select_view": "select_view(view_name)",
    "get_view": "get_view(tx, ty, tz, rx, ry, rz)",
    "answer": "answer(tx, ty, tz, rx, ry, rz)",
}


def build_action_space_instruction(
    *,
    is_discrete: bool,
    snap_rotations: bool,
    step_translation: str,
    step_rotation_deg: str,
    mode_description: str,
    coordinate_description: str,
    snap_description: str,
    actions: Sequence[str],
    action_descriptions: Mapping[str, str],
    action_only_mode: bool,
    motion_wildcards: str,
    translation_actions: str = "every move_* action",
    rotation_actions: str = "every rotation action",
    action_signatures: Mapping[str, str] | None = None,
) -> str:
    """Build a compact, mode-aware navigation-tool instruction.

    ``is_discrete`` controls whether the camera implementation may quantize its
    orientation.  The public actions remain fixed-step even when it is false, so the
    non-discrete label says exactly that instead of incorrectly promising continuous
    action arguments.
    """

    heading = "ACTION SPACE"

    if snap_rotations:
        snapping = f"enabled; {snap_description}"
    elif is_discrete:
        snapping = (
            "disabled; rotations are not rounded to the rotation-step grid "
            "(`is_snap_every_step=false`)."
        )
    else:
        snapping = (
            "disabled; rotations are not rounded to the rotation-step grid "
            "(`is_discrete=false`)."
        )

    lines = [
        heading,
        "-" * len(heading),
        f"- Mode semantics: {mode_description}",
        f"- Coordinates: {coordinate_description}",
        f"- Translation step: {translation_actions} translates {step_translation} meters.",
        f"- Rotation step: {rotation_actions} changes its angle by {step_rotation_deg} degrees.",
        f"- Rotation snapping: {snapping}",
        "",
        "Actions (arguments are inside parentheses):",
    ]
    signatures = dict(_DEFAULT_SIGNATURES)
    signatures.update(action_signatures or {})
    lines.extend(
        f"- {signatures.get(name, name)}: {action_descriptions[name]}"
        for name in actions
    )

    if not action_only_mode:
        lines.extend(
            [
                "",
                "ACTION ORDER CONSTRAINTS",
                "------------------------",
                "- Call exactly one of select_view(view_name) or "
                "get_view(tx, ty, tz, rx, ry, rz) before any " + motion_wildcards + ".",
                "- Calling a movement or rotation action before selecting a view is invalid.",
                "- query_pose(view_name) does not select a view.",
            ]
        )

    return "\n".join(lines)
