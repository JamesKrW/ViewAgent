"""Run established ViewSuite eval configs through VAGEN-SLIME's evaluator."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from vagen_agent.evaluation.recording import json_safe
from vagen_agent.evaluation.runner import run_evaluation

from view_suite.envs.slime_adapter import register_viewsuite_envs
from view_suite.evaluation.config import load_legacy_config


def _legacy_content(
    message: dict[str, Any],
    *,
    source_root: Path,
    target_root: Path,
    turn_index: int,
) -> tuple[list[dict[str, Any]], int]:
    """Convert a persisted SLIME message to the historical evaluator schema."""

    descriptors = iter(message.get("images") or [])
    output: list[dict[str, Any]] = []
    image_index = 0
    raw = message.get("content", "")
    parts = raw if isinstance(raw, list) else [{"type": "text", "text": str(raw)}]
    for part in parts:
        if not isinstance(part, dict) or part.get("type") == "text":
            text = str(part.get("text", "")) if isinstance(part, dict) else str(part)
            output.append({"type": "text", "text": text})
            continue
        if part.get("type") != "image":
            continue
        descriptor = next(descriptors, None)
        if not isinstance(descriptor, dict) or not descriptor.get("path"):
            continue
        image_index += 1
        filename = f"turn_{turn_index:02d}_{image_index:02d}.png"
        source = source_root / str(descriptor["path"])
        destination = target_root / "images" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_file() and not destination.exists():
            shutil.copy2(source, destination)
        output.append({"type": "image_url", "image_url": {"url": f"images/{filename}"}})
    return output, image_index


def _export_legacy_layout(config) -> None:
    """Materialize the old ``tag_*/seed`` view for existing analysis tools.

    VAGEN-SLIME remains the authoritative store.  This is a compatibility
    projection, not a second evaluator, and can be regenerated after resume.
    """

    root = config.output_dir / config.experiment_id
    for model in config.models:
        model_root = root / model.name
        if not model_root.is_dir():
            continue
        for result_path in model_root.glob("tag_*/seed_*/result.json"):
            result = json.loads(result_path.read_text(encoding="utf-8"))
            source_root = result_path.parent
            tag = str(result["tag"])
            seed = int(result["seed"])
            target = root / f"tag_{tag}" / str(seed)
            target.mkdir(parents=True, exist_ok=True)

            messages: list[dict[str, Any]] = []
            system = result.get("system_prompt")
            if isinstance(system, dict):
                content, _ = _legacy_content(
                    system, source_root=source_root, target_root=target, turn_index=0
                )
                messages.append({"role": "system", "content": content})

            infos = []
            for turn, call in enumerate(result.get("trajectory") or [], start=1):
                candidates = [
                    item for item in call.get("messages") or []
                    if isinstance(item, dict) and item.get("role") == "user"
                ]
                if candidates:
                    content, _ = _legacy_content(
                        candidates[-1],
                        source_root=source_root,
                        target_root=target,
                        turn_index=turn,
                    )
                    messages.append({"role": "user", "content": content})
                response = call.get("response") or {}
                messages.append(
                    {"role": "assistant", "content": str(response.get("content") or "")}
                )
                transition = call.get("transition") or {}
                if isinstance(transition.get("info"), dict):
                    infos.append(transition["info"])

            (target / "messages.json").write_text(
                json.dumps(messages, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            metrics = {
                **dict(result.get("metrics") or {}),
                "env_name": result.get("environment"),
                "tag_id": tag,
                "seed": seed,
                "success": bool(result.get("success")),
                "terminated": result.get("finish_reason") == "done",
                "finish_reason": result.get("finish_reason"),
                "reward": result.get("return"),
                "infos": infos,
            }
            (target / "metrics.json").write_text(
                json.dumps(json_safe(metrics), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            (target / "meta.json").write_text(
                json.dumps(
                    {
                        "env_name": result.get("environment"),
                        "tag_id": tag,
                        "seed": seed,
                        "model": result.get("model"),
                        "source": str(source_root.relative_to(root)),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="translate and validate the config without contacting a model or environment",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides retained for existing shell scripts",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    register_viewsuite_envs()
    config = load_legacy_config(Path(args.config), list(args.overrides))
    print("=== Effective VAGEN-SLIME Evaluation Config ===")
    print(json.dumps(json_safe(config.public_config), ensure_ascii=False, indent=2))
    if args.validate_only:
        print(
            json.dumps(
                {
                    "models": [model.name for model in config.models],
                    "environments": [
                        {
                            "name": env.name,
                            "tag": env.tag,
                            "harness": env.harness,
                            "episodes": len(env.seeds),
                        }
                        for env in config.environments
                    ],
                    "total_jobs": sum(len(env.seeds) for env in config.environments)
                    * len(config.models),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    summary = asyncio.run(run_evaluation(config))
    _export_legacy_layout(config)
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))
    if any(value.get("errors") for value in summary["tasks"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
