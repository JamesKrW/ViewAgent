from __future__ import annotations

from pathlib import Path

from view_suite.evaluation.config import load_legacy_config


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_submodule_backend_uses_unambiguous_vagen_slime_directory() -> None:
    backend = REPO_ROOT / "GraphRL" / "VAGEN-SLIME"
    assert (backend / "vagen_agent").is_dir()
    assert (backend / "slime").is_dir()
    assert not (REPO_ROOT / "GraphRL" / "VAGEN").exists()


def test_every_training_entrypoint_uses_vagen_slime() -> None:
    scripts = sorted((REPO_ROOT / "GraphRL" / "examples").rglob("run*.sh"))
    assert scripts
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert "_slime_env.sh" in text, script
        assert "vagen.main_ppo" not in text, script
        assert "verl.trainer" not in text, script
        assert (
            "graphrl.main" in text
            or "slime/run_habitat_gs.py" in text
        ), script
        if script.name == "run_adaptive.sh":
            assert "graphrl.main_adaptive" in text, script
        elif "slime/run_habitat_gs.py" not in text:
            assert "graphrl.main_adaptive" not in text, script


def test_every_evaluation_entrypoint_uses_vagen_slime() -> None:
    evaluation_root = REPO_ROOT / "examples" / "evaluation"
    helpers = {
        evaluation_root / "_slime_env.sh",
        evaluation_root / "eval_sglang" / "wait_for_server.sh",
    }
    scripts = sorted(set(evaluation_root.rglob("*.sh")) - helpers)
    assert scripts
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert "_slime_env.sh" in text, script
        assert "view_suite.evaluation.run_eval" in text, script
        assert "vagen.evaluate" not in text, script


def test_every_evaluation_yaml_translates_to_vagen_slime(monkeypatch) -> None:
    monkeypatch.setenv("VIEWSUITE_ROOT", str(REPO_ROOT))
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")

    configs = sorted((REPO_ROOT / "examples" / "evaluation").rglob("*.yaml"))
    assert configs
    for path in configs:
        translated = load_legacy_config(path, [f"fileroot={REPO_ROOT}"])
        assert translated.models, path
        assert translated.environments, path
        assert all(
            environment.harness in {"concat", "no_concat"}
            for environment in translated.environments
        ), path
