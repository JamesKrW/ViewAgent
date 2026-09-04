"""Launch VAGEN's trainer with the adaptive schedule swapped in.

Replaces the backend's old ``vagen/main_ppo_adaptive.py``. That file existed only
to select a trainer class, and lived in the submodule purely because it imported
one from there -- so a backend swap deleted the experiment's entry point. It is
ViewAgent's choice of trainer, so it belongs to ViewAgent.

Run exactly like ``vagen.training.main``; the command builder appends the same
Hydra arguments:

    python -m graphrl.adaptive.entrypoint --config-path=... --config-name=...

Reachable because ``build_vagen_env`` puts the repo root on the subprocess
PYTHONPATH alongside the VAGEN and verl checkouts.
"""

from __future__ import annotations

import hydra
import ray

from vagen.training.main import TaskRunner, run_ppo


class AdaptiveTaskRunner(TaskRunner):
    """Reuse VAGEN's setup while selecting the adaptive trainer class."""

    def run(self, config):
        # TaskRunner.run resolves the trainer from vagen.training.main's module
        # globals, so rebinding the name there is what selects it. Done inside the
        # Ray worker process and restored afterwards, so nothing else in the
        # cluster sees a patched module.
        from vagen.training import main as vagen_main

        from graphrl.adaptive.trainer import AdaptivePPOTrainer

        original = vagen_main.VagenPPOTrainer
        vagen_main.VagenPPOTrainer = AdaptivePPOTrainer
        try:
            return super().run(config)
        finally:
            vagen_main.VagenPPOTrainer = original


# Same Hydra entry contract as vagen.training.main: config_path/config_name are
# supplied on the command line, and these defaults are never the ones used.
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(AdaptiveTaskRunner))


if __name__ == "__main__":
    main()
