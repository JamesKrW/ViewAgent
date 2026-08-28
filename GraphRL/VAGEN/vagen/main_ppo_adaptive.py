"""Isolated VAGEN entry point for the adaptive GraphRL controller."""

from __future__ import annotations

import hydra
import ray

from vagen.adaptive_ray_trainer import AdaptiveRayPPOTrainer
from vagen.main_ppo import TaskRunner, run_ppo


class AdaptiveTaskRunner(TaskRunner):
    """Reuse VAGEN setup while selecting the adaptive trainer class."""

    def run(self, config):
        # TaskRunner.run resolves RayPPOTrainer from vagen.main_ppo's module
        # globals. Replace it only inside this dedicated Ray worker process.
        from vagen import main_ppo

        legacy_trainer = main_ppo.RayPPOTrainer
        main_ppo.RayPPOTrainer = AdaptiveRayPPOTrainer
        try:
            return super().run(config)
        finally:
            main_ppo.RayPPOTrainer = legacy_trainer


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    task_runner = ray.remote(num_cpus=1)(AdaptiveTaskRunner)
    run_ppo(config, task_runner_class=task_runner)


if __name__ == "__main__":
    main()
