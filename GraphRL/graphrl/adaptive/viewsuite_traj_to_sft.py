"""Strict ViewSuite trajectory conversion used only by adaptive pipelines."""

from __future__ import annotations

import json
import os
from pathlib import Path

from graphrl.envs.viewsuite.viewsuite_interactive_view_planning import (
    InteractiveViewPlanningTrajToSFT,
)
from graphrl.envs.viewsuite.viewsuite_interactive_view_planning import (
    interactive_view_planning_graph_builder as graph_builder_module,
)
from graphrl.envs.viewsuite.viewsuite_interactive_view_planning.utils import (
    graph_atomize,
)

ATOMIZE_STATE_FILENAME = "adaptive_atomize_state.json"


class StrictAdaptiveInteractiveViewPlanningGraphBuilder(
    graph_builder_module.InteractiveViewPlanningGraphBuilder
):
    """Reject any adaptive SFT graph with failed atomization renders."""

    @staticmethod
    def _require_complete_atomization(stats: dict) -> None:
        dropped = int(stats.get("dropped", 0))
        leftover = int(stats.get("leftover_removed", 0))
        multi_edges = int(stats.get("multi_edges", 0))
        rendered = int(stats.get("rendered", 0))
        if dropped or leftover or (multi_edges and not rendered):
            raise RuntimeError(
                "adaptive atomize must render every intermediate view: "
                f"multi_edges={multi_edges}, rendered={rendered}, "
                f"dropped={dropped}, leftover_removed={leftover}"
            )

    def convert_files(
        self,
        files: list[Path],
        rollout_dir: Path,
        graph_dir: Path,
    ) -> None:
        original = graph_atomize.atomize_graph
        atomize_stats: dict = {}

        def strict_atomize(*args, **kwargs):
            stats = original(*args, **kwargs)
            self._require_complete_atomization(stats)
            atomize_stats.update(stats)
            return stats

        # The legacy builder imports atomize_graph inside convert_files. Patch
        # that module symbol only for this synchronous adaptive conversion, then
        # restore it even when strict validation raises.
        graph_atomize.atomize_graph = strict_atomize
        try:
            super().convert_files(files, rollout_dir, graph_dir)
        finally:
            graph_atomize.atomize_graph = original

        if (self.config.get("atomize") or {}).get("enabled"):
            marker = Path(graph_dir) / ATOMIZE_STATE_FILENAME
            temporary = marker.with_name(f".{marker.name}.tmp")
            temporary.write_text(
                json.dumps(atomize_stats, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, marker)


class AdaptiveInteractiveViewPlanningTrajToSFT(
    InteractiveViewPlanningTrajToSFT
):
    """Adaptive-only TrajToSFT with fail-closed atomization."""

    def graph_builder_class(self):
        return StrictAdaptiveInteractiveViewPlanningGraphBuilder
