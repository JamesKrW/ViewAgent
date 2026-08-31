"""Compatibility subprocess entry point for ViewSuite's SLIME evaluator.

It keeps the historical module path used by the random-SFT pipeline while
delegating environment registration, harness execution and recording to the
shared ViewSuite/VAGEN-SLIME evaluator. ``argv`` passes through unchanged.
"""
from __future__ import annotations

if __name__ == "__main__":
    from view_suite.evaluation.run_eval import main

    main()
