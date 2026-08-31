# Habitat-GS IVP on synchronous SLIME

This ViewAgent-owned entry point uses the vendored ``GraphRL/VAGEN``
(VAGEN-SLIME) backend. The environment implementation and its thin Gymnasium
adapter remain under ``view_suite``; SLIME owns rollout, PPO, model weight
synchronization, checkpointing, and resume.

The migration launcher currently uses synchronous `train.py` deliberately. Async overlap
will remain disabled until step-0 eval, checkpoint boundaries, and exact resume have all
passed the local smoke sequence.

The launcher enables ViewAgent's render-only TLS bypass because the current internal
endpoint uses a trusted self-signed certificate. Pass `--no-insecure-render-tls` when the
service has a certificate signed by the machine's CA bundle.

The first check is rollout-only and evaluates two held-out episodes before any training:

```bash
python GraphRL/examples/viewsuite/habitat_gs_interactive_view_planning/slime/run_habitat_gs.py \
  --rollout-only \
  --eval-yaml GraphRL/examples/viewsuite/habitat_gs_interactive_view_planning/slime/verify.yaml \
  --exp-name habitat_gs_ivp_slime_verify
```

Once that dump passes `tests/verify_rollout_dump.py`, remove `--rollout-only` for a short
training run. Checkpoints are written every 20 rollout steps; actor and critic use separate
directories and `--resume` requires both to be present.

The checkpoint/resume smoke is deliberately synchronous and uses one rollout update per
launch:

```bash
python GraphRL/examples/viewsuite/habitat_gs_interactive_view_planning/slime/run_habitat_gs.py \
  --num-rollout 1 --rollout-batch-size 4 \
  --eval-yaml GraphRL/examples/viewsuite/habitat_gs_interactive_view_planning/slime/verify.yaml \
  --eval-interval 1 --save-interval 1 \
  --exp-name habitat_gs_ivp_slime_sync_smoke

python GraphRL/examples/viewsuite/habitat_gs_interactive_view_planning/slime/run_habitat_gs.py \
  --resume --num-rollout 2 --rollout-batch-size 4 \
  --eval-yaml GraphRL/examples/viewsuite/habitat_gs_interactive_view_planning/slime/verify.yaml \
  --eval-interval 1 --save-interval 1 \
  --exp-name habitat_gs_ivp_slime_sync_smoke
```

The first command must evaluate at step 0, train rollout 0, and write both actor and critic
checkpoint trackers. The second must load rollout 0 and continue at rollout 1; starting
again at rollout 0 is a resume failure.
