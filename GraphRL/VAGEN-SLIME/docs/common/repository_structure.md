# Repository structure rules

This document records the structure shared by HIDAgent and VAGEN-Agent after
their 2026-08 refactors. It is a durable rule for new code, not a snapshot of
one branch.

## 1. Separate invariants from extension axes

An **invariant** is orchestration or protocol code whose shape should remain
stable across implementations. Keep invariants as ordinary modules with
specific names. Examples include rollout coordination, trajectory records,
evaluation runners, configuration loading, recording, and result viewing.

An **extension axis** is a family for which new implementations are expected.
Every extension axis uses this shape:

```text
axis/
├── __init__.py             # public facade and registry
├── _common/                # shared contracts and implementation helpers
│   ├── __init__.py
│   └── ...
├── implementation_a/
│   ├── __init__.py
│   └── implementation_a.py
└── implementation_b/
    ├── __init__.py
    └── implementation_b.py
```

The root `__init__.py` is the stable import boundary. `_common/` contains only
interfaces, value types, registry machinery, and helpers genuinely shared by
multiple implementations. Each concrete implementation owns a directory,
even when it currently consists of one file.

Do not create a plugin-shaped directory for code that is not actually an
extension axis. Do not use ambiguous top-level names such as `adapter/` or
`common.py` when the owning subsystem can be named explicitly.

## 2. Dependency direction

Dependencies must flow in one direction:

```text
orchestration/core -> public axis facade -> concrete implementation
                                      \-> _common contracts
concrete implementation ----------------> _common contracts
```

- `_common` must not import a concrete implementation.
- Core orchestration should select implementations through a registry or a
  declared import path, not through hard-coded implementation imports.
- Concrete implementations may depend on their axis's `_common` package and
  on stable public APIs from other axes.
- Cross-repository consumers must import the public facade. Imports into
  another repository's `_common` or concrete implementation are private and
  may break during refactors.
- Compatibility aliases belong in the public facade and should have an
  explicit removal plan; do not preserve an obsolete directory tree merely to
  keep an internal import working.

## 3. VAGEN-Agent layout

The main VAGEN-Agent extension axes are:

```text
vagen_agent/
├── algorithms/
│   ├── _common/            # AlgorithmSpec, reward helpers, shared contracts
│   └── <algorithm>/        # default_gae, trajectory_grpo, ...
├── envs/
│   ├── _common/            # BaseEnv protocol, BaseVagenEnv implementation, capability markers
│   └── <environment>/      # sokoban, HIDAgent bridge, ...
├── harness/
│   ├── _common/            # BaseHarness and shared harness helpers
│   └── <harness>/          # compact, concat, no_concat, ...
├── models/
│   ├── _common/            # model contracts and image-token helpers
│   └── <model_family>/     # qwen, ...
├── evaluation/
│   ├── backends/
│   │   ├── _common/        # backend contract and registry
│   │   └── <backend>/      # openai, ...
│   ├── config.py
│   ├── recording.py
│   ├── runner.py
│   └── viewer.py
└── rollout/                # fixed rollout pipeline, not an extension axis
    ├── client.py
    ├── frames.py
    ├── rendering.py
    ├── runner.py
    ├── scoring.py
    └── trajectory.py
```

The facade for each axis owns name-to-implementation registration. Config
files should use registered names where possible. Dynamic Python paths remain
supported only where they are part of an intentional public contract.
Algorithm-specific behavior, including reward folding, belongs in the
algorithm specification rather than in the shared rollout runner.

Project-specific integrations belong in the downstream project and are selected through
the supported Python import-path configuration. For example, a real-time Harness for
HIDAgent belongs in HIDAgent's own `harnesses/realtime/`; VAGEN keeps only generic
Harness contracts, registries, client routing, and its built-in synchronous policies.

## 4. HIDAgent layout

HIDAgent follows the same rule at its own extension points:

```text
hidagent/
├── datasets/
│   ├── _common/
│   └── <dataset>/
├── envs/
│   ├── _common/            # environment contracts and remote protocol
│   ├── commons/utils/      # current shared VLM/action utility boundary
│   └── <environment>/
└── harnesses/
    ├── _common/
    └── <harness>/
```

HIDAgent owns environment integrations, action protocols, dataset conversion,
and benchmark-specific recipes. VAGEN-Agent owns the reusable training rollout
and evaluation orchestration. Do not reintroduce a second evaluation runner
under `hidagent/eval`.

Repository-level operational content is separated by purpose:

```text
examples/<phase>/<experiment>/   # framework configs and local launchers
cluster/<phase>/<experiment>/    # cluster-only submission and run-node logic
docs/common/                     # durable documentation
docs/dates/YY-MM-DD/             # dated reports and incidents
exps/<phase>/<experiment>/<run>/ # ignored generated artifacts
```

An experiment name and phase must match across `examples/`, `cluster/`, `docs/`,
and `exps/`. Generated checkpoints, logs, W&B files, and evaluation results do
not belong inside source packages.

## 5. Naming and public API rules

- Use `_common`, not `common`, for a directory that is internal to one
  extension axis. Prefer descriptive module names inside it.
- Use plural names for collections of implementations (`envs`, `models`,
  `algorithms`, `backends`) and keep an existing public singular package such
  as `harness` stable unless a deliberate API migration is planned.
- Scope adapters to what they adapt: `envs/_common/gymnasium_adapter.py` or
  `rollout/client.py` is clearer than a repository-level `adapter/` package.
- Keep public exports small and intentional. New cross-package imports should
  work through the axis facade, for example `from vagen_agent.envs import ...`.
- Keep class, registry, configuration, CLI entry-point, and dynamic import
  names synchronized when moving a module.

## 6. Refactor and synchronization checklist

When adding or moving an extension implementation:

1. Classify it as invariant core, shared axis code, or one concrete
   implementation.
2. Move it to the corresponding fixed module, `_common/`, or implementation
   directory.
3. Add or update the axis facade and registry.
4. Update Python imports, dynamic import strings, YAML/JSON configs, CLI entry
   points, tests, examples, and documentation in the same change.
5. Preserve branch-specific implementations while applying the shared
   structure to maintained checkouts; synchronization is not permission to
   reset a branch to another branch's feature set.
6. Run import/registry smoke tests, focused unit tests, lint for undefined or
   stale imports, and `git diff --check`.
7. Remove generated caches before committing and verify that run artifacts
   remain ignored.

The result should make two questions obvious from the path alone: which axis
owns the code, and whether the code is a shared contract or one selectable
implementation.
