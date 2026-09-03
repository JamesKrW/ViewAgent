# Pinned LLaMA-Factory submodule

`GraphRL/LLaMA-Factory` is pinned to `JamesKrW/LLaMA-Factory` commit
`d6bb97dd` on the `graphrl` branch (LLaMA-Factory `0.9.6.dev0`).

This replaces the earlier pin at `0a3ddaef` on the `viewagent-submodule`
branch, which was a snapshot of `f80e15db` plus five patched files under
`src/llamafactory/data/`. Those five files carried no ViewAgent-specific
logic — they were upstream compatibility fixes (transformers 5.x
`mm_token_type_ids` rope arguments, the `audio_processor` feature-extractor
fallback, the `qwen3_5` template and tool-call utilities, MiniCPM-V video
sizing, and `read_cloud_json` returning a list). Every one of them has since
landed on `graphrl`, so the new pin is a strict superset and no patch is
carried on top of it.

The SFT stage imports this tree through `PYTHONPATH`
(`GraphRL/LLaMA-Factory/src`), not an installed distribution, so the pinned
checkout is what actually runs.

Initialize it together with VAGEN-SLIME and nested SLIME using:

```bash
git submodule update --init --recursive
```

The `viewagent-submodule` branch is still present in the submodule's local
clone and on the remote, so `git checkout viewagent-submodule` inside
`GraphRL/LLaMA-Factory` restores the previous pin.
