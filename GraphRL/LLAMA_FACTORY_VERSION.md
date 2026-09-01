# Pinned LLaMA-Factory submodule

`GraphRL/LLaMA-Factory` is pinned to the ViewAgent fork at
`0a3ddaef4cb8c289c0f4238901d6ed477958ff47` on the
`viewagent-submodule` branch. Its tree is byte-for-byte identical to the
LLaMA-Factory source that was previously stored directly in this repository.

The snapshot is based on `JamesKrW/LLaMA-Factory` commit
`f80e15dbb41cafc3a6f662aa520f40e596a41997` and retains the ViewAgent-specific
multimodal data and trainer compatibility changes.

Initialize it together with VAGEN-SLIME and nested SLIME using:

```bash
git submodule update --init --recursive
```
