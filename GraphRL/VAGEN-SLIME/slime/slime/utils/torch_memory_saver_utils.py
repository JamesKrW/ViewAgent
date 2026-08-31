from contextlib import contextmanager
from pathlib import Path


_DEFAULT_TAG = "default"


def resolve_preload_library() -> str:
    """Return the torch-memory-saver preload library for this Torch CUDA build."""
    try:
        from torch_memory_saver.utils import get_binary_path_from_package

        return str(get_binary_path_from_package("torch_memory_saver_hook_mode_preload"))
    except (ImportError, RuntimeError):
        # Compatibility with the older slime fork, which shipped one unsuffixed
        # preload library and did not expose get_binary_path_from_package().
        import torch
        import torch_memory_saver

        package_root = Path(torch_memory_saver.__file__).resolve().parent.parent
        cuda_major = str(torch.version.cuda).split(".", 1)[0] if torch.version.cuda else None
        names = []
        if cuda_major:
            names.append(f"torch_memory_saver_hook_mode_preload_cu{cuda_major}.abi3.so")
        names.append("torch_memory_saver_hook_mode_preload.abi3.so")
        for name in names:
            path = package_root / name
            if path.exists():
                return str(path)
        raise FileNotFoundError(
            "Cannot find a torch_memory_saver preload library compatible with "
            f"torch CUDA {torch.version.cuda}."
        )


@contextmanager
def _nested_region_config(impl, tag: str, enable_cpu_backup: bool):
    """Temporarily override a TMS region, including an active preload root region."""
    cdll = impl._binary_wrapper.cdll
    was_active = bool(cdll.tms_get_interesting_region())
    original_enable_cpu_backup = bool(cdll.tms_get_enable_cpu_backup())
    tag_stack = getattr(impl, "_slime_region_tag_stack", None)
    if tag_stack is None:
        tag_stack = []
        impl._slime_region_tag_stack = tag_stack
    previous_tag = tag_stack[-1] if tag_stack else _DEFAULT_TAG

    impl._binary_wrapper.set_config(
        tag=tag,
        interesting_region=True,
        enable_cpu_backup=enable_cpu_backup,
    )
    tag_stack.append(tag)
    try:
        yield
    finally:
        if not cdll.tms_get_interesting_region():
            raise RuntimeError("torch_memory_saver region was unexpectedly disabled")
        tag_stack.pop()
        impl._binary_wrapper.set_config(
            tag=previous_tag,
            interesting_region=was_active,
            enable_cpu_backup=original_enable_cpu_backup,
        )


def patch_nested_regions() -> None:
    """Allow nested TMS regions used by slime's patched Megatron allocator.

    The slime launcher enables a process-wide preload region so model weights
    can be restored after offload.  Its Megatron patch then creates nested
    no-CPU-backup regions for disposable parameter and gradient buffers.  The
    PyPI TMS release rejects that valid nesting, while slime's TMS fork accepts
    it.  Patch only this context-manager behavior when the fork is unavailable.
    """
    from torch_memory_saver.entrypoint import _TorchMemorySaverImpl

    if getattr(_TorchMemorySaverImpl, "_slime_nested_regions", False):
        return
    _TorchMemorySaverImpl._with_region_config = _nested_region_config
    _TorchMemorySaverImpl._slime_nested_regions = True
